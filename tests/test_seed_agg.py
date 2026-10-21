"""Unit tests for the pure seed-aggregation helpers used by circuit_compress_eval.py.

These cover the only new *logic* the aggregation adds -- descriptive statistics and per-decision
record extraction. The eval itself (`llm_clustering_eval`) is called unchanged, so it isn't
re-tested here. Run with the project conda env:

    python -m pytest tests/test_seed_agg.py -q
    # or, without pytest:
    python tests/test_seed_agg.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exp_config as ec


def _approx(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"{a} != {b}"


def test_summarize_empty():
    s = ec.summarize_stats([])
    assert s["n"] == 0
    assert s["mean"] is None and s["ci90_low"] is None


def test_summarize_single():
    s = ec.summarize_stats([0.7])
    assert s["n"] == 1
    _approx(s["mean"], 0.7)
    _approx(s["std"], 0.0)
    _approx(s["sem"], 0.0)
    # zero-width CI for a single sample
    _approx(s["ci90_low"], 0.7)
    _approx(s["ci90_high"], 0.7)
    _approx(s["median"], 0.7)


def test_summarize_matches_numpy_and_ci():
    import numpy as np
    vals = [0.2, 0.4, 0.6, 0.9]
    s = ec.summarize_stats(vals)
    arr = np.asarray(vals)
    _approx(s["mean"], float(arr.mean()))
    _approx(s["std"], float(arr.std(ddof=1)))
    _approx(s["sem"], float(arr.std(ddof=1) / math.sqrt(len(vals))))
    _approx(s["min"], 0.2)
    _approx(s["max"], 0.9)
    _approx(s["median"], float(np.median(arr)))
    # CI is symmetric about the mean and non-degenerate for n>1
    _approx((s["ci90_low"] + s["ci90_high"]) / 2, s["mean"])
    assert s["ci90_high"] > s["mean"] > s["ci90_low"]
    # matches t-based half width when scipy is present
    from scipy import stats as st
    half = float(st.t.ppf(0.95, len(vals) - 1)) * s["sem"]
    _approx(s["ci90_high"] - s["mean"], half)


def test_extract_decisions():
    cluster_details = [
        {"cluster_id": 0, "predictions": [True, False, True], "ground_truth": [True, True, True]},
        {"cluster_id": 1, "predictions": [False], "ground_truth": [False]},
        {"cluster_id": 2, "predictions": [], "ground_truth": []},  # skipped/empty
    ]
    decisions = ec.extract_decisions(cluster_details)
    assert len(decisions) == 4  # 3 + 1 + 0
    # correctness matches predicted == ground_truth
    correct = [d for d in decisions if d["correct"]]
    assert len(correct) == 3  # (T==T), (T==T) for cluster0 idx0/2; cluster1 (F==F). idx1 T!=T? no: F vs T -> wrong
    # explicit check of the wrong one
    wrong = [d for d in decisions if not d["correct"]]
    assert len(wrong) == 1
    assert wrong[0] == {"cluster_id": 0, "feature_index": 1,
                        "predicted": False, "ground_truth": True, "correct": False}
    # types are plain python (json-serializable)
    for d in decisions:
        assert isinstance(d["predicted"], bool) and isinstance(d["ground_truth"], bool)


def test_extract_decisions_mismatched_lengths():
    # zip stops at the shorter list; should not raise
    cd = [{"cluster_id": 5, "predictions": [True, True], "ground_truth": [True]}]
    d = ec.extract_decisions(cd)
    assert len(d) == 1 and d[0]["feature_index"] == 0


def _make_prompt(gidx, inter, intra, per_seed):
    """per_seed: {seed: (overall_accuracy, total_correct, total_predictions)}"""
    return {
        "prompt_idx": gidx, "global_prompt_idx": gidx, "inter": inter, "intra": intra,
        "per_seed": {s: {"overall_accuracy": oa, "total_correct": tc, "total_predictions": tp,
                         "decisions": [], "cluster_details": []}
                     for s, (oa, tc, tp) in per_seed.items()},
    }


def test_aggregate_seed_prompts_skips_ungradable():
    """A prompt whose circuit grades nothing (total_predictions == 0) is excluded from macro, whether
    it carries the new NaN accuracy or an old 0.0 -- it must not drag the mean down."""
    seeds = [1]
    for undefined in (float("nan"), 0.0):
        prompts = [
            _make_prompt(0, 0.1, 0.5, {1: (0.5, 1, 2)}),
            _make_prompt(1, 1.0, 1.0, {1: (undefined, 0, 0)}),
        ]
        agg = ec.aggregate_seed_prompts(prompts, seeds)
        _approx(agg["llm_accuracy"]["macro_by_seed"][1], 0.5)
        _approx(agg["llm_accuracy"]["micro_by_seed"][1], 0.5)


def test_aggregate_seed_prompts_micro_macro():
    seeds = [1, 2]
    prompts = [
        _make_prompt(0, 0.1, 0.5, {1: (0.5, 1, 2), 2: (1.0, 2, 2)}),
        _make_prompt(1, 0.3, 0.7, {1: (1.0, 4, 4), 2: (0.5, 2, 4)}),
    ]
    agg = ec.aggregate_seed_prompts(prompts, seeds)
    # micro seed1 = (1+4)/(2+4)=5/6 ; seed2 = (2+2)/(2+4)=4/6
    _approx(agg["llm_accuracy"]["micro_by_seed"][1], 5 / 6)
    _approx(agg["llm_accuracy"]["micro_by_seed"][2], 4 / 6)
    # macro = mean of per-prompt overall accuracies
    _approx(agg["llm_accuracy"]["macro_by_seed"][1], 0.75)
    _approx(agg["llm_accuracy"]["macro_by_seed"][2], 0.75)
    _approx(agg["llm_accuracy"]["micro_stats"]["mean"], (5 / 6 + 4 / 6) / 2)
    _approx(agg["inter_stats"]["mean"], 0.2)
    _approx(agg["intra_stats"]["mean"], 0.6)


def test_str_seed_keys_roundtrip():
    """After JSON, per_seed keys are strings but config seeds stay ints; aggregation must still
    find them (the merge path relies on this)."""
    import json
    seeds = [1, 2]
    prompts = [_make_prompt(0, 0.1, 0.5, {1: (0.5, 1, 2), 2: (1.0, 2, 2)})]
    reloaded = json.loads(json.dumps(prompts))  # per_seed keys -> "1","2"
    agg = ec.aggregate_seed_prompts(reloaded, seeds)
    _approx(agg["llm_accuracy"]["micro_by_seed"][1], 0.5)
    _approx(agg["llm_accuracy"]["micro_by_seed"][2], 1.0)


def test_merge_invariant():
    """Splitting prompts across disjoint ranges and merging with aggregate_seedagg must reproduce
    the exact statistics of one combined run."""
    import json
    import tempfile

    import aggregate_seedagg as agm

    seeds = [1, 2]
    all_prompts = [
        _make_prompt(0, 0.1, 0.5, {1: (0.5, 1, 2), 2: (1.0, 2, 2)}),
        _make_prompt(1, 0.3, 0.7, {1: (1.0, 4, 4), 2: (0.5, 2, 4)}),
        _make_prompt(2, 0.2, 0.6, {1: (0.75, 3, 4), 2: (0.25, 1, 4)}),
        _make_prompt(3, 0.4, 0.9, {1: (1.0, 2, 2), 2: (0.5, 1, 2)}),
    ]
    combined = ec.aggregate_seed_prompts(all_prompts, seeds)

    def _report(prompts, start, end):
        return {"config": {"model": "m", "sae_type": "mlp", "cluster_type": "naive",
                           "cluster_seed": 0, "n_clusters": 5, "node_threshold": 50,
                           "seeds": seeds, "judge": "llama70b", "judge_model": "x",
                           "data_range": [start, end], "llm": {}},
                "n_prompts": len(prompts), **ec.aggregate_seed_prompts(prompts, seeds),
                "prompts": prompts}

    with tempfile.TemporaryDirectory() as d:
        f1 = os.path.join(d, "naive_50_5_0_2_seedagg.json")
        f2 = os.path.join(d, "naive_50_5_2_4_seedagg.json")
        json.dump(_report(all_prompts[:2], 0, 2), open(f1, "w"))
        json.dump(_report(all_prompts[2:], 2, 4), open(f2, "w"))

        agm.main(["--inputs", f1, f2, "--output_dir", d])
        merged = json.load(open(os.path.join(d, "naive_50_5_0_4_seedagg_merged.json")))

    assert merged["n_prompts"] == 4
    for view in ("micro", "macro"):
        for stat in ("mean", "std", "sem", "ci90_low", "ci90_high", "median"):
            _approx(merged["llm_accuracy"][f"{view}_stats"][stat],
                    combined["llm_accuracy"][f"{view}_stats"][stat])
    for stat in ("mean", "std", "ci90_low", "ci90_high"):
        _approx(merged["inter_stats"][stat], combined["inter_stats"][stat])
        _approx(merged["intra_stats"][stat], combined["intra_stats"][stat])
    # merged config records provenance and the widened range
    assert merged["config"]["data_range"] == [0, 4]
    assert merged["config"]["source_ranges"] == [[0, 2], [2, 4]]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
