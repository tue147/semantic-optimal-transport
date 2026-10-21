"""Unit tests for the feature-matching seed-aggregation helpers in exp_config.py.

These back `feature_matching_test.py`'s `*_seedagg.json` details log: the run-file naming (which
decides *which* files `--backfill` picks up) and the per-run metric extraction. The statistics
themselves are `exp_config.summarize_stats`, covered by tests/test_seed_agg.py. The helpers live in
exp_config so this test stays light -- importing feature_matching_test would pull in
Evaluation_Utils and trigger the HuggingFace login. Run with the project conda env:

    python -m pytest tests/test_matching_agg.py -q
    # or, without pytest:
    python tests/test_matching_agg.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exp_config as ec


def _args(**kw):
    base = dict(sae_a_id=25, sae_b_id=12, top=1, number_of_pairs=100,
                min_activation_examples=30, run_prefix="run")
    base.update(kw)
    return argparse.Namespace(**base)


def _match(name, **kw):
    return ec.matching_run_file_regex(_args(**kw)).match(name)


def test_run_file_name_round_trips_through_the_regex():
    name = ec.matching_run_file_name(_args(), "mech_permute_ml", 4)
    assert name == "mech_permute_ml_25_12_1_100_30_run4.pt"
    m = _match(name)
    assert m.group("method") == "mech_permute_ml" and m.group("seed") == "4"


def test_seedagg_file_name():
    assert ec.matching_seedagg_file_name(_args()) == "matching_25_12_1_100_30_seedagg.json"


def test_regex_accepts_the_old_untagged_run():
    m = _match("ot_25_12_1_100_30.pt")
    assert m and m.group("method") == "ot" and m.group("seed") is None


def test_regex_rejects_other_hyperparameters():
    # different layer pair, different filename layout, different top-k: none may be collected
    assert _match("ot_13_12_1_100_30_run1.pt") is None
    assert _match("ot_25_12_25_25_euclidean_500_0.1.pt") is None
    assert _match("otscale_25_12_100_30_5.pt") is None
    assert _match("ot_25_12_5_100_30_run1.pt") is None


def test_regex_honours_run_prefix():
    assert _match("ot_25_12_1_100_30_seed2.pt") is None
    assert _match("ot_25_12_1_100_30_seed2.pt", run_prefix="seed").group("seed") == "2"


def _results(scores, skipping=None):
    res = {"llm_eval": {"llm_eval": {
        "overall_score": sum(scores) / len(scores),
        "total_score": sum(scores),
        "total_predictions": len(scores),
        "matching_details": [{"feature_pair": (i, 100 + i), "score": s, "raw_response": ""}
                             for i, s in enumerate(scores)],
    }}}
    if skipping is not None:
        res["skipping"] = {"skipping_effect": skipping[0], "var_ex_effect": skipping[1]}
    return res


def test_run_metrics_accuracy_is_the_score3_fraction():
    rec = ec.matching_run_metrics(_results([3, 3, 2, 1]))
    assert rec["n_pairs"] == 4
    assert rec["accuracy"] == 0.5 == rec["frac3"]
    assert rec["frac2"] == 0.25 and rec["frac1"] == 0.25
    assert rec["n3"] == 2 and rec["n2"] == 1 and rec["n1"] == 1
    assert abs(rec["llm_score"] - 2.25) < 1e-9
    assert "pairs" not in rec


def test_run_metrics_records_pairs_on_request():
    rec = ec.matching_run_metrics(_results([3, 1]), with_pairs=True)
    assert rec["pairs"] == [[0, 100, 3], [1, 101, 1]]


def test_run_metrics_skipping_is_optional():
    assert "skipping" not in ec.matching_run_metrics(_results([1, 2]))
    rec = ec.matching_run_metrics(_results([1, 2], skipping=(2.748, 0.911)))
    assert rec["skipping"] == 2.748 and rec["var_ex"] == 0.911


def test_summarize_seeds_is_per_metric():
    """A seed missing CE/VE still contributes its LLM numbers; each metric keeps its own n."""
    per_seed = {
        "1": ec.matching_run_metrics(_results([3, 1], skipping=(2.0, 0.9))),
        "2": ec.matching_run_metrics(_results([3, 3])),
    }
    stats = ec.summarize_matching_seeds(per_seed)
    assert stats["llm_score"]["n"] == 2 and abs(stats["llm_score"]["mean"] - 2.5) < 1e-9
    assert stats["skipping"]["n"] == 1 and stats["var_ex"]["n"] == 1


def test_format_stat_hides_fake_spread_for_a_single_run():
    assert ec.format_stat(None, "{:.3f}") == "-"
    assert ec.format_stat(ec.summarize_stats([1.5]), "{:.3f}") == "1.500 (n=1)"
    assert ec.format_stat(ec.summarize_stats([1.0, 2.0]), "{:.3f}").startswith("1.500 ± 0.707 [")


def test_format_table_drops_empty_columns_and_lists_seeds():
    per_seed = {"1": ec.matching_run_metrics(_results([3, 1])),
                "2": ec.matching_run_metrics(_results([3, 3]))}
    methods = {"ot": {"n_runs": 2, "per_seed": per_seed,
                      "stats": ec.summarize_matching_seeds(per_seed)}}
    table = ec.format_matching_table(methods)
    assert "LLM score" in table and "CE loss" not in table  # no skipping recorded -> no column
    assert "1:2.00/0.50" in table and "2:3.00/1.00" in table


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
