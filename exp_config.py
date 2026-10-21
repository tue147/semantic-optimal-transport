"""Shared configuration, CLI arguments and boilerplate for the main experiment scripts.

The six entry scripts (`feature_matching_run.py`, `feature_matching_test.py`,
`circuit_compress_eval.py`, `circuit_behavior_clustering.py`, `modular_feature_group.py`,
`run_addition_matching.py`) used to carry their configuration as a block of commented-out
assignments at the top. Those stanzas are preserved here as named presets so every recorded
experiment stays runnable, and every value is also overridable from the command line.

Defaults are the values that were *active* (uncommented) in each script, so running a script
with no flags reproduces its previous behaviour.

Heavy imports (`Evaluation_Utils`, `sae.*`) happen inside the functions that need them, so
`--help` and argument parsing stay cheap and don't trigger the HuggingFace login in
`Evaluation_Utils`.
"""

import argparse
import os
import re
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch as t
from torch import Tensor

# ---------------------------------------------------------------------------------------------
# Model / SAE presets
# ---------------------------------------------------------------------------------------------
# Each entry is one of the mutually-exclusive `model_name / tok_name / sae_type / dtype /
# layer_range` stanzas that used to be commented out at the top of the entry scripts.

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "gemma_mlp": dict(
        model_name="google/gemma-2-2b",
        tok_name="google/gemma-2-2b",
        sae_type=["mlp"],
        dtype="float16",
        layer_range=None,
    ),
    "gemma_ds": dict(
        model_name="google/gemma-2-2b",
        tok_name="google/gemma-2-2b",
        sae_type=["ds"],
        dtype="float16",
        layer_range=None,
    ),
    "gpt2_ds": dict(
        model_name="gpt2-small",
        tok_name="gpt2",
        sae_type=["ds"],
        dtype="float16",
        layer_range=None,
    ),
    "llama_post": dict(
        model_name="meta-llama/Llama-3.2-1B",
        tok_name="meta-llama/Llama-3.2-1B",
        sae_type=["post"],
        dtype="float16",
        layer_range=[6, 7],
    ),
    "qwen_mlp": dict(
        model_name="qwen3-1.7b",
        tok_name="Qwen/Qwen3-1.7B",
        sae_type=["mlp"],
        dtype="bfloat16",
        layer_range=None,
    ),
}

DTYPE_MAP: Dict[str, t.dtype] = {
    "float16": t.float16,
    "bfloat16": t.bfloat16,
    "float32": t.float32,
    "float64": t.float64,
}


# ---------------------------------------------------------------------------------------------
# argparse helpers
# ---------------------------------------------------------------------------------------------
def str_to_bool(value: Any) -> bool:
    """Same semantics as `sae.Training_Utils.str_to_bool`, duplicated to keep imports light."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def int_or_float(value: Any) -> int | float:
    """Keep `--threshold 1` an int and `--threshold 0.5` a float.

    The pruning thresholds are interpreted as counts when `threshold_type="number"` and as
    fractions when `threshold_type="percen"`, so the int/float distinction is preserved.
    """
    if isinstance(value, (int, float)):
        return value
    return int(value) if value.strip().lstrip("+-").isdigit() else float(value)


def optional_str(value: str) -> str | None:
    """Allow `--foo none` on the command line to mean Python `None`."""
    return None if value.lower() in ("none", "null", "") else value


def set_seed(seed: int | None) -> None:
    """Seed every RNG a run may touch. A `None` seed leaves all RNGs untouched."""
    if seed is None:
        return
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------------------------
# LLM judge presets
# ---------------------------------------------------------------------------------------------
# Both LLM evals (feature_matching_test.py, circuit_compress_eval.py) score with an LLM judge.
# A preset names the model, the endpoint it lives on, and which env var(s) hold that endpoint's
# key. Selected on the command line with `--judge`.
#
#   llama70b / gpt-oss-120b : any OpenAI-compatible endpoint serving these models, set with
#                             JUDGE_BASE_URL (default: the University of Florida gateway the paper
#                             used). Key read from JUDGE_API_KEY, then UF_AI_API_KEY, then
#                             OPENAI_API_KEY. The paper's judges: gpt-oss-120b for feature
#                             matching, Llama-3.3-70B-Instruct for circuit compression.
#   gpt5-mini / gpt5-nano   : OpenAI's public API (base_url=None -> api.openai.com). Key read
#                             from OPENAI_API_KEY. These are *reasoning* models, so the request
#                             uses `max_completion_tokens` + `reasoning_effort` rather than the
#                             `max_tokens` that the chat models take.
UF_GATEWAY_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.ai.it.ufl.edu/v1")

JUDGE_PRESETS: Dict[str, Dict[str, Any]] = {
    "llama70b": dict(
        model="llama-3.3-70b-instruct",
        base_url=UF_GATEWAY_BASE_URL,
        api_key_env=("JUDGE_API_KEY", "UF_AI_API_KEY", "OPENAI_API_KEY"),
        style="chat",
        timeout=100.0,
    ),
    "gpt-oss-120b": dict(
        model="gpt-oss-120b",
        base_url=UF_GATEWAY_BASE_URL,
        api_key_env=("JUDGE_API_KEY", "UF_AI_API_KEY", "OPENAI_API_KEY"),
        style="chat",
        timeout=100.0,
    ),
    "gpt5-mini": dict(
        model="gpt-5-mini",
        base_url=None,
        api_key_env=("OPENAI_API_KEY",),
        style="reasoning",
        reasoning_effort="low",
        timeout=120.0,
    ),
    "gpt5-nano": dict(
        model="gpt-5-nano",
        base_url=None,
        api_key_env=("OPENAI_API_KEY",),
        style="reasoning",
        reasoning_effort="low",
        timeout=120.0,
    ),
    # Self-hosted Llama-3.3-70B served by vLLM on the job's own node (see
    # scripts/circuit_compress_eval.sh). Same weights as the `llama70b` gateway judge (the unsloth
    # mirror), but no API key, no rate limit and no per-call cost, so it is the judge to use for
    # the big seed sweeps. `local=True` lets the launcher move the endpoint: VLLM_JUDGE_URL /
    # VLLM_JUDGE_MODEL override the defaults below at resolve time, so the sbatch owns the port.
    # The timeout is long because a 70B on one GPU queues requests instead of rate-limiting them.
    "llama70b-vllm": dict(
        model="unsloth/Llama-3.3-70B-Instruct",
        base_url="http://127.0.0.1:8010/v1",
        api_key_env=("VLLM_JUDGE_API_KEY",),
        api_key_default="EMPTY",   # vLLM does not check the key unless started with --api-key
        style="chat",
        timeout=600.0,
        local=True,
    ),
}

DEFAULT_JUDGE = "gpt5-mini"


def add_judge_args(parser: argparse.ArgumentParser, default_judge: str = DEFAULT_JUDGE) -> None:
    """CLI selector for the LLM judge, shared by the two eval entry scripts."""
    group = parser.add_argument_group("LLM judge")
    group.add_argument("--judge", type=str, default=default_judge, choices=sorted(JUDGE_PRESETS.keys()),
                       help="Which LLM scores the eval. gpt5-* hit OpenAI (OPENAI_API_KEY); "
                            "llama70b / gpt-oss-120b hit the UF gateway (UF_AI_API_KEY).")
    group.add_argument("--judge_effort", type=str, default=None,
                       choices=["minimal", "low", "medium", "high"],
                       help="Reasoning effort for gpt5-* judges (ignored by the chat judges); "
                            "defaults to the preset value (low).")


def resolve_judge(name: str, effort: str | None = None) -> Dict[str, Any]:
    """Return a copy of the judge preset with the API key resolved from the environment.

    Raises if no key is set for the chosen endpoint, so a misconfigured run fails before it
    loads a model rather than after fanning out API calls.
    """
    if name not in JUDGE_PRESETS:
        raise ValueError(f"Unknown judge {name!r}; choices: {sorted(JUDGE_PRESETS)}")
    cfg = dict(JUDGE_PRESETS[name])
    if cfg.get("local"):
        # The launcher that started the server decides the port and the served model name.
        cfg["base_url"] = os.getenv("VLLM_JUDGE_URL", cfg["base_url"])
        cfg["model"] = os.getenv("VLLM_JUDGE_MODEL", cfg["model"])
    api_key = None
    for env_name in cfg["api_key_env"]:
        api_key = os.getenv(env_name)
        if api_key:
            cfg["api_key_env_used"] = env_name
            break
    if not api_key and cfg.get("api_key_default"):
        api_key = cfg["api_key_default"]          # self-hosted endpoint: no real key exists
        cfg["api_key_env_used"] = "preset default"
    if not api_key:
        raise RuntimeError(
            f"No API key for judge {name!r}: set one of {cfg['api_key_env']} in the environment / .env"
        )
    cfg["api_key"] = api_key
    if effort is not None and cfg.get("style") == "reasoning":
        cfg["reasoning_effort"] = effort
    cfg["name"] = name
    print(f"[judge] {name} -> model={cfg['model']} endpoint={cfg['base_url'] or 'api.openai.com'} "
          f"key={cfg['api_key_env_used']}")
    return cfg


# ---------------------------------------------------------------------------------------------
# Seed-aggregation statistics (circuit_compress_eval.py --aggregate_seeds)
# ---------------------------------------------------------------------------------------------
def summarize_stats(values: Sequence[float]) -> Dict[str, Any]:
    """Descriptive stats for a list of scalars: n, mean, sample std (ddof=1), sem, two-sided 90%
    CI, min, max, median. CI uses a t critical value when n>1 (falls back to z=1.645 if SciPy is
    unavailable). Returns None fields for an empty input rather than raising."""
    import numpy as np

    arr = np.asarray([float(v) for v in values], dtype=float)
    n = int(arr.size)
    if n == 0:
        return {k: None for k in ("mean", "std", "sem", "ci90_low", "ci90_high",
                                  "min", "max", "median")} | {"n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 1 else 0.0
    if n > 1:
        try:
            from scipy import stats as _stats
            tcrit = float(_stats.t.ppf(0.95, n - 1))  # 0.05 in each tail -> two-sided 90%
        except Exception:
            tcrit = 1.645
        half = tcrit * sem
    else:
        half = 0.0
    return {
        "n": n, "mean": mean, "std": std, "sem": sem,
        "ci90_low": mean - half, "ci90_high": mean + half,
        "min": float(arr.min()), "max": float(arr.max()), "median": float(np.median(arr)),
    }


def _get_seed_entry(per_seed: Dict[Any, Any], seed: Any) -> Any:
    """Look up a seed in a per_seed dict tolerating int vs str keys (JSON stringifies int keys)."""
    if seed in per_seed:
        return per_seed[seed]
    return per_seed.get(str(seed), per_seed.get(int(seed) if str(seed).lstrip("-").isdigit() else seed))


def aggregate_seed_prompts(prompts: Sequence[Dict[str, Any]], seeds: Sequence[Any]) -> Dict[str, Any]:
    """Compute the seed-aggregated statistics from a list of per-prompt records.

    Each record is ``{"inter": float|None, "intra": float|None, "per_seed": {seed: {overall_accuracy,
    total_correct, total_predictions, ...}}}``. Two accuracy views across seeds (seed = unit of
    variation): ``micro`` pools every decision within a seed (sum correct / sum predictions across
    prompts); ``macro`` averages the per-prompt overall accuracies within a seed. Each gets full
    descriptive stats. inter/intra are clustering-only, aggregated across prompts.

    This is the single source of truth for the stats, shared by circuit_compress_eval.py's
    ``--aggregate_seeds`` writer and aggregate_seedagg.py's merger, so a merge of disjoint ranges
    recomputes exactly what a single combined run would.
    """
    import numpy as np

    per_seed_micro: Dict[Any, float] = {}
    per_seed_macro: Dict[Any, float] = {}
    for seed in seeds:
        tc = tp = 0
        overalls: List[float] = []
        for prec in prompts:
            ps = _get_seed_entry(prec.get("per_seed", {}), seed)
            if ps is None:
                continue
            tc += int(ps["total_correct"])
            tp += int(ps["total_predictions"])
            if int(ps["total_predictions"]) > 0:  # ungradable prompts (nothing judged) don't count as 0
                overalls.append(float(ps["overall_accuracy"]))
        per_seed_micro[seed] = (tc / tp) if tp > 0 else 0.0
        per_seed_macro[seed] = float(np.mean(overalls)) if overalls else 0.0

    inter = [p["inter"] for p in prompts if p.get("inter") is not None]
    intra = [p["intra"] for p in prompts if p.get("intra") is not None]

    return {
        "llm_accuracy": {
            "micro_by_seed": per_seed_micro,
            "micro_stats": summarize_stats([per_seed_micro[s] for s in seeds]),
            "macro_by_seed": per_seed_macro,
            "macro_stats": summarize_stats([per_seed_macro[s] for s in seeds]),
        },
        "inter_stats": summarize_stats(inter),
        "intra_stats": summarize_stats(intra),
    }


# ---------------------------------------------------------------------------------------------
# Feature-matching seed aggregation (feature_matching_test.py)
# ---------------------------------------------------------------------------------------------
# feature_matching_test.py stores one scalar per metric per seed, so the spread only exists across
# runs. These helpers turn one run's result dict into a flat record and summarize the records of a
# method across seeds; the script writes both into its `*_seedagg.json` details log.

# Metrics reported per method, in table order: key -> (header, format spec).
MATCHING_METRIC_SPECS: List[Tuple[str, str, str]] = [
    ("llm_score", "LLM score", "{:.3f}"),
    ("accuracy", "Acc", "{:.3f}"),  # == frac3; the score-3 fraction is the reported accuracy
    ("frac2", "frac2", "{:.3f}"),
    ("frac1", "frac1", "{:.3f}"),
    # CE/VE barely move between methods (the patch touches one feature), so they need the extra
    # decimals to be readable at all -- at 4 dp every std rounds to 0.0000.
    ("skipping", "CE loss", "{:.5f}"),
    ("var_ex", "VE", "{:.5f}"),
]


def matching_run_stem(args: argparse.Namespace) -> str:
    """The hyperparameter part of a run file name: `_{a}_{b}_{top}_{n_pairs}_{min_act}`."""
    return (f"_{args.sae_a_id}_{args.sae_b_id}_{args.top}_"
            f"{args.number_of_pairs}_{args.min_activation_examples}")


def matching_run_file_name(args: argparse.Namespace, method_name: str, seed: Any) -> str:
    """`{method}_{a}_{b}_{top}_{n_pairs}_{min_act}_run{seed}.pt` -- one seed of one method."""
    return f"{method_name}{matching_run_stem(args)}_{args.run_prefix}{seed}.pt"


def matching_run_file_regex(args: argparse.Namespace) -> "re.Pattern":
    """Match a run file name and capture `method` + `seed` (seed is None for the old untagged name).

    The stem is anchored, so sibling files sharing a method prefix but using a different layout
    (`ot_25_12_25_25_euclidean_500_0.1.pt`, `otscale_...`, another layer pair) can't be swept in.
    """
    return re.compile(rf"^(?P<method>.+){matching_run_stem(args)}"
                      rf"(?:_{re.escape(args.run_prefix)}(?P<seed>\d+))?\.pt$")


def matching_seedagg_file_name(args: argparse.Namespace) -> str:
    """The one details log per layer pair, written next to the per-seed run files."""
    return f"matching{matching_run_stem(args)}_seedagg.json"


def matching_run_metrics(results: Dict[str, Any], with_pairs: bool = False) -> Dict[str, Any]:
    """Flatten one seed's `{llm_eval: ..., skipping: ...}` result dict into JSON-safe scalars.

    `llm_score` is the judge's own `overall_score`; the score distribution is recomputed from
    `matching_details` (score 3 = correct, the accuracy convention `test.py` uses). `skipping`
    (layer-skipping CE) and `var_ex` are included when the caller passes a `skipping` block.
    `with_pairs` also stores every judged `[feat_a, feat_b, score]`, so the JSON is enough for
    re-analysis without loading the `.pt`.
    """
    llm = results["llm_eval"]["llm_eval"]
    details = llm.get("matching_details", []) or []
    scores = [int(d["score"]) for d in details]
    n = len(scores)

    out: Dict[str, Any] = {
        "llm_score": float(llm["overall_score"]),
        "n_pairs": n,
        "total_score": llm.get("total_score"),
        "total_predictions": llm.get("total_predictions"),
    }
    for s in (3, 2, 1):
        out[f"n{s}"] = sum(1 for v in scores if v == s)
        out[f"frac{s}"] = (out[f"n{s}"] / n) if n else None
    out["accuracy"] = out["frac3"]

    skip = results.get("skipping")
    if skip is not None:
        out["skipping"] = float(skip["skipping_effect"])
        out["var_ex"] = float(skip["var_ex_effect"])

    if with_pairs:
        out["pairs"] = [[int(d["feature_pair"][0]), int(d["feature_pair"][1]), int(d["score"])]
                        for d in details]
    return out


def summarize_matching_seeds(per_seed: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    """`{metric: summarize_stats(values across seeds)}` for every metric that was recorded.

    Metrics are summarized independently, so a run whose CE/VE is missing still contributes its
    LLM numbers (and the `n` of each metric shows how many seeds actually carried it).
    """
    stats: Dict[str, Any] = {}
    for key, _, _ in MATCHING_METRIC_SPECS:
        values = [r[key] for r in per_seed.values() if r.get(key) is not None]
        if values:
            stats[key] = summarize_stats(values)
    return stats


def format_stat(s: Dict[str, Any] | None, fmt: str) -> str:
    """`mean ± std [ci90_low, ci90_high]`, or a dash when the metric was never recorded.

    A single run has no spread, so it prints the bare mean rather than a fake `± 0.000 [m, m]`.
    """
    if not s or s.get("mean") is None:
        return "-"
    f = fmt.format
    if s["n"] < 2:
        return f"{f(s['mean'])} (n=1)"
    return f"{f(s['mean'])} ± {f(s['std'])} [{f(s['ci90_low'])}, {f(s['ci90_high'])}]"


def format_matching_table(methods: Dict[str, Dict[str, Any]]) -> str:
    """One row per method, cells `mean ± std [90% CI]`, plus the per-seed score/accuracy lines.

    `methods` is the `{name: {n_runs, per_seed, stats}}` block of the seedagg JSON. Columns with no
    data in any method (e.g. CE/VE on runs that never recorded them) are dropped.
    """
    if not methods:
        return "(no runs)"
    present = [(k, h, f) for k, h, f in MATCHING_METRIC_SPECS
               if any(k in m.get("stats", {}) for m in methods.values())]

    headers = ["method", "n"] + [h for _, h, _ in present]
    rows = [[name, str(m.get("n_runs", len(m.get("per_seed", {}))))] +
            [format_stat(m.get("stats", {}).get(k), f) for k, _, f in present]
            for name, m in methods.items()]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.append("-" * len(lines[0]))
    lines += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    lines.append("")
    lines.append("(cells are mean ± std [90% CI] across seeds)")
    lines.append("per-seed LLM score / accuracy:")
    for name, m in methods.items():
        vals = ", ".join(
            f"{seed}:{r['llm_score']:.2f}/{r['accuracy']:.2f}" if r.get("accuracy") is not None
            else f"{seed}:{r['llm_score']:.2f}"
            for seed, r in m.get("per_seed", {}).items()
        )
        lines.append(f"  {name:<18} {vals}")
    return "\n".join(lines)


def extract_decisions(cluster_details: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten the per-cluster LLM-eval output into one record per feature decision:
    cluster_id, feature_index, predicted, ground_truth, correct. Reads the `predictions` /
    `ground_truth` lists that `llm_clustering_eval` already produces -- it does not re-derive
    anything, so the eval logic is untouched."""
    decisions: List[Dict[str, Any]] = []
    for cd in cluster_details:
        preds = cd.get("predictions", []) or []
        gts = cd.get("ground_truth", []) or []
        for i, (p, g) in enumerate(zip(preds, gts)):
            decisions.append({
                "cluster_id": cd.get("cluster_id"),
                "feature_index": i,
                "predicted": bool(p),
                "ground_truth": bool(g),
                "correct": bool(p == g),
            })
    return decisions


# ---------------------------------------------------------------------------------------------
# Model / SAE arguments
# ---------------------------------------------------------------------------------------------
def add_model_args(
    parser: argparse.ArgumentParser,
    default_preset: str = "gemma_mlp",
    default_device_id: int = 0,
) -> argparse.ArgumentParser:
    """Model, SAE and device selection. Individual flags override the chosen preset."""
    group = parser.add_argument_group("model / SAE")
    group.add_argument(
        "--preset",
        type=str,
        default=default_preset,
        choices=sorted(MODEL_PRESETS.keys()),
        help="Named model/SAE stanza to start from.",
    )
    group.add_argument("--model_name", type=str, default=None, help="Override the preset model name.")
    group.add_argument("--tok_name", type=str, default=None, help="Override the preset tokenizer name.")
    group.add_argument(
        "--sae_type", type=str, nargs="+", default=None, help="Override the preset SAE types, e.g. mlp."
    )
    group.add_argument(
        "--dtype", type=str, default=None, choices=sorted(DTYPE_MAP.keys()), help="Override the preset dtype."
    )
    group.add_argument(
        "--layer_range",
        type=int,
        nargs="*",
        default=None,
        help="Override the preset layer range. Pass with no values to mean all layers.",
    )
    group.add_argument(
        "--device_id", type=int, default=default_device_id, help="CUDA device index; falls back to CPU."
    )
    return parser


class ModelConfig:
    """Resolved model/SAE settings plus the `*_flat` strings used to build file names."""

    def __init__(self, args: argparse.Namespace):
        preset = MODEL_PRESETS[args.preset]

        self.preset: str = args.preset
        self.model_name: str = args.model_name if args.model_name is not None else preset["model_name"]
        self.tok_name: str = args.tok_name if args.tok_name is not None else preset["tok_name"]
        self.sae_type: List[str] = list(args.sae_type) if args.sae_type is not None else list(preset["sae_type"])

        dtype_name: str = args.dtype if args.dtype is not None else preset["dtype"]
        self.dtype: t.dtype = DTYPE_MAP[dtype_name]
        self.dtype_name: str = dtype_name

        if args.layer_range is None:
            self.layer_range: List[int] | None = preset["layer_range"]
        else:
            # An explicitly empty `--layer_range` means "all layers", which load_saes spells None.
            self.layer_range = list(args.layer_range) if len(args.layer_range) > 0 else None

        self.device: t.device = t.device(f"cuda:{args.device_id}" if t.cuda.is_available() else "cpu")

        self.model_name_flat: str = self.model_name.replace("/", "_")
        self.sae_type_flat: str = "_".join(self.sae_type)

    @property
    def results_dir(self) -> str:
        return f"results/{self.model_name_flat}_{self.sae_type_flat}"

    def results_dir_for_judge(self, judge_name: str) -> str:
        """Judge-scoped results dir: `results/<model>_<sae>_<judge>`. The LLM evals write here so
        different judges don't overwrite each other; the run file names inside are unchanged. Judge
        names are already filesystem-safe (e.g. `gpt5-mini`, `llama70b`)."""
        return f"{self.results_dir}_{judge_name}"

    def describe(self) -> str:
        return (
            f"preset={self.preset} model={self.model_name} tok={self.tok_name} "
            f"sae_type={self.sae_type} dtype={self.dtype_name} layer_range={self.layer_range} "
            f"device={self.device}"
        )


def load_model_and_saes(cfg: ModelConfig):
    """`load_saes` with the resolved config. Returns `(model, dict_saes, list_saes)`."""
    from Evaluation_Utils import load_saes

    print(f"[exp_config] loading model and SAEs: {cfg.describe()}")
    return load_saes(
        model_name=cfg.model_name,
        sae_types=cfg.sae_type,
        device=cfg.device,
        dtype=cfg.dtype,
        layer_range=cfg.layer_range,
    )


# ---------------------------------------------------------------------------------------------
# Pile activation cache
# ---------------------------------------------------------------------------------------------
def add_data_args(
    parser: argparse.ArgumentParser,
    default_batchsize: int = 64,
    default_max_length: int = 128,
    default_num_batches: int = 10000,
) -> argparse.ArgumentParser:
    group = parser.add_argument_group("data / activation cache")
    group.add_argument("--data_path", type=str, default="monology/pile-uncopyrighted", help="HF dataset for activations.")
    group.add_argument("--batchsize", type=int, default=default_batchsize, help="Batch size for activation extraction.")
    group.add_argument("--max_length", type=int, default=default_max_length, help="Token length per sequence.")
    group.add_argument("--num_batches", type=int, default=default_num_batches, help="Number of sequences to extract.")
    group.add_argument(
        "--cache_dir", type=str, default="data/cache_data", help="Directory holding the cached full_data_*.pt files."
    )
    group.add_argument(
        "--full_data_cache",
        type=str,
        default=None,
        help="Explicit path to the full_data cache; derived from --cache_dir when omitted.",
    )
    return parser


def full_data_cache_path(args: argparse.Namespace, cfg: ModelConfig) -> str:
    if args.full_data_cache is not None:
        return args.full_data_cache
    return os.path.join(
        args.cache_dir,
        f"full_data_{cfg.model_name_flat}_{cfg.sae_type_flat}_{args.max_length}_{args.num_batches}.pt",
    )


def load_or_build_full_data(
    args: argparse.Namespace,
    cfg: ModelConfig,
    model,
    list_saes,
    cache_path: str | None = None,
):
    """Load the cached `(data_toks, full_data_dict)` pair, extracting and saving it when absent.

    Caches are keyed by file name and reused via `os.path.exists`; changing the path invalidates
    hours of activation extraction, so `--full_data_cache` exists to point at an existing file.
    """
    from Evaluation_Utils import extract_data, get_sae_activation_from_tok
    from sae.Load_Data import load_data
    from sae.SAE_Trainer import DataConfig

    cache_path = cache_path if cache_path is not None else full_data_cache_path(args, cfg)

    if not os.path.exists(cache_path):
        print(f"[exp_config] no cache at {cache_path}; extracting activations")
        data_config = DataConfig(
            data_path=args.data_path,
            batch_size=1,
            max_length=args.max_length,
        )
        _, data_loader = load_data(cfg.tok_name, data_config)
        data_toks = extract_data(data_loader, num_batches=args.num_batches)

        full_data_dict = get_sae_activation_from_tok(
            {sae.cfg.hook_name: sae for sae in list_saes},
            model,
            data_toks,
            args.batchsize,
            None,
            False,
        )
        ensure_parent_dir(cache_path)
        t.save((data_toks, full_data_dict), cache_path)
    else:
        print(f"[exp_config] loading cached activations from {cache_path}")
        data_toks, full_data_dict = t.load(cache_path, map_location="cpu")

    return data_toks, full_data_dict


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------------------------
# Circuit construction / pruning
# ---------------------------------------------------------------------------------------------
def add_circuit_args(
    parser: argparse.ArgumentParser,
    default_node_threshold: int = 50,
    include_edge_pruning: bool = True,
) -> argparse.ArgumentParser:
    """Graph-construction and pruning hyperparameters shared by the three circuit scripts."""
    group = parser.add_argument_group("circuit / pruning")
    group.add_argument("--method_name", type=str, default="fc", help="Feature graph type passed to load_graph.")
    group.add_argument("--use_error_term", type=str_to_bool, default=True)
    group.add_argument("--pass_through_grad", type=str_to_bool, default=True)
    group.add_argument("--reverse_pruning", type=str_to_bool, default=False)
    group.add_argument("--gradient_mode", type=str, default="standard")
    group.add_argument("--edge_gradient_mode", type=str, default="gradient")
    group.add_argument("--ig_steps", type=int, default=10, help="Integrated-gradients steps.")
    group.add_argument("--transfer_grad", type=str_to_bool, default=True)
    group.add_argument("--save_node_grads", type=str_to_bool, default=False)

    group.add_argument("--prune_type", type=str, default="attrib")
    group.add_argument("--scoring_mode", type=str, default="less")

    group.add_argument("--node_threshold_type", type=str, default="number", choices=["value", "percen", "number"])
    group.add_argument("--node_threshold", type=int_or_float, default=default_node_threshold)

    if include_edge_pruning:
        group.add_argument("--cut_mode", type=str, default="edge")
        group.add_argument("--threshold_type", type=str, default="percen", choices=["value", "percen", "number"])
        group.add_argument("--threshold", type=int_or_float, default=1, help="1 with percen keeps all edges.")

    group.add_argument(
        "--direct_effect",
        type=str_to_bool,
        default=False,
        help="False = total effect, which is what the causal clustering uses.",
    )

    group.add_argument(
        "--build_connection_token_wise", type=str_to_bool, default=True, help="build_default_connection(token_wise=)"
    )
    group.add_argument("--build_connection_inter", type=str_to_bool, default=True)
    group.add_argument(
        "--build_connection_intra",
        type=str_to_bool,
        default=True,
        help="Only the causal clustering actually needs intra connections.",
    )
    return parser


def make_metrics(important_token_idx: Tensor) -> Tuple[Callable[..., Tensor], Callable[..., List[Tuple[int, int]]]]:
    """Build the `(metrics, metrics_top_tokens_and_pos)` pair the pruner and visualiser need.

    Identical to the closures that were redefined inline in each circuit script.
    """

    def metrics(
        logit: Tensor,
        k: int = 1,
        important_token_idx: Tensor = important_token_idx,
    ) -> Tensor:
        # positions in logit[:, :-1] that correspond to important_token_idx (which are positions in clean_token)
        pos = important_token_idx - 1  # shape [k]
        vocab_logits = logit[:, :-1][0, pos, :]  # shape [k, d_vocab]

        top_vals, top_ids = t.topk(vocab_logits, k=k, dim=-1)
        return top_vals.sum()

    def metrics_top_tokens_and_pos(
        logit: Tensor,
        k: int = 1,
        important_token_idx: Tensor = important_token_idx,
    ):
        pos = important_token_idx - 1  # shape [k]
        vocab_logits = logit[:, :-1][0, pos, :]  # shape [k, d_vocab]

        top_vals, top_ids = t.topk(vocab_logits, k=k, dim=-1)  # [k, 5], [k, 5]

        return [(int(p.item()), int(idx.item())) for i, p in enumerate(pos) for idx in top_ids[i]]

    return metrics, metrics_top_tokens_and_pos


def build_pruner(fg, metrics, device, args: argparse.Namespace, seq_len: int):
    """Wire up the default connections for a prompt of `seq_len` tokens and return a Pruner."""
    from Utils import Pruner

    fg.build_default_connection(
        seq_len,
        token_wise=args.build_connection_token_wise,
        inter=args.build_connection_inter,
        intra=args.build_connection_intra,
    )
    return Pruner(fg, metrics, device, verbose=False)


def run_node_pruning(pruner, clean_token: Tensor, corrupt_token: Tensor, args: argparse.Namespace):
    """The node-pruning pass, identical across the three circuit scripts."""
    return pruner(  # type: ignore
        clean_token,
        corrupt_token,
        args.node_threshold,
        prune_type=args.prune_type,
        cut_mode="node",
        scoring_mode=args.scoring_mode,
        threshold_type=args.node_threshold_type,
        modify_inplace=True,  # modify inplace
        return_type="retained",
        reverse_pruning=args.reverse_pruning,
        verbose=False,
        pass_through_grad=args.pass_through_grad,
        gradient_mode=args.gradient_mode,
        transfer_grad=args.transfer_grad,
        steps=args.ig_steps,
        save_node_grads=args.save_node_grads,
    )


def run_edge_pruning(pruner, clean_token: Tensor, corrupt_token: Tensor, args: argparse.Namespace):
    """The edge-pruning pass used by circuit_compress_eval.py."""
    return pruner(  # type: ignore
        clean_token,
        corrupt_token,
        args.threshold,
        prune_type=args.prune_type,
        cut_mode=args.cut_mode,
        scoring_mode=args.scoring_mode,
        threshold_type=args.threshold_type,
        modify_inplace=True,  # modify inplace
        return_type="retained",
        reverse_pruning=args.reverse_pruning,
        verbose=False,
        pass_through_grad=args.pass_through_grad,
        gradient_mode=args.gradient_mode,
        edge_gradient_mode=args.edge_gradient_mode,
        direct_effect=args.direct_effect,
    )


def make_corrupt_and_index(model, clean_token: Tensor, device: t.device) -> Tuple[Tensor, Tensor]:
    """`(corrupt_token, important_token_idx)`: an all-EOS baseline and the last-token position."""
    pad_token = model.tokenizer.eos_token_id  # type: ignore
    corrupt_token = t.tensor([[pad_token for _ in range(clean_token.shape[1])]], device=device)
    important_token_idx = t.tensor([clean_token.shape[1] - 1], device=device)
    return corrupt_token, important_token_idx


# ---------------------------------------------------------------------------------------------
# Clustering method table (circuit_compress_eval.py)
# ---------------------------------------------------------------------------------------------
# Maps a `cluster_type` string to the SAEGraphVisualizer method and the keyword arguments it was
# called with. `n_clusters` is injected at call time. The OT variants are the alpha sweep.

CLUSTER_TYPE_SPECS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "naive": ("run_clustering", {}),
    "ot": ("run_clustering_ot_fast", dict(alpha=0.8, lambda_pen=1.0)),
    "ot_05": ("run_clustering_ot_fast", dict(alpha=0.5, lambda_pen=1.0)),
    "ot_01": ("run_clustering_ot_fast", dict(alpha=0.1, lambda_pen=1.0)),
    "ot_10": ("run_clustering_ot_fast", dict(alpha=1, lambda_pen=1.0)),
    "ot_0": ("run_clustering_ot_fast", dict(alpha=0, lambda_pen=1.0)),
    "ot_global": ("run_clustering_ot_global_concat_fast", dict(alpha=0.8, lambda_pen=1.0)),
    "ot_global_05": ("run_clustering_ot_global_concat_fast", dict(alpha=0.5, lambda_pen=1.0)),
    "ot_global_01": ("run_clustering_ot_global_concat_fast", dict(alpha=0.1, lambda_pen=1.0)),
    "ot_global_10": ("run_clustering_ot_global_concat_fast", dict(alpha=1, lambda_pen=1.0)),
    "ot_global_0": ("run_clustering_ot_global_concat_fast", dict(alpha=0, lambda_pen=1.0)),
    "feat_flow_llama": ("run_clustering_decoder", dict(method="feature_flow")),
    "mech_permute_llama": ("run_clustering_decoder", dict(method="mech_permute")),
    "llm": ("run_clustering_llm", dict(n_intervals=3, k_per_interval=3)),
    "causal": ("run_clustering_causal", dict(lambda_pen=0)),
}


def run_clustering_by_type(graph_vis, cluster_type: str, n_clusters: int) -> None:
    """Dispatch to the SAEGraphVisualizer clustering method for `cluster_type`.

    Reproduces the original if/elif chain, including the positional-vs-keyword `n_clusters`
    convention each method was called with.
    """
    if cluster_type not in CLUSTER_TYPE_SPECS:
        raise ValueError(f"Unknown cluster type: {cluster_type}")

    method_name, kwargs = CLUSTER_TYPE_SPECS[cluster_type]
    method = getattr(graph_vis, method_name)

    # run_clustering_decoder / _llm take n_clusters as a keyword;
    # the rest take it positionally. Both reach the same parameter, kept as-is.
    if method_name in ("run_clustering_decoder", "run_clustering_llm"):
        method(n_clusters=n_clusters, **kwargs)
    else:
        method(n_clusters, **kwargs)


# ---------------------------------------------------------------------------------------------
# Layer-pair presets (feature_matching_test.py)
# ---------------------------------------------------------------------------------------------
# The `file_names` dicts that used to sit commented out in feature_matching_test.py, keyed by
# (sae_a_id, sae_b_id). A `None` value means "compute this method now" rather than "load it from
# disk" -- only `causal` is ever computed on the fly.

LAYER_PAIR_PRESETS: Dict[Tuple[int, int], Dict[str, str | None]] = {
    (1, 0): {
        "ot": "ot_1_0_25_25_euclidean_500_0.1.pt",
        "causal": None,
    },
    (11, 0): {
        "ot": "ot_11_0_25_25_euclidean_500_0.1.pt",
        "feat_flow": "feat_flow_11_0_25_0.1.pt",
        "mech_permute": "mech_permute_11_0_25_0.1.pt",
        "naive": "naive_11_0_25_euclidean_0.1.pt",
        "feat_flow_ml": "feat_flow_ml_11_0_25_0.1.pt",
        "mech_permute_ml": "mech_permute_ml_11_0_25_0.1.pt",
        "causal": None,
    },
    (11, 5): {
        "ot": "ot_11_5_25_25_euclidean_500_0.1.pt",
        "feat_flow": "feat_flow_11_5_25_0.1.pt",
        "mech_permute": "mech_permute_11_5_25_0.1.pt",
        "naive": "naive_11_5_25_euclidean_0.1.pt",
        "feat_flow_ml": "feat_flow_ml_11_5_25_0.1.pt",
        "mech_permute_ml": "mech_permute_ml_11_5_25_0.1.pt",
        "causal": None,
    },
    (6, 5): {
        "ot": "ot_6_5_25_25_euclidean_500_0.1.pt",
        "feat_flow": "feat_flow_6_5_25_0.1.pt",
        "mech_permute": "mech_permute_6_5_25_0.1.pt",
        "naive": "naive_6_5_25_euclidean_0.1.pt",
        "causal": None,
    },
    (25, 0): {
        "ot": "ot_25_0_25_25_euclidean_500_0.1.pt",
        "feat_flow": "feat_flow_25_0_25_0.1.pt",
        "mech_permute": "mech_permute_25_0_25_0.1.pt",
        "naive": "naive_25_0_25_euclidean_0.1.pt",
        "feat_flow_ml": "feat_flow_ml_25_0_25_0.1.pt",
        "mech_permute_ml": "mech_permute_ml_25_0_25_0.1.pt",
        "causal": None,
    },
    (25, 12): {
        "ot": "ot_25_12_25_25_euclidean_500_0.1.pt",
        "feat_flow": "feat_flow_25_12_25_0.1.pt",
        "mech_permute": "mech_permute_25_12_25_0.1.pt",
        "naive": "naive_25_12_25_euclidean_0.1.pt",
        "feat_flow_ml": "feat_flow_ml_25_12_25_0.1.pt",
        "mech_permute_ml": "mech_permute_ml_25_12_25_0.1.pt",
        "causal": None,
    },
    (13, 12): {
        "ot": "ot_13_12_25_25_euclidean_500_0.1.pt",
        "feat_flow": "feat_flow_13_12_25_0.1.pt",
        "mech_permute": "mech_permute_13_12_25_0.1.pt",
        "naive": "naive_13_12_25_euclidean_0.1.pt",
        "causal": None,
    },
}


def resolve_file_names(sae_a_id: int, sae_b_id: int, methods: Sequence[str]) -> Dict[str, str | None]:
    """Select the `{method: file_name}` subset for a layer pair, in the requested order."""
    key = (sae_a_id, sae_b_id)
    if key not in LAYER_PAIR_PRESETS:
        raise ValueError(
            f"No layer-pair preset for {key}. Known pairs: {sorted(LAYER_PAIR_PRESETS.keys())}. "
            "Add one to LAYER_PAIR_PRESETS in exp_config.py."
        )
    preset = LAYER_PAIR_PRESETS[key]
    missing = [m for m in methods if m not in preset]
    if missing:
        raise ValueError(f"Methods {missing} are not defined for layer pair {key}: available {sorted(preset)}")
    return {m: preset[m] for m in methods}


# ---------------------------------------------------------------------------------------------
# Misc shared helpers
# ---------------------------------------------------------------------------------------------
def show_cluster_strings(model, cluster_id, strings_list, labels, max_to_show=10):
    """
    Prints the strings that belong to a specific cluster.

    Args:
        cluster_id (int): The ID of the cluster to inspect.
        strings_list (list): The original list of strings (your `all_strings`).
        labels (np.ndarray): The array of cluster labels from KMeans.
        max_to_show (int): Maximum number of examples to print.
    """
    import numpy as np

    # Find all indices where the label matches our target cluster
    indices = np.where(labels == cluster_id)[0]

    print(f"--- Cluster {cluster_id} ---")
    print(f"Total items in cluster: {len(indices)}")
    print(f"Showing up to {max_to_show} examples:\n")

    # Iterate through the indices and print the corresponding string
    for idx in indices[:max_to_show]:
        # If your 'sequence' is a list of token IDs, you might need to decode it here
        # e.g., model.to_string(strings_list[idx])
        print(
            f"[{idx}]: {strings_list[idx]}".replace("\n", "↵"),
            "\nTarget tokens:",
            model.to_str_tokens(strings_list[idx])[-1].replace("\n", "↵"),  # type: ignore
        )

    print("-" * 25)


def load_prompt_sequences(prompt_file: str) -> List[Dict[str, Any]]:
    """Read a `loss_below_<L>_<model>_sequences.json` prompt set."""
    import json

    with open(prompt_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Number of sequences: {len(data)}")
    print("Data example 1:", data[0])
    return data


def default_prompt_file(cfg: ModelConfig, loss_threshold: str = "0.1") -> str:
    return f"loss_below_{loss_threshold}_{cfg.model_name_flat}_sequences.json"
