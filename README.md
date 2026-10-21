# Semantic Optimal Transport for SAE Feature Matching and Circuit Compression

Code for *Semantic Optimal Transport for Sparse Autoencoder Feature Matching and Circuit
Compression* (Tue M. Cao, Nguyen Do, My T. Thai; [arXiv:2605.28567](https://arxiv.org/abs/2605.28567)).

A sparse-autoencoder feature is represented not by its decoder vector but by the
activation-weighted distribution of the hidden states that express it. Distributions from different
layers are projected into a shared reference space and compared with the Wasserstein distance. The
same distance gives cross-layer **feature matching** (the reference space is the target layer) and
**circuit compression** (the reference space concatenates all layers, and circuit nodes are grouped
into supernodes by agglomerative clustering).

## Setup

```bash
conda env create -f environment.yml      # creates the env "semantic-ot" (python 3.11, torch 2.5.1)
conda activate semantic-ot
cp .env.example .env                     # then fill in the keys you need
```

* **Models and SAEs.** Gemma-2-2B uses the Gemma Scope JumpReLU SAEs on the MLP outputs (downloaded
  through `sae_lens`; Gemma is gated, so set `HUGGING_FACE_API_KEY`). GPT2-small uses our own TopK
  SAEs on the residual stream (`hook_resid_pre` of every layer). Train them with
  `sbatch scripts/Train_SAE.sh` (Appendix D.4), or point `GPT2_SAE_TEMPLATE` at existing checkpoints
  (`{layer}` is filled in).
* **Data.** Activations come from the Pile (`monology/pile-uncopyrighted`), streamed from
  HuggingFace and cached under `data/cache_data/` on first use.
* **LLM judges** (see `exp_config.JUDGE_PRESETS`):
  * feature matching uses **gpt-oss-120b**;
  * circuit compression uses **Llama-3.3-70B-Instruct**. `scripts/circuit_compress_eval.sh` serves
    it with vLLM on the job's own GPU (`JUDGE=llama70b-vllm`, needs `vllm` and two large GPUs);
  * the **LLM Sel** baseline calls gpt-oss-120b.

  For hosted endpoints, set `JUDGE_BASE_URL` / `JUDGE_API_KEY` (judges) and `LLMSEL_URL` /
  `LLMSEL_API_KEY` (LLM Sel) to any OpenAI-compatible server. The defaults point at the University of
  Florida gateway the paper used.

Every experiment has a SLURM launcher in `scripts/` that sources `scripts/_common.sh`. Adjust the
`#SBATCH --partition` line for your cluster, and set `CONDA_ENV`, `CUDA_MODULE` or `CACHE_ROOT` in the
environment if needed. Submit from the repo root. The launchers fan out LLM calls; the judged
experiments cost API credit unless the judge is self-hosted.

## Paper results → commands

| Paper | What | Command |
|---|---|---|
| Table 1, Table 6 | Feature matching: LLM Eval, Acc, CE, VE | `feature_matching_run.sh` → `feature_matching_test.sh` → `analysis/summarize_matching.py` |
| Table 2, Table 7 | Circuit compression, 50 and 200 nodes | `circuit_compress_eval.sh` → `analysis/summarize_compression.py` |
| Fig. 2, Fig. 11 | Two-digit addition matching (human study) | `run_addition_matching.sh` |
| Section 6.1 | Automated circuit compression case studies | `case_study_circuit.sh` |
| Section 6.2 | Modular feature groups | `circuit_behavior_clustering.sh` → `modular_feature_group.sh` → `analysis/modular_groups.py` |
| Table 4 | Support-size (K) ablation | `ot_topk_ablation.sh` |
| Appendix D.4 | GPT2 TopK SAE training | `Train_SAE.sh` |

### Feature matching (Section 5.1, Tables 1 and 6)

Layer pairs (Table 3): GPT2 6←5, 11←5, 11←0; Gemma-2-2B 13←12, 25←12, 25←0.

```bash
# 1. matchings for one pair (all methods: ot, feat_flow, mech_permute, naive, *_ml)
sbatch --export=ALL,SAE_A_ID=13,SAE_B_ID=12 scripts/feature_matching_run.sh
# 2. judge the same 100 query features 5 times: seed 1 samples them, seeds 2-5 reuse them
sbatch --export=ALL,SAE_A_ID=13,SAE_B_ID=12,SEEDS=1 scripts/feature_matching_test.sh
sbatch --export=ALL,SAE_A_ID=13,SAE_B_ID=12,SEEDS="2 3 4 5",EXTRACT_FROM_PREV=true scripts/feature_matching_test.sh
# 3. table rows (mean ± 1.645·std/√5 over the 5 runs)
python analysis/summarize_matching.py --results_dir results/google_gemma-2-2b_mlp_gpt-oss-120b --pair 13 12
```

For GPT2, add `PRESET=gpt2_ds,NUM_BATCHES=1000` (the GPT2 matchings use 1,000 Pile sequences, 128K
tokens). The two GPT2 pairs with target layer 11 use the normalized final residual stream as the
reference space: add `CACHING_HOOK=ln_final.hook_normalized` to step 1. The attribution baseline
(`causal`) is computed inside step 2.

### Circuit compression (Section 5.2, Tables 2 and 7)

```bash
# 50 nodes -> 5 supernodes, 200 prompts in 8 slices of 25
sbatch --array=0-7 --export=ALL,PRESET=gemma_mlp scripts/circuit_compress_eval.sh
# 200 nodes -> 20 supernodes, 20 slices of 10 (no LLM Sel: the circuits exceed its context window)
sbatch --array=0-19 --export=ALL,PRESET=gemma_mlp,NODE_THRESHOLD=200,N_CLUSTERS=20,SLICE=10,CLUSTER_TYPES="ot_global_10 feat_flow_llama mech_permute_llama causal naive" scripts/circuit_compress_eval.sh
python analysis/summarize_compression.py --results_dir results/google_gemma-2-2b_mlp_llama70b-vllm
python analysis/summarize_compression.py --results_dir results/google_gemma-2-2b_mlp_llama70b-vllm --node_threshold 200 --n_clusters 20
```

Use `PRESET=gpt2_ds` for GPT2. The prompts come from `loss_below_0.1_<model>_sequences.json`; build
it first with `python find_prompt_examples.py --preset gemma_mlp` (or `gpt2_ds`). The cluster
types are `ot_global_10` (ours), `feat_flow_llama`, `mech_permute_llama`, `llm` (LLM Sel), `causal`
(Attrib) and `naive`. The summary averages over gradable circuits (at least one supernode with two
or more features) and reports a 90% interval of 1.645 × the standard error, with a paired t-test
against our method.

### Other experiments

* **Two-digit addition (Section 5.3).** `run_addition_matching.sh` matches the layer-19 Gemma-2-2B
  features on the `p1+p2=` prompts with every method. `addition_data/blind_ranking_vis.py` is the
  blind-ranking notebook widget used for the volunteer study.
* **Case studies (Section 6.1).** `case_study_circuit.sh` builds the Dallas, Rhythm and Opposite
  circuits (Table 5) and writes the full and compressed dashboards to `case_studies/`.
* **Modular feature groups (Section 6.2).** `circuit_behavior_clustering.sh` clusters ~18,000 prompts
  into 500 behaviors. `modular_feature_group.sh` then compresses each prompt's 200-node circuit
  into 10 supernodes and counts how often feature pairs share a supernode.
  `python analysis/modular_groups.py <its output .pt>` turns those counts into Jaccard affinities and
  spectral-clusters them into modular groups.
* **Support-size ablation (Appendix D).** `ot_topk_ablation.sh` reruns our matcher with
  K ∈ {1, 16, 256, 500} on the Table 1 query features. It reads the step-2 output of the feature
  matching run (`QUERY_PREV_FILE`).

## Reproduction notes

* **Sinkhorn helper.** The paper's feature-matching numbers were computed with
  `feature_matching/utils.py::compute_sinkhorn_ot_gpu_old`, which divides geomloss' `0.5·W₂²` by the
  per-pair maximum point norm. The launchers default to it (`--sinkhorn_impl old`).
  `compute_sinkhorn_ot_gpu` (`--sinkhorn_impl fixed`) returns the true `W₂` under one scaling and is
  the better choice for new work. Circuit compression uses exact OT (POT `emd2`) and is unaffected.
* **Run-to-run variance.** The LLM judges are sampled, so reruns move by a few hundredths of LLM
  Eval. The matching summary prints the ± from unrounded values, so it can differ from the paper's
  Table 1 in the last digit.
* **Tests.** `python -m pytest tests -q` checks the seed aggregation and the result-file helpers
  (CPU only).

## Citation

```bibtex
@article{cao2026semantic,
  title   = {Semantic Optimal Transport for Sparse Autoencoder Feature Matching and Circuit Compression},
  author  = {Cao, Tue M. and Do, Nguyen and Thai, My T.},
  journal = {arXiv preprint arXiv:2605.28567},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
