# ATML PA2 — LLM Post-Training: DPO, PPO, GRPO, Safety, RLVR vs. RLAIF

## 1. Setup

```bash
git clone https://github.com/Abdullah-61/ATML_PA2.git
cd ATML_PA2
python -m pip install -r requirements.txt
python -m scripts.download_assets        # fixed course data, caches and checkpoints (pinned HF revision)
python -m scripts.prepare_transfer_eval  # Task 5 SVAMP-100 file (no-op if it already exists)
python -m scripts.validate_assets
python -m pytest -q tests                # unit tests for the three objective fixes + Task 4/5 helpers (20 tests)
```


## 2. Repository layout

```
task1_dpo_colab.ipynb ... task5_feedback_colab.ipynb   the Colab notebooks that were executed (one per task)
common/            shared data / model / generation / metric helpers (release code, unchanged)
configs/           release YAMLs (+ appended evaluation-only settings)
data/              word_limit_prompts.jsonl (tracked course file; everything else is downloaded)
scripts/           asset download/validation, scripts/make_figures.py (report figures)
task1_dpo/         DPO objective (fixed), training loop, evaluation, beta ablation, length study
task2_ppo/         PPO objective (fixed), 20-update continuation, evaluation, clipping study, KL ablation
task3_grpo/        GRPO objective (fixed), continuation, evaluation, group-size study, normalization forks
task4_safety/      XSTest generation, fixed AI judge orchestration, audit sheet, calibration analysis
task5_feedback/    GSM8K/SVAMP generation, verifier + pairwise judge evaluation, controlled diagnostics
taskX/report_tables.py   regenerates every table / plot / qualitative example of that task from results/
tests/             unit tests
results/           ALL logs and machine-readable results used in the report (JSON / JSONL / CSV / PNG)
report/            LaTeX source (main.tex, figs/), and the final PDF
```

The experiment pipeline is the Python modules above. Each `taskN_*_colab.ipynb` notebook is the exact
Colab run for that task: it calls these modules (plus Drive mounting / resuming after disconnects), and its
saved outputs show the logs of the runs reported in the paper.

## 3. Objective fixes (one deliberate defect per Task 1–3)

| Task | File | Starter defect | Fix | Test |
|---|---|---|---|---|
| DPO | `task1_dpo/dpo.py` | logits `β·(policy_margin + ref_margin)` | `β·(policy_margin − ref_margin)` | `tests/test_dpo_loss.py` (closed form; loss = log 2 at init) |
| PPO | `task2_ppo/ppo.py` | `torch.maximum` in the clipped surrogate | `torch.minimum` (pessimistic bound) | `tests/test_ppo.py` |
| GRPO | `task3_grpo/grpo.py` | advantages standardised over the whole batch, `group_ids` ignored | `(r − μ_g)/(σ_g + ε)` within each prompt group | `tests/test_grpo.py` |

Other implementation decisions that affect results (identical for every compared condition):

* Pairs/prompts whose **prompt alone exceeds the sequence limit** are skipped deterministically (the starter
  encoder raises otherwise); skipped indices are logged (`results/task1_dpo/data_filter_*.json`,
  `n_prompts_skipped_too_long` and `source_indices` in the PPO/GRPO eval JSONs).
* Trainable LoRA weights in **fp32 + fp16 autocast + GradScaler** (fp16 AdamW states underflowed to NaN);
  non-finite steps are skipped and counted (`skipped_optimizer_steps` in the PPO logs).
* **Dropout disabled** during RL updates so π_θ = π_old at the first epoch.
* KL = released sampled-response estimator (masked token mean); the per-sequence sum is stored as well.

## 4. Reproducing every task

Run all commands from the repository root. Long runs are resumable: finished adapters/evaluations are
detected and skipped.

### Task 1 — DPO
```bash
python -m task1_dpo.train --run-name standard                                   # 1 epoch, beta=0.10
python -m task1_dpo.evaluate --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta                                                 # beta in {0.03,0.10,0.30}, 600 pairs each
python -m task1_dpo.analyze_length --standard-adapter outputs/task1_dpo/standard  # length-balanced run + stratified eval
python -m task1_dpo.report_tables
```
Results: `results/task1_dpo/` — `eval_*.json`, `stratified_*.json`, `beta_ablation_summary.json`,
`length_study_summary.json`, `train_log_*.jsonl`, `generations_*.jsonl`, `wordlimit_*.jsonl`,
`data_filter_*.json` (kept/skipped source indices), `run_meta_*.json`.

### Task 2 — PPO
```bash
python -m task2_ppo.continue_train --run-name smoke --updates 1      # optional smoke test
python -m task2_ppo.continue_train --run-name standard               # 20 updates from the supplied midpoint
python -m task2_ppo.evaluate --name reference                        # frozen reference
python -m task2_ppo.evaluate --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --skip-forks                    # cached-rollout study, eps in {0.05,0.20,0.50}
python -m task2_ppo.analyze_clipping --skip-cached                   # matched 8-update eps forks
python -m task2_ppo.ablate_kl                                        # beta_KL in {0,0.10,0.20} forks
python -m task2_ppo.report_tables
```
Results: `results/task2_ppo/` — `train_log_*.jsonl`, `run_meta_*.json` (wall-clock, peak VRAM),
`eval_*.json`, `generations_*.jsonl`, `cached_clipping_study.json`, `kl_ablation_summary.json`, `figures/`.

### Task 3 — GRPO
```bash
python -m task3_grpo.continue_train --run-name smoke --updates 1 --output outputs/task3_grpo/smoke   # optional
python -m task3_grpo.continue_train --run-name standard              # 20 updates, K=4
python -m task3_grpo.evaluate --name reference
python -m task3_grpo.evaluate --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.evaluate --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size                              # K in {2,4,8} on the cached completions (no training)
python -m task3_grpo.compare_normalization                           # canonical vs Dr.-GRPO forks + gradient statistics
python -m task3_grpo.report_tables
```
Results: `results/task3_grpo/` — `train_log_*.jsonl`, `seqstats_*.jsonl` (per-completion length, reward,
advantage, gradient norm), `eval_*.json`, `group_size_study.json` (binning rule included),
`normalization_summary.json`, `run_meta_*.json`, `figures/`.

### Task 4 — Safety calibration
Requires the three standard adapters from Tasks 1–3 at `outputs/task1_dpo/standard`,
`outputs/task2_ppo/standard`, `outputs/task3_grpo/standard` (SFT = base model, no adapter).
```bash
python -m task4_safety.generate_responses        # greedy, 256 tokens, all 450 XSTest prompts x 4 policies
python -m task4_safety.make_audit_sheet          # fixed 30 SAFE + 30 UNSAFE ids -> blind, shuffled sheet
# manual step: fill manual_label in results/task4_safety/manual_audit_sheet.csv BEFORE looking at judge labels
python -m task4_safety.judge_responses           # fixed categorical AI judge
python -m task4_safety.evaluate_safety           # calibration / category / agreement tables
python -m task4_safety.report_tables             # + plots and qualitative disagreement examples
```
Results: `results/task4_safety/` — `generated_*.jsonl`, `judged_*.jsonl`, `manual_audit_ids.csv`
(fixed audit IDs), `manual_audit_sheet.csv` (blind manual labels), `manual_audit_key.csv`,
`calibration_table.csv`, `category_label_distribution.csv`, `audit_confusion.csv`,
`audit_policy_metrics.csv`, `judge_diagnostics.csv`, `paired_tests.csv`, `safety_summary.json`, plots.

### Task 5 — RLVR vs. RLAIF
```bash
python -m task5_feedback.evaluate_math --dataset both --stage generate --batch-size 16   # SFT / RLVR / RLAIF
python -m task5_feedback.evaluate_math --dataset both --stage judge                      # fixed pairwise judge (cached)
python -m task5_feedback.score_perturbations                                             # controlled diagnostic set
python -m task5_feedback.report_tables                                                   # all tables, plots, cost
```
Results: `results/task5_feedback/` — `gen_*.jsonl`, `judge_*.jsonl`, `judge_cache.json`,
`diag_pairs.jsonl`, `diag_variants.jsonl`, `diag_pair_table.csv`, `diag_sensitivity.csv`,
`gsm_*.csv`, `transfer_*.csv`, `task5_summary.json`, `generation_timing.json`, plots.

### Report figures
```bash
python -m scripts.make_figures     # writes report/figs/online_traj.pdf and report/figs/grpo_k_norm.pdf
```

## 5. Where each report table comes from

| Report | Source file(s) |
|---|---|
| Table 1 (DPO summary) | `results/task1_dpo/eval_{standard,beta0.03,beta0.1,beta0.3,length_balanced_lenstudy}.json`, `run_meta_*.json` |
| Table 2 (length strata) | `results/task1_dpo/stratified_{standard,length_balanced}.json` |
| Fig. 1 (PPO/GRPO trajectories) | `results/task2_ppo/train_log_standard.jsonl`, `results/task3_grpo/train_log_standard.jsonl` |
| Table 3 (PPO clipping / KL) | `results/task2_ppo/cached_clipping_study.json`, `eval_*.json`, `train_log_eps*_kl*.jsonl` |
| Table 4, Fig. 2 (group size) | `results/task3_grpo/group_size_study.json` |
| Table 5, Fig. 2 (normalization) | `results/task3_grpo/eval_{canonical,dr}_grpo.json`, `normalization_summary.json`, `seqstats_*.jsonl` |
| Table 6 (safety calibration) | `results/task4_safety/calibration_table.csv`, `audit_policy_metrics.csv`, `judged_*.jsonl` |
| Table 7 (judge vs manual) | `results/task4_safety/audit_confusion.csv`, `safety_summary.json` |
| Table 8 (GSM8K / SVAMP) | `results/task5_feedback/gsm_policy_table.csv`, `transfer_policy_table.csv`, `*_paired_accuracy_tests.csv` |
| Table 9 (controlled diagnostics) | `results/task5_feedback/diag_pair_table.csv`, `diag_sensitivity.csv`, `diag_variant_table.csv` |
| Compute (VRAM, wall-clock) | `results/task2_ppo/run_meta_standard.json`, `results/task3_grpo/run_meta_standard.json` |

## 6. Key hyperparameters (from the release configs)

| | |
|---|---|
| Policy | Qwen/Qwen2.5-1.5B-Instruct, LoRA r=8, α=16, dropout 0.05 on `q_proj,v_proj`; frozen models 4-bit, fp16 |
| DPO | lr 2e-5, batch 2 × grad-accum 8, 1 epoch, β=0.10, max seq 768, generation cap 256, T=0.7, top-p 0.9 |
| PPO | policy lr 3e-6, value LoRA/head lr 1e-4/3e-4, 2 PPO epochs, ε=0.2, β_KL=0.1, γ=1, λ=0.95, 1 prompt/update, 20 updates (forks: 8) |
| GRPO | lr 5e-6, K=4, 1 epoch, ε=0.2, β=0.1, completion cap 512, truncated completions masked, 20 updates (forks: 8) |
| Safety / Feedback | greedy; 256 (safety) / 512 (math) tokens; judges at temperature 0 |

## 7. Attribution

* Starter code (`common/`, `scripts/`, configs, the Task 4 safety judge, the Task 5 verifier `rlvr.py` and
  pairwise judge `rlaif.py`, and the entry-point skeletons) is from the course repository
  <https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining>.
* Libraries: PyTorch, Hugging Face Transformers, PEFT, bitsandbytes, pandas, NumPy, Matplotlib.
* Models/data: Qwen2.5-1.5B/3B-Instruct, `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`, UltraFeedback, XSTest,
  GSM8K, SVAMP (via the course asset release).
  
