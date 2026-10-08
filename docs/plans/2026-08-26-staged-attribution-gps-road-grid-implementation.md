# Staged Attribution GPS/Road/Grid Implementation Plan

> Design authority: `docs/plans/2026-08-26-staged-attribution-gps-road-grid-design.md`.

## Delivery invariants

- Work only in `/workspace/guozuyu/HTP-staged-attribution` on `exp/staged-attribution-gps-road-grid`.
- Preserve the four-code HTP token protocol and route-conditioned decoder.
- No POI input, target, stratum, or reported claim.
- Every scientific cell reaches Stage 2 training and Stage 3/4 evaluation; a missing downstream stage is `incomplete`, never a proxy result.
- Every formal run starts from a clean commit and stores revision, command, resolved config, data hashes, seed, logs, artifacts and result JSON.
- Write a failing test before each implementation unit, observe the expected failure, implement minimally, then run the focused and full test suites.

## Task 1: Reproducible environment and frozen experiment registry

Files:

- modify `pyproject.toml`, `uv.lock`, `.gitignore`;
- create `stage_1/configs/staged_attribution/experiment.json`;
- create `stage_1/src/staged_attribution/{__init__,config}.py`;
- create `stage_1/tests/staged_attribution/test_config.py`.

Steps:

1. Add the exact dependency versions recorded by `/workspace/guozuyu/HTP-main/reproduction/code/requirements.txt`; omit only `xfinance-python-sdk`, which is unused by HTP. Add `pyarrow` explicitly because parquet is a runtime interface. Resolve with `uv lock`, then `uv sync --frozen`.
2. Test that the registry expands the ten unique core cells: seven `E*/Q0` plus `E6/Q1..Q3`, with `E6/Q0` deduplicated.
3. Test rejection of POI, unknown views, changed RQ vocabulary, empty seeds, duplicate cell IDs and a downstream budget that omits Stage 2 or Stage 3.
4. Record the two cities, three seeds, smoke/controlled/confirmation budgets, model root, data roots and artifact root in config. Environment-specific absolute paths may be overridden only by explicit CLI flags and must appear in resolved config.
5. Commit: `build(stage1): lock staged attribution environment`.

Verification:

```sh
uv sync --frozen
uv run pytest stage_1/tests/staged_attribution/test_config.py -v
uv run pytest -q
```

## Task 2: Point-aligned GPS/Road/Grid dataset

Files:

- create `stage_1/src/staged_attribution/data.py`;
- create `stage_1/src/staged_attribution/prepare_data.py`;
- create `stage_1/tests/staged_attribution/test_data.py`;
- create `stage_1/configs/staged_attribution/data_schema.json`.

Steps:

1. Test a tiny fixture where GPS/Grid/pointwise Road, time, dxy and percent have identical lengths but route path has a different length.
2. Test failures for mismatched `traj_id`, offsets, point counts, split overlap and out-of-range Grid/Road IDs.
3. Implement a memory-mapped dataset joining aligned arrays to HTP processed records by `traj_id`, never by unchecked row position.
4. Implement deterministic 100 m Grid construction and pointwise Road extraction for Chengdu with source/split SHA256 manifests.
5. Implement collate output containing original views, route decoder inputs, reconstruction targets, masks and physical units.
6. Compare every Porto trajectory ID/length and a deterministic 1,000-trajectory value sample against the existing aligned artifact; save audit JSON.
7. Commit: `feat(stage1): add aligned trajectory dataset`.

Verification:

```sh
uv run pytest stage_1/tests/staged_attribution/test_data.py -v
uv run python -m staged_attribution.prepare_data --config stage_1/configs/staged_attribution/experiment.json --city porto --audit-only
uv run pytest -q
```

## Task 3: E-stage compatible model

Files:

- create `stage_1/src/staged_attribution/model.py`;
- create `stage_1/tests/staged_attribution/test_model.py`.

Steps:

1. Test all seven view masks on the same fixture.
2. Test identical output code shape, four vocabulary bounds, `ceil(T/8)` lengths and identical total/trainable parameter counts across masks.
3. Test that perturbing a disabled view leaves codes/latents unchanged in eval mode, while perturbing an enabled view changes its fused input.
4. Implement three always-instantiated adapters, learned null vectors, normalized masked fusion, the existing HTP encoder/decoder and compatible `forward/encode/decode` methods.
5. Load the original GPS baseline weights with a documented compatibility mapper; report missing/unexpected keys rather than silently dropping them.
6. Commit: `feat(stage1): add gps road grid view ablations`.

## Task 4: Q-stage losses and sampling views

Files:

- create `stage_1/src/staged_attribution/{quantizer,losses,views}.py`;
- create `stage_1/tests/staged_attribution/{test_quantizer,test_losses,test_views}.py`.

Steps:

1. Test four residual codebooks, prefix conditioning, soft assignment entropy and decode round trip.
2. Test sampling perturbation determinism by seed, preserved trajectory/event alignment, GPS interpolation and nearest-neighbor Road/Grid restoration.
3. Test invariance/variance/covariance losses on known tensors, including a collapsed tensor with nonzero variance penalty.
4. Test auxiliary labels for speed, turn, Road transition and Grid transition; thresholds must come only from train metadata.
5. Implement Q0–Q3 as resolved loss-weight profiles with the exact frozen weights; return every component even when its weight is zero.
6. Commit: `feat(stage1): add quantization attribution losses`.

## Task 5: Stage 1 trainer, evaluator and run record

Files:

- create `stage_1/src/staged_attribution/{recording,train_stage1,evaluate_stage1}.py`;
- create `stage_1/tests/staged_attribution/{test_recording,test_train_stage1,test_evaluate_stage1}.py`.

Steps:

1. Test the run state machine: `planned→running→completed/failed`; reject completion without required artifacts.
2. Test resume, atomic JSON writes, command capture, dirty-tree rejection for formal budgets and failure preservation.
3. Implement training with deterministic seeds, BF16, gradient clipping and checkpoint/resolved-config/history outputs.
4. Implement codebook, collision, probe, reconstruction and sampling-robustness JSON metrics plus validity gate.
5. Run CPU fixture and single-GPU smoke once resources permit.
6. Commit: `feat(stage1): add attribution trainer and metrics`.

## Task 6: Stable Semantic ID export and Stage 2 data

Files:

- create `stage_1/src/staged_attribution/{token_protocol,export_tokens,prepare_llm_data}.py`;
- create `stage_1/tests/staged_attribution/{test_token_protocol,test_prepare_llm_data}.py`.

Steps:

1. Golden-test exact `<t_n>` and `<a/b/c/d_n>` serialization against the HTP baseline.
2. Reject code-range, length, duplicate/missing trajectory and population errors.
3. Export oracle train/test SID records with hashes and a one-to-one population audit.
4. Reuse the original prompt content and oracle route; only the answer SID changes between cells.
5. Create deterministic smoke/controlled/full parquet subsets with explicit trajectory manifests.
6. Commit: `feat(stage2): export attribution semantic ids`.

## Task 7: Stage 2 training and token-level evaluation

Files:

- create `stage_1/src/staged_attribution/{train_stage2,evaluate_stage2}.py`;
- create `stage_1/tests/staged_attribution/{test_stage2_collator,test_stage2_metrics}.py`.

Steps:

1. Test assistant-only label masking, special-token ranges and equivalent token budget across cells.
2. Implement lazy imports for Transformers/PEFT and a dry-run fake model path for tests.
3. Record teacher-forced NLL, per-level accuracy, sampling accuracy and first-error propagation.
4. Use the local Qwen3-1.7B path and the same LoRA target modules/config across cells; save adapter, tokenizer and trainer state.
5. Commit: `feat(stage2): add controlled llm training and evaluation`.

## Task 8: Robust generation and G-stage counterfactuals

Files:

- create `stage_1/src/staged_attribution/{generate_stage3,g_attribution,decode_stage3}.py`;
- create `stage_1/tests/staged_attribution/{test_generation_parser,test_g_attribution,test_decode_stage3}.py`.

Steps:

1. Test valid, malformed, truncated, duplicate and out-of-range generations; never silently drop failures.
2. Build G00/G10/G01/G11 for the exact same IDs; road must equal the oracle/prompt route in all four.
3. Test decode compatibility and failure records without requiring a GPU.
4. Implement batched generation, deterministic seeds, merge audit and four paired decode artifacts.
5. Commit: `feat(stage3): add paired generation attribution`.

## Task 9: Stage 4 JSON metrics, bootstrap and Shapley

Files:

- create `stage_1/src/staged_attribution/{stage4_metrics,statistics,analyze}.py`;
- create `stage_1/tests/staged_attribution/{test_stage4_metrics,test_statistics,test_analyze}.py`.

Steps:

1. Golden-test existing HTP aggregate metrics on fixed toy trajectories and add per-trajectory sufficient statistics where mathematically valid.
2. Test population-aware invalid penalties and successful-only sensitivity analysis.
3. Test paired trajectory bootstrap, SID/sampling Shapley identities and interaction identity.
4. Implement E contrasts, Q incremental contrasts, adaptive interaction selection, Pareto gates, city heterogeneity and multi-seed aggregation.
5. Commit: `feat(stage4): add attribution statistics and reports`.

## Task 10: End-to-end runner and queue

Files:

- create `stage_1/src/staged_attribution/{run_cell,run_experiment}.py`;
- create `stage_1/tests/staged_attribution/{test_run_cell,test_run_experiment}.py`;
- create `stage_1/run_staged_attribution.sh`.

Steps:

1. Test dry-run DAG, resume semantics, artifact prerequisites and no-proxy completion rules.
2. Run all ten core smoke cells end-to-end; fix and commit failures separately.
3. Freeze smoke audit and controlled preregistration on a clean commit.
4. Queue all core cells for two cities × three seeds, then adaptive interaction cells. Wait for safe GPU availability; do not preempt other users.
5. Run confirmations for baseline and selected candidates.
6. Commit runner, preregistration, each formal result set and final report separately.

## Task 11: Final evidence and experiment records

Files:

- create `stage_1/reports/staged_attribution/preregistration.md`;
- create `stage_1/reports/staged_attribution/controlled_results.md`;
- create `stage_1/reports/staged_attribution/confirmation_results.md`;
- create `stage_1/reports/staged_attribution/final_conclusion.md`.

Steps:

1. Audit every required cell/stage/city/seed against manifests and actual artifacts.
2. Separate observations, controlled contrasts, G Shapley attribution, interpretation and limitations.
3. State a GPS/Road/Grid conclusion only at the strength supported by confirmation; preserve negative and heterogeneous findings.
4. Mirror experiment IDs, commits, commands, metrics, failures and next actions to Notion.
5. Run `uv sync --frozen`, full pytest, smoke DAG verifier, report completeness verifier and `git diff --check` from a clean commit before completion.
