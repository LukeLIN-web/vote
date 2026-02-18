# Repository Guidelines

## Project Structure & Module Organization
- `prismatic/`: core library code (model backbones, VLA heads, training strategies, dataset loaders, and configs in `prismatic/conf/`).
  - `prismatic/models/action_heads.py`: action head architectures (`L1RegressionActionHeadmulmlpk`, `L1RegressionActionHeadFunnel`).
  - `prismatic/extern/hf/modeling_prismatic.py`: HuggingFace model class with `mul_predict_action()` for trajectory ensemble voting.
  - `prismatic/conf/vla.py`: VLA model configs with predefined experiments.
- `vla-scripts/train.py`: main distributed training entrypoint.
- `experiments/speed/`: inference/speed benchmark scripts (`effvla.py`, `llama3-1B.py`, `openvla.py`, `cogact.py`, `pi0.py`, `spatialvla.py`).
- `experiments/robot/libero/`: LIBERO evaluation tools (`run_libero_eval.py`, `batch_eval.py`, `libero_utils.py`, `batch_plot.ipynb`).
- `experiments/robot/openvla_utils.py`: `GenerateConfig` dataclass and model loading utilities.
- `experiments/robot/robot_utils.py`: image processing, action normalization, model loading helpers.
- `LIBERO/` and `src/libero/`: vendored LIBERO package/assets used by robot benchmarks.
- `train.sh`: Fractal single-GPU training launcher.
- `trainlibero.sh`: LIBERO multi-GPU training launcher.
- `run_libero_goal_eval.sh`: multi-GPU evaluation shell launcher with env-var configuration.

## Build, Test, and Development Commands
- `pip install -e .`: install the repo in editable mode.
- `pip install -e .[dev]`: install contributor tools (`black`, `ruff`, `pre-commit`, etc.).
- `bash train.sh`: start single-GPU training with the default Fractal config.
- `bash trainlibero.sh`: start LIBERO multi-GPU training (2 GPUs by default).
- `python experiments/speed/effvla.py`: quick local inference/speed smoke check (OpenVLA-7B).
- `python experiments/speed/llama3-1B.py`: LLAMA3.2-1B-VLA inference check (~4.34 GB VRAM).
- `bash run_libero_goal_eval.sh`: run multi-GPU LIBERO evaluation (configurable via env vars, see LIBERO.md).
- `python experiments/robot/libero/batch_eval.py --dir <ckpt_parent> --task_suite libero_goal --devices 0 1 2 3`: run LIBERO eval with Python scheduler.
- `python experiments/robot/libero/batch_eval.py --hf_ckpts --task_suite libero_spatial`: evaluate predefined HF Hub checkpoints.

## LIBERO Eval Runbook
- Prefer changing runtime eval hyperparameters in shell launchers (`run_libero_goal_eval.sh`) instead of editing Python eval sources.
- Treat `experiments/robot/libero/batch_eval.py` and `experiments/robot/libero/run_libero_eval.py` as stable logic files; only change them for real bug fixes.
- For one-off comparisons (for example `num_blocks=4`), pass/override values from shell env vars or CLI args and keep Python defaults unchanged.
- Key eval parameters: `--num_blocks`, `--hidden_dim`, `--num_actions_chunk`, `--num_actions_per_token`, `--action_head_name`, `--mode`.
- The shell launcher `run_libero_goal_eval.sh` accepts all params as env vars (e.g. `NUM_BLOCKS=2 bash run_libero_goal_eval.sh`).
- `model_type` must match the backbone: `"llama2"` for OpenVLA-7B, `"llama3.2"` for LLAMA3.2-1B-VLA.

## Coding Style & Naming Conventions
- Python style is enforced by `black` and `ruff` from `pyproject.toml`.
- Use 4-space indentation and keep line length `<= 121`.
- Run formatting/linting before opening a PR:
  - `black .`
  - `ruff check .`
- Naming patterns in this repo: `snake_case` for functions/files, `PascalCase` for classes, descriptive config names under `prismatic/conf/`.

## Configuration Parameters
Training and evaluation share these key parameters (set via CLI args or env vars):

| Parameter | Description | Typical Values |
|---|---|---|
| `num_actions_chunk` | Total action sequence length | 8, 16 |
| `num_actions_per_token` | Actions decoded per `<ACT>` token | 8 |
| `num_blocks` | MLPResNet depth in action head | 2 (Fractal), 4 (LIBERO) |
| `hidden_dim` | Hidden dimension | 4096 (7B), 2048 (1B) |
| `mode` | Prediction mode | `"mul"` (multi-token ensemble) |
| `action_head_name` | Action head type | `"mlp"`, `"funnel"`, `"fel"` |
| `model_type` | Backbone family | `"llama2"`, `"llama3.2"` |

## Testing Guidelines
- There is no dedicated unit-test suite yet; rely on targeted smoke checks.
- For model changes, run at least one speed script and one LIBERO eval command.
- Keep evaluation artifacts out of commits (`eval_logs_*`, large checkpoints, generated logs, and `rollouts/`).

## Temporary Files Policy
- Put all temporary artifacts under a dedicated directory: `.tmp/session/`.
- Avoid creating temporary files in the repository root unless explicitly required.
- At the end of each conversation, delete temporary files/directories created during the session.
- Keep only user-requested persistent outputs; remove transient logs and debug artifacts.

## Commit & Pull Request Guidelines
- Recent history uses short, imperative commits (for example: `update libero`, `fix badge link formatting`). Keep subject lines concise and specific.
- Prefer one logical change per commit.
- PRs should include:
  - what changed and why,
  - exact commands used for verification,
  - linked issue/paper context when relevant,
  - result snapshots or key metrics for training/eval-impacting changes.
