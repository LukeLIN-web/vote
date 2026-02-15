# Repository Guidelines

## Project Structure & Module Organization
- `prismatic/`: core library code (model backbones, VLA heads, training strategies, dataset loaders, and configs in `prismatic/conf/`).
- `vla-scripts/train.py`: main distributed training entrypoint.
- `experiments/speed/`: lightweight inference/speed scripts (for example `python experiments/speed/effvla.py`).
- `experiments/robot/libero/`: LIBERO evaluation tools (`run_libero_eval.py`, `batch_eval.py`).
- `LIBERO/` and `src/libero/`: vendored LIBERO package/assets used by robot benchmarks.
- `train.sh` and `trainlibero.sh`: reproducible launch examples for Fractal and LIBERO-style runs.

## Build, Test, and Development Commands
- `pip install -e .`: install the repo in editable mode.
- `pip install -e .[dev]`: install contributor tools (`black`, `ruff`, `pre-commit`, etc.).
- `bash train.sh`: start single-node training with the default Fractal config.
- `bash trainlibero.sh`: start LIBERO-oriented training.
- `python experiments/speed/effvla.py`: quick local inference/speed smoke check.
- `python experiments/robot/libero/batch_eval.py --dir <ckpt_parent> --task_suite libero_goal --devices 0`: run LIBERO evaluation on checkpoints.

## Coding Style & Naming Conventions
- Python style is enforced by `black` and `ruff` from `pyproject.toml`.
- Use 4-space indentation and keep line length `<= 121`.
- Run formatting/linting before opening a PR:
  - `black .`
  - `ruff check .`
- Naming patterns in this repo: `snake_case` for functions/files, `PascalCase` for classes, descriptive config names under `prismatic/conf/`.

## Testing Guidelines
- There is no dedicated unit-test suite yet; rely on targeted smoke checks.
- For model changes, run at least one speed script and one LIBERO eval command.
- Keep evaluation artifacts out of commits (`eval_logs_*`, large checkpoints, and generated logs).

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
