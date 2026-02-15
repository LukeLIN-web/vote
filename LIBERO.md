# Libero Evaluation


## Relevant Files

Evaluation
* `experiments/robot/libero/`: LIBERO eval files
    * `run_libero_eval.py`: LIBERO eval script
    * `libero_utils.py`: LIBERO eval utils
    * `batch_eval.py`: Multiple-GPU parallel evaluation script
    * `batch_plot.ipynb`: Plotting script for batch evaluation results
* `experiments/robot/`: General eval utils files
    * `openvla_utils.py`: OpenVLA-specific eval utils
    * `robot_utils.py`: Other eval utils

Training
* `vla-scripts/train.py`: VLA train script


## Environment

Requires 1 GPU with ~16 GB VRAM.

Install LIBERO package (editable mode), then install LIBERO runtime deps needed for evaluation:
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e . --config-settings editable_mode=compat

# Runtime deps used by LIBERO eval
pip install robosuite==1.4.0 robomimic==0.2.0 bddl==1.0.1 gym==0.25.2
```

Initialize LIBERO config once to avoid interactive prompts during import:
```bash
LIBERO_ROOT=$(pwd)
mkdir -p ~/.libero "${LIBERO_ROOT}/libero/datasets"
cat > ~/.libero/config.yaml <<EOF
benchmark_root: ${LIBERO_ROOT}/libero/libero
bddl_files: ${LIBERO_ROOT}/libero/libero/bddl_files
init_states: ${LIBERO_ROOT}/libero/libero/init_files
datasets: ${LIBERO_ROOT}/libero/datasets
assets: ${LIBERO_ROOT}/libero/libero/assets
EOF
```

Notes:
- Keep evaluation in the same Python environment as `effvla` (`conda activate vote`).
- `batch_eval.py` launches `run_libero_eval.py` with the current Python interpreter.
- Avoid `pip install -r LIBERO/requirements.txt` directly in the `vote` env, since it can downgrade core `effvla` dependencies (e.g., `numpy`, `transformers`, `tokenizers`).


## Evaluation

```bash
conda activate vote

# Evaluate all checkpoint subdirectories in a parent ckpt folder.
# Use task_suite names: libero_spatial / libero_object / libero_goal / libero_10 / libero_90
python experiments/robot/libero/batch_eval.py \
  --dir /shared/user71/workspace/juyi/ckpts \
  --task_suite libero_goal \
  --devices 0 1 2 3 4 5 6 7 \
  --log_dir /shared/user71/workspace/juyi/vote/eval_logs_goal_8gpu
```

Notes:
- `--dir` must point to a parent directory whose direct children are checkpoint directories.
- The task string for "goal" is `libero_goal` (not `goal`).
- `batch_eval.py --hf_ckpts` only evaluates the hard-coded HF list in the script.
- If your checkpoints are under `/shared/user71/workspace/juyi/ckpts`, use that path (not `/home/user1/workspace/juyi/ckpts`).
When you have multiple checkpoints, the results could be plotted with `batch_plot.ipynb`.

## Training

Dataset

```
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
```


We fine-tune on OpenVLA model using AdamW with a learning rate of 1e-4. Fine-tuning employs LoRA with rank r = 32 and α = 16. By default, the model is finetuned to output one token $\texttt{<ACT>}$ with a chunk size of 8.
