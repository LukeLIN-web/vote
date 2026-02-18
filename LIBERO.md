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

### Shell Launcher (Recommended)

The `run_libero_goal_eval.sh` script handles multi-GPU scheduling, logging, and error tracking:

```bash
conda activate vote

# Defaults: 8 GPUs, libero_goal task suite, fel action head
bash run_libero_goal_eval.sh

# Override via environment variables
CKPT_DIR=/path/to/ckpts \
TASK_SUITE=libero_spatial \
DEVICES="0 1 2 3" \
NUM_BLOCKS=4 \
HIDDEN_DIM=2048 \
ACTION_HEAD_NAME=fel \
MODE=mul \
LOG_DIR=./eval_logs \
bash run_libero_goal_eval.sh
```

Shell launcher environment variables:

| Variable | Default | Description |
|---|---|---|
| `CKPT_DIR` | `/shared/user71/workspace/juyi/ckpts` | Parent dir of checkpoint subdirs |
| `TASK_SUITE` | `libero_goal` | Task suite name |
| `DEVICES` | `0 1 2 3 4 5 6 7` | Space-separated GPU IDs |
| `NUM_BLOCKS` | `4` | MLPResNet depth |
| `HIDDEN_DIM` | `2048` | Hidden dimension |
| `NUM_ACTIONS_CHUNK` | `8` | Action chunk size |
| `NUM_ACTIONS_PER_TOKEN` | `8` | Actions per token |
| `ACTION_HEAD_NAME` | `fel` | Action head (`mlp`, `fel`) |
| `MODE` | `mul` | Prediction mode |
| `LOG_DIR` | `.tmp/session/eval_manual` | Log output dir |

### Python Script (Alternative)

```bash
conda activate vote

# Evaluate all checkpoint subdirectories in a parent ckpt folder.
# Task suite names: libero_spatial / libero_object / libero_goal / libero_10 / libero_90
python experiments/robot/libero/batch_eval.py \
  --dir /path/to/ckpts \
  --task_suite libero_goal \
  --devices 0 1 2 3 4 5 6 7 \
  --log_dir eval_logs
```

### Evaluating HuggingFace Checkpoints

```bash
python experiments/robot/libero/batch_eval.py --hf_ckpts --task_suite libero_spatial
```

This evaluates the predefined HF checkpoints listed in `batch_eval.py` (e.g., `juyil/llama3.2-1B-spatial`).

### Single Checkpoint Evaluation

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --base_vla_path openvla/openvla-7b \
  --pretrained_checkpoint /path/to/checkpoint \
  --task_suite_name libero_goal \
  --center_crop True \
  --use_l1_regression True \
  --num_actions_chunk 8 \
  --num_actions_per_token 8 \
  --num_blocks 4 \
  --hidden_dim 2048 \
  --mode mul \
  --action_head_name fel
```

### Task Suite Step Limits

| Task Suite | Max Steps |
|---|---|
| `libero_spatial` | 220 |
| `libero_object` | 280 |
| `libero_goal` | 300 |
| `libero_10` | 520 |
| `libero_90` | 400 |

### Notes

- `--dir` must point to a parent directory whose direct children are checkpoint directories.
- The task string for "goal" is `libero_goal` (not `goal`).
- Ensure `model_type` matches the backbone family in `experiments/robot/openvla_utils.py` (`GenerateConfig`):
  - `model_type="llama2"` for LLaMA2-based checkpoints (`base_vla_path="openvla/openvla-7b"`)
  - `model_type="llama3.2"` for LLaMA3.2-based checkpoints (`base_vla_path="juyil/llama3.2-1B-VLM"`)
- When you have multiple checkpoints, results can be plotted with `batch_plot.ipynb`.

## Training

### Dataset

```bash
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
```

### Running LIBERO Training

We fine-tune OpenVLA using AdamW with a learning rate of 1e-4. Fine-tuning employs LoRA with rank r = 32 and alpha = 16. By default, the model is finetuned to output one `<ACT>` token with a chunk size of 8.

```bash
bash trainlibero.sh
```

Default configuration (2 GPUs):

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/train.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/modified_libero_rlds/ \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/ckpts \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --use_proprio False \
  --num_images_in_input 1 \
  --batch_size 20 \
  --learning_rate 1e-4 \
  --shuffle_buffer_size 256_000 \
  --max_steps 100005 \
  --save_freq 5000 \
  --image_aug True \
  --lora_rank 32 \
  --num_actions_chunk 8 \
  --num_actions_per_token 8 \
  --num_blocks 4 \
  --mode "mul" \
  --action_head_name "funnel"
```

Available LIBERO dataset names: `libero_spatial_no_noops`, `libero_object_no_noops`, `libero_goal_no_noops`, `libero_10_no_noops`, `libero_90_no_noops`.
