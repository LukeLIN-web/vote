# VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting 🚀🤖

<div align="center">

[![Slide](https://img.shields.io/badge/Slides-07C160?style=for-the-badge&logo=slides&logoColor=white)](https://docs.google.com/presentation/d/1zId-ygV3gObqHgm4gLdM4euGUpHomxwDZjJ7UcEmzVs/edit?usp=sharing) 
[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2507.05116) 
[![Hugging Face](https://img.shields.io/badge/model-fcd022?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/juyil/vote-vision-language-action-model-686f5dac2775080477a86cdf)

</div>

## Overview

**VOTE** is a framework for building fast and accurate Vision-Language-Action (VLA) models for robotic manipulation. It introduces **Trajectory Ensemble Voting**, a method that optimizes action prediction by ensembling multiple trajectory candidates decoded from a vision-language backbone. VOTE achieves state-of-the-art performance on both simulated (LIBERO, SimplerEnv) and real-world benchmarks while being **3× faster** than prior VLA methods. Its modular design allows easy migration to any VLM backbone with just 2 lines of code—no complex action tokenizers required.

## Table of Contents

- [News](#news)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## News

- `2025/09/22`: ✨ Released **VOTE LLAMA3.2-1B-VLA** model 👉 [script](https://github.com/LukeLIN-web/vote/blob/main/experiments/speed/llama3-1B.py) — inference with only **4.34 GB** VRAM usage.
- `2025/07/10`: 🎉 Released [VOTE 1.0](https://huggingface.co/collections/juyil/vote-vision-language-action-model-686f5dac2775080477a86cdf). ➡️ No need for **complex tokenizers** — migrate to a new VLM with just **2 lines of code** ⚡️

## Installation

```bash
conda create -n vote python=3.10 -y
conda activate vote

git clone https://github.com/LukeLIN-web/vote.git
cd vote
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install flash-attn==2.6.1 --no-build-isolation
```

### Quick Start

```bash
cd experiments/speed/
python effvla.py
```

### Speed Benchmarks

We provide speed measurement scripts under `experiments/speed/`:

```bash
# OpenVLA-7B (2-token, 16-action trajectory ensemble)
python experiments/speed/effvla.py

# LLAMA3.2-1B-VLA (lightweight, ~4.34 GB VRAM)
python experiments/speed/llama3-1B.py

# Other baselines
python experiments/speed/openvla.py
python experiments/speed/cogact.py
python experiments/speed/pi0.py
python experiments/speed/spatialvla.py
```

### Troubleshooting

<details>
<summary><code>No module named prismatic</code> / <code>No module named experiments</code></summary>

This usually means the package was not installed correctly. Verify with:

```bash
pip list | grep effvla
```

If `effvla` is not listed, re-run `pip install -e .` from the repo root.
</details>

If you run into any other issues, please [open a GitHub issue](https://github.com/LukeLIN-web/vote/issues).

### Installation on Jetson AGX Orin

<details>
<summary>Click to expand</summary>

```bash
python -m venv orin
source orin/bin/activate

# Install transformers and other dependencies
pip3 install packaging ninja transformers==4.51.0 tokenizers==0.21.4 timm==0.9.10 diffusers==0.32.2

# Install TensorFlow 2.15.0
pip3 install tensorflow==2.15.0

# Install TensorFlow addons from source
git clone https://github.com/tensorflow/addons
cd addons
pip3 install -e .
cd ..

# Install VOTE
git clone https://github.com/LukeLIN-web/vote.git
cd vote
pip3 install -e .
cd ..
# Note: This step installs torch/torchvision versions incompatible with Jetson.
# We will override them with precompiled wheels below.

# Install torch & torchvision using NVIDIA's precompiled wheels for Jetson
# torch:       https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl
# torchvision: https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl
pip3 install torch*.whl torchvision*.whl
```

> **Note:** You may see a dependency conflict warning like  
> `effvla 0.0.1 requires torchvision==0.18.1, but you have torchvision 0.18.0a0+6043bc2`.  
> This is expected and can be safely ignored.

</details>

## Architecture

### Multi-Token Trajectory Ensemble

VOTE replaces standard single-token action decoding with a **multi-token trajectory ensemble** approach. The VLM backbone generates multiple `<ACT>` tokens, each decoded into an action chunk by a lightweight action head. The final trajectory is assembled by ensembling predictions across tokens.

Key parameters:

| Parameter | Description | Example |
|---|---|---|
| `num_actions_chunk` | Total actions in the output sequence | 8 or 16 |
| `num_actions_per_token` | Actions predicted per `<ACT>` token | 8 |
| `mode` | Prediction mode (`"mul"` for multi-token ensemble) | `"mul"` |

For example, `num_actions_chunk=16` with `num_actions_per_token=8` produces 2 tokens, each predicting 8 actions (16 total).

### Action Heads

Three action head architectures are available via `--action_head_name`:

| Name | Class | Description |
|---|---|---|
| `mlp` | `L1RegressionActionHeadmulmlpk` | Standard MLPResNet-based head |
| `fel` | `L1RegressionActionHeadFunnel` | Funnel architecture with progressive dimension reduction (more parameter-efficient) |

Additional parameters:
- `--num_blocks`: Number of MLPResNet blocks (typically 2 for Fractal, 4 for LIBERO)
- `--hidden_dim`: Hidden dimension size (default: 4096 for OpenVLA-7B, 2048 for LLAMA3.2-1B)

### Supported Backbones

| Backbone | Base Model Path | `model_type` | Params |
|---|---|---|---|
| OpenVLA-7B (LLaMA 2) | `openvla/openvla-7b` | `llama2` | 7B |
| LLAMA3.2-1B-VLA | `juyil/llama3.2-1B-VLM` | `llama3.2` | 2.3B |

## Training

### Training Environment

Training runs on NVIDIA H100 NVL GPUs (94 GB VRAM each) with 756 GB RAM. We use a shuffle buffer of 256K samples.

### Data Preparation

BridgeDataV2 and Fractal are part of the [Open X-Embodiment](https://robotics-transformer-x.github.io/) dataset. Follow [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod) for data preparation.

### Running Training

**Fractal (single GPU):**

```bash
bash train.sh
```

The default configuration uses:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/train.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /data/ \
  --dataset_name fractal20220817_data \
  --run_root_dir /data/wandbrun \
  --use_l1_regression True \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --max_steps 200005 \
  --save_freq 5000 \
  --image_aug True \
  --lora_rank 32 \
  --num_actions_chunk 8 \
  --num_actions_per_token 8 \
  --num_blocks 2 \
  --mode "mul" \
  --action_head_name "funnel"
```

For LIBERO training, refer to [LIBERO.md](LIBERO.md).

### Training Parameters

| Parameter | Description | Default |
|---|---|---|
| `--vla_path` | Base VLM checkpoint (HF Hub or local) | `openvla/openvla-7b` |
| `--use_l1_regression` | Use L1 regression action head | `True` |
| `--use_diffusion` | Use diffusion-based action head (DDIM) | `False` |
| `--use_film` | Use FiLM language-vision conditioning | `False` |
| `--use_proprio` | Include proprioceptive state in input | `False` |
| `--num_images_in_input` | Number of camera images (1 = 3rd person only) | `1` |
| `--lora_rank` | LoRA rank for fine-tuning | `32` |
| `--image_aug` | Enable random crop image augmentation | `True` |
| `--num_actions_chunk` | Total action sequence length | — |
| `--num_actions_per_token` | Actions decoded per token | — |
| `--num_blocks` | MLPResNet depth in action head | — |
| `--mode` | Prediction mode (`"mul"` for ensemble) | `"mul"` |
| `--action_head_name` | Action head type (`"mlp"`, `"funnel"`, `"fel"`) | `"funnel"` |

## Evaluation

### LIBERO

Follow [LIBERO.md](LIBERO.md) for LIBERO setup, training, and evaluation.

**Quick multi-GPU evaluation:**

```bash
# Using the shell launcher (recommended)
CKPT_DIR=/path/to/ckpts TASK_SUITE=libero_goal bash run_libero_goal_eval.sh

# Using the Python script
python experiments/robot/libero/batch_eval.py \
  --dir /path/to/ckpts \
  --task_suite libero_goal \
  --devices 0 1 2 3 4 5 6 7
```

### SimplerEnv

> **Important:** Install SimplerEnv _before_ installing effvla, because installing TensorFlow 2.15 may break the CUDA environment for PyTorch.

<details>
<summary>SimplerEnv installation steps</summary>

```bash
conda create -n simpler_env python=3.10
conda activate simpler_env

git clone https://github.com/LukeLIN-web/simplerenv.git --recurse-submodules
pip install numpy==1.24.4  # numpy >= 1.26 causes issues in SimplerEnv

cd simplerenv/ManiSkill2_real2sim
pip install -e .
cd ..

pip install -e .

cd ..
git clone https://github.com/LukeLIN-web/vote.git
cd vote
pip install -e .
cd ..

sudo apt install ffmpeg

cd simplerenv
pip install tensorflow==2.15.0
pip install "tensorflow[and-cuda]==2.15.1"  # TensorFlow GPU support

# If you encounter: libtorch_cuda.so: undefined symbol: ncclCommRegister
# Re-install torch and torchvision:
pip install torch==2.3.1 torchvision==0.18.1

pip install mediapy pandas gymnasium==0.28.1
```

</details>

## Results

### SimplerEnv (WidowX, Visual Matching)

| Method | Put Spoon | Put Carrot | Stack Block | Put Eggplant | Avg. | Latency (ms) ↓ | Speedup ↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RT-1-X | 0.0 | 4.2 | 0.0 | 0.0 | 1.1 | — | — |
| Octo | 47.2 | 9.7 | 4.2 | 56.9 | 30.0 | — | — |
| OpenVLA | 0.0 | 0.0 | 0.0 | 4.1 | 1.0 | 240 | 1.00 |
| RoboVLM | 29.2 | 25.0 | 12.5 | 58.3 | 31.3 | — | — |
| OpenPI0 | 29.1 | 0.0 | 16.6 | 62.5 | 27.1 | 470 | 0.50 |
| SpatialVLA | 16.7 | 25.0 | 29.2 | 100.0 | 42.7 | 400 | 0.60 |
| CogACT | 71.7 | 50.8 | 15.0 | 67.5 | 51.3 | 220 | 1.09 |
| **VOTE (Ours)** | **58.3** | **29.2** | **50.0** | **95.8** | **58.3** | **78** | **3.1** |

### LLAMA3.2-1B-VLA

| Model | Params (B) | libero_spatial | libero_object | libero_goal | libero_10 | Average SR | VRAM (GB) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LLAMA3.2-1B-VLA | 2.3 | 98.4% | 96.0% | 95.0% | 82.4% | 92.95% | 4.34 |

**Edge deployment latency:**
- Jetson AGX Orin: 108 ms (chunk = 8, ≈ 73 Hz)
- Jetson Nano: 387 ms

## Citation

If you find this work useful, please cite [our paper](https://arxiv.org/abs/2507.05116):

```bibtex
@misc{lin2025votevisionlanguageactionoptimizationtrajectory,
      title={VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting}, 
      author={Juyi Lin and Amir Taherin and Arash Akbari and Arman Akbari and Lei Lu and Guangyu Chen and Taskin Padir and Xiaomeng Yang and Weiwei Chen and Yiqian Li and Xue Lin and David Kaeli and Pu Zhao and Yanzhi Wang},
      year={2025},
      eprint={2507.05116},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05116}, 
}
```

## License

This project is released under the [MIT License](LICENSE).
