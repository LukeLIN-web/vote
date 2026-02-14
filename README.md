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

We provide speed measurement scripts under `experiments/speed/`.

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

## Training

### Training Environment

Training runs on NVIDIA H100 NVL GPUs (94 GB VRAM each) with 756 GB RAM. We use a shuffle buffer of 256K samples.

### Data Preparation

BridgeDataV2 and Fractal are part of the [Open X-Embodiment](https://robotics-transformer-x.github.io/) dataset. Follow [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod) for data preparation.

### Running Training

```bash
bash train.sh
```

For LIBERO training, refer to [LIBERO.md](LIBERO.md).

## Evaluation

### LIBERO

Follow [LIBERO.md](LIBERO.md) for LIBERO evaluation.

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
