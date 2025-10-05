# VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting 🚀🤖

**📄 Paper:** https://arxiv.org/abs/2507.05116  

 [\[🤗 Model Zoo\]](https://huggingface.co/collections/juyil/vote-vision-language-action-model-686f5dac2775080477a86cdf) 

## News 
- `2025/09/22` ✨ Released **VOTE llama3.2-1B-VLA** model 👉  👉 [script](https://github.com/LukeLIN-web/vote/blob/main/experiments/speed/llama3-1B.py)  
- `2025/07/10`: 🎉 We release the [Vote 1.0](https://huggingface.co/collections/juyil/vote-vision-language-action-model-686f5dac2775080477a86cdf).  ➡️ No need for **complex tokenizers** — migrate to a new VLM with just **2 lines of code** ⚡️  


# Installation


```bash
conda create -n effvla python=3.10 -y
conda activate effvla

cd ~/ 
git clone https://github.com/LukeLIN-web/vote.git
cd vote
pip install -e .

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install flash-attn==2.6.1 --no-build-isolation
```


### Quick start

```bash
cd experiments/speed/
python effvla.py
```

### Speed

We have speed measurement codes under `experiments/speed/`. 

## Installation on AGX Orin

```bash
python -m venv orin
source orin/bin/activate

# Install transformers and other dependencies
pip3 install packaging ninja transformers==4.51.0 tokenizers==0.21.4 timm==0.9.10 diffusers==0.32.2

# Install Tensorflow 2.15.0
pip3 install tensorflow==2.15.0

# Install Tensorflow's addons from source
git clone https://github.com/tensorflow/addons
cd addons
pip3 install -e .

# Clone QwenVLA repo and pip install to download dependencies
git clone https://github.com/LukeLIN-web/vote.git vote
cd vote
pip3 install -e .
cd ..
# This step will install the wrong versions of torch, torchvision that would not work on Jetson machine.
# We need to install the precompiled wheels for Jetson

# Install torch, torchvision, torchaudio using Nvidia's precompiled wheels for Jetson. 
# torch: https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl
# torchvision: https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl
pip3 install torch*.whl torchvision*.whl

# This step will output dependency error as followed, ignore them.
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is 
# the source of the following dependency conflicts.
# effvla 0.0.1 requires torchvision==0.18.1, but you have torchvision 0.18.0a0+6043bc2 which is incompatible.
```


# Training

## Training Setting

Training runs on NVIDIA H100 NVL GPUs (94 GB VRAM each) with 756 GB RAM. We set a shuffle buffer of 256K samples.


## Steps

BridgeDataV2 and Fractal are a part of Open X-Embodiment Dataset,  the preparation follows: [rlds_dataset_mod](https://github.com/kpertsch/rlds_dataset_mod)

Then run train script:
```
bash train.sh
```


### Q&A

For libero, read `LIBERO.md` carefully.


```
No module named prismatic
No module named experiments
```

It is because you don't install correctly. Check `pip list | grep effvla`. 

If you run into any issues, please open a new GitHub issue.


# Evaluation

For libero evaluation, follow `LIBERO.md`.

## SimplerEnv
For SimplerEnv installation:

You may install SimplerEnv before you install effvla.
Because install tensorflow 2.15 will break the cuda env in torch.


```bash
conda create -n simpler_env python=3.10
conda activate simpler_env

git clone  https://github.com/LukeLIN-web/simplerenv.git --recurse-submodules
pip install numpy==1.24.4 # important, numpy >=1.26 has problem in simpler env

cd simplerenv/ManiSkill2_real2sim
pip install -e .

cd simplerenv
pip install -e .

git clone https://github.com/LukeLIN-web/vote.git
cd vote
pip install -e .

sudo apt install ffmpeg

cd simplerenv
pip install tensorflow==2.15.0
pip install tensorflow[and-cuda]==2.15.1 # tensorflow gpu support


# we  need to install torch and torchvision again if it shows libtorch_cuda.so: undefined symbol: ncclCommRegister 
pip install torch==2.3.1 torchvision==0.18.1
pip install mediapy pandas
pip install gymnasium==0.28.1
```

### Results

Evaluation results on the WidowX robot in the SimplerEnv Visual Matching setting. 

| Method                  | Put Spoon | Put Carrot | Stack Block | Put Eggplant | Avg. | Latency (ms) ↓ | Speed up ↑ |
|-------------------------|-----------|------------|--------------|---------------|------|----------------|-------------|
| RT-1-X                  | 0.0       | 4.2        | 0.0          | 0.0           | 1.1  | --             | --          |
| Octo             | 47.2      | 9.7        | 4.2          | 56.9          | 30.0 | --             | --          |
| OpenVLA                 | 0.0       | 0.0        | 0.0          | 4.1           | 1.0  | 240            | 1.00        |
| RoboVLM    | 29.2      | 25.0       | 12.5         | 58.3          | 31.3 | --             | --          |
| Openpi0                 | 29.1      | 0.0        | 16.6         | 62.5          | 27.1 | 470            | 0.50        |
| SpatialVLA | 16.7      | 25.0       | 29.2         | 100.0         | 42.7 | 400            | 0.60        |
| CogACT                  | 71.7      | 50.8       | 15.0         | 67.5          | 51.3 | 220            | 1.09        |
| __Ours__               | __58.3__  | __29.2__   | __50.0__     | __95.8__      | __58.3__ | __78__     | __3.1__    |



LLAMA3.2-1B-VLA:

Jetson AGX Orin: Inference latency 108 ms (chunk = 8, ≈ 73 Hz)
Jetson Nano: Inference latency 387 ms

| Model          | Parameters (B) | libero_spatial SR (%) | libero_object SR (%) | libero_goal SR (%) | libero_10 SR (%) | Average (SR%)  | VRAM(GB) |
|----------------|----------------|------------------------------|-----------------------------|---------------------------|-------------------------|---------|------|
| LLAMA3.2-1B-VLA| 2.3            | 98.4                        |      96                       |         95%                  |            82.4%             |   92.95%      | 4.34  |

The accuracy curve is shown here: https://www.notion.so/How-much-data-need-for-small-VLA-fitting-2796566ea37a80ec8334d65fe0d365cd?source=copy_link

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2507.05116):

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
