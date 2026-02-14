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

Install LIBERO package.
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```



## Evaluation

```bash
python experiments/robot/libero/batch_eval.py  --hf_ckpts
```
When you have multiple checkpoints, the results could be plotted with `batch_plot.ipynb`.

## Training

Dataset

```
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds
```


We fine-tune on OpenVLA model using AdamW with a learning rate of 1e-4. Fine-tuning employs LoRA with rank r = 32 and α = 16. By default, the model is finetuned to output one token $\texttt{<ACT>}$ with a chunk size of 8.

