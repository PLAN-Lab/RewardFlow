# $\color{orange}{\textbf{{[CVPR 2026]}}}$ RewardFlow: Generate Images by Optimizing What You Reward

<div align="center">
  <a href="https://plan-lab.github.io/rewardflow"><img src="https://img.shields.io/badge/Project-Website-blue?style=for-the-badge&logo=googlechrome"></a>
  <a href="https://arxiv.org/abs/2604.08536"><img src="https://img.shields.io/badge/arXiv-2604.08536-b31b1b.svg?style=for-the-badge"></a>
  <a href="https://github.com/PLAN-Lab/RewardFlow"><img src="https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github"></a>
  <a href="https://huggingface.co/onkarsus13/RewardFlow"><img src="https://img.shields.io/badge/Model-HuggingFace-orange?style=for-the-badge&logo=huggingface"></a>
</div>

<p style="text-align: justify;">RewardFlow is an inversion-free framework that steers pretrained diffusion and flow-matching models at inference time using multi-reward Langevin dynamics. It combines differentiable rewards for semantic alignment, perceptual fidelity, localized grounding, object consistency, and human preference. A prompt-aware adaptive policy modulates reward weights and sampling steps, while a clean-latent KL regularizer preserves the original latent structure. Across editing and compositional generation benchmarks, RewardFlow achieves state-of-the-art zero-shot fidelity and alignment without fine-tuning.</p>

[![RewardFlow Teaser](docs/teaser.jpg)](docs/CVPR2_abs_diagram_8_compressed.pdf)

## Overview

This repository contains RewardFlow code and scripts for:
- running single-image inference (`test_rewardflow.py`)
- generating batch edited images (`pie_solver.py`)
- computing evaluation metrics (`eval.py`)
- All code tested on A100 80GB, There is SDP-Numel error with Ada and Litz architechures

## Repository Layout

- `test_rewardflow.py`: quick single-image RewardFlow inference test.
- `download.py`: helper script to download model files from Hugging Face.

## End-to-End Setup and Run

Run all commands from repo root:

```bash
cd /data/home/onkar/offical_code/RewardFlow
```

### 1) Create Environment

```bash
conda create -n rewardflow python=3.10 -y
conda activate rewardflow
pip install --upgrade pip
```

Install PyTorch for your CUDA version (example for CUDA 12.4):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install RewardFlow / diffusers code and runtime dependencies:

```bash
pip install -e ".[torch]"
pip install torchmetrics transformers bitsandbytes sentencepiece opencv-python timm pillow
```

Notes:
- `eval.py` uses `torchmetrics`, `transformers`, and a DINO model loaded via `torch.hub`.
- First metric run may download model artifacts (internet required).

### 2) Download RewardFlow Weights

```bash
export HF_TOKEN=hf_your_token_here   # only needed for private access
python download.py \
  --repo-id onkarsus13/RewardFlow \
  --local-dir /data/onkar/models/RewardFlow
```




### 3) Run Quick Inference Test (`test_rewardflow.py`)

`test_rewardflow.py` has hardcoded paths. Update these two values first:
- `model_dir` (set to your downloaded weights path, e.g. `/data/onkar/models/RewardFlow`)
- input image path in `Image.open(...)` (point to a real image under `annotation_images`)

Then run:

```bash
python test_rewardflow.py
```

## Contact

Please contact to ```onkarsus13@gmail.com``` if you have face any challenges regarding the running the code.

## Citation

:star: If you find this work useful, please cite our [paper](https://arxiv.org/abs/2604.08536)

```bibtex
@inproceedings{rewardflow2026,
  title     = {RewardFlow: Generate Images by Optimizing What You Reward},
  author    = {Susladkar, Onkar Kishor and Jang, Dong-Hwan and Prakash, Tushar and Juvekar, Adheesh Sunil and Shah, Vedant and Barik, Ayush and Bashir, Nabeel and Wahed, Muntasir and Shrirao, Ritish and Lourentzou, Ismini},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
