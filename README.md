# PyTorch-SFT-and-RL

Utilities and training pipeline for fine-tuning **Qwen2.5-Math-1.5B** on the GSM8K dataset to improve reasoning abilities in solving school-level math problems.

---

## Project Overview
This project implements:
- **SFT (Supervised Fine-Tuning)** on GSM8K.
- **GRPO (Group Relative Policy Optimization)** RL algorithm.
- Custom utilities for both SFT and GRPO training.

After fine-tuning, the model shows significantly improved reasoning abilities on math tasks.

---

## Goals
This is an **educational project** to explore post-training methods for large language models, focusing on reasoning improvement and reinforcement learning.

---

## Technical Details
- **Base model:** Qwen2.5-Math-1.5B  
- **Dataset:** GSM8K  
- **Frameworks:** Hugging Face Transformers, PyTorch  
- **Hardware:** 2 × NVIDIA T4 GPUs  
- **Training steps:**  
  - Stage 1: Supervised Fine-Tuning (SFT)  
  - Stage 2: GRPO  

---

## Results on math test dataset
| Stage | Accuracy |
|-------|-----------|
| After SFT | 0.64 |
| After GRPO | 0.73 |


---

## How to Use

### Install dependencies
```bash
git clone https://github.com/danilkos00/PyTorch-SFT-and-RL.git -qq
cd PyTorch-SFT-and-RL
pip install -r requirements.txt
```

---

## Demo Notebook
### A Google Colab notebook is available to quickly test the fine-tuned model:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danilkos00/PyTorch-Transformer/blob/main/Transformer_demo.ipynb)

---

## Examples

(Examples of model outputs will be added soon.)

