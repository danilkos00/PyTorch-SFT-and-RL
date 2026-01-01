# PyTorch-SFT-and-RL

Utilities and training pipeline for fine-tuning **Qwen2.5-Math-1.5B** on the GSM8K dataset to improve reasoning abilities in solving school-level math problems and FastAPI inference with fine-tuned model.

---

## FastAPI inference

Run local API:
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Building and Running Docker Image

### Build the Docker image
```bash
docker build -t <image_name> .
```

### Run the container
```bash
docker run -d -p 80:80 <image_name>
```

---

## Project Overview
This project implements:
- **SFT (Supervised Fine-Tuning)** on GSM8K.
- **GRPO (Group Relative Policy Optimization)** RL algorithm.
- Custom utilities for both SFT and GRPO training.
- FastAPI app for model inference

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

## Demo Notebook
### A Google Colab notebook is available to quickly test the fine-tuned model:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danilkos00/PyTorch-SFT-and-RL/blob/main/pytorch-posttraining-demo.ipynb)

---

## Examples

**Question**: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?

**Generated answer**: <think>Kylar needs to pay 8 glasses at 5 dollars each, so that's 8 * 5 = $<<8*5=40>>40. The remaining 8 glasses cost 60% of $5, which is 0.6 * 5 = $<<0.6*5=3>>3 each. So he needs to pay 8 * 3 = $<<8*3=24>>24 for the remaining glasses. In total, he needs to pay $40 + $24 = $<<40+24=64>>64 for all the glasses.</think> <answer>#### 64</answer>

---

**Question**: Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?

**Generated answer**: <think>If Seattle has 20 sheep, then Charleston has 20 sheep * 4 = <<20*4=80>>80 sheep. Toulouse has 80 sheep * 2 = <<80*2=160>>160 sheep. Together, they have 160 sheep + 80 sheep + 20 sheep = <<160+80+20=260>>260 sheep.</think> <answer>#### 260</answer>

---

**Question**: Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?

**Generated answer**: <think>Eliza's overtime pay is 1.2 * $10 = 1.2*10=12 per hour. Eliza worked 45 - 40 = 5 hours of overtime.\nEliza's earnings for the overtime pay is 5 * $12 = 5*12=60. Eliza's earnings for the first 40 hours is 40 * $10 = 400. Eliza's total earnings for the week is $400 + $60 = 400 + 60 = 460.</think> <answer>#### 460</answer>
