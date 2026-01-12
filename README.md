# Irony Detection NLP Benchmark

A comprehensive benchmark comparing multiple approaches for irony detection on Twitter data, from classic ML to state-of-the-art transformers and large language models.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Methodology](#methodology)
- [Results](#results)
- [Key Insights](#key-insights)
- [Installation](#installation)
- [Usage](#usage)
- [Hardware](#hardware)
- [License](#license)
- [References](#references)

## Overview

This project benchmarks **8 different models** for binary irony detection (Irony vs Not Irony) on the TweetEval dataset. The goal is to evaluate the trade-off between **performance** (F1-Score) and **efficiency** (inference speed, model size, training time).

Note: Dataset might reload due to VRAM clearance (preventing OOM).
### Approaches Compared

| Category | Models |
|----------|--------|
| Fine-tuned Transformers | Cardiff RoBERTa-2021-124M, Cardiff RoBERTa-Base |
| Few-Shot Learning | SetFit (MPNet) |
| Classic ML + Embeddings | XGBoost, Random Forest (with Jina v3) |
| LLM (In-Context Learning) | Qwen 2.5-7B-Instruct |

## Dataset

**TweetEval - Irony Subset** (derived from SemEval-2018 Task 3)

| Split | Samples | Not Irony | Irony | Balance |
|-------|---------|-----------|-------|---------|
| Train | 2,862 | ~1,430 | ~1,432 | ~50% (Balanced) |
| Validation | 955 | ~470 | ~485 | ~50% (Balanced) |
| Test | 784 | 473 | 311 | ~40% (Imbalanced) |

> **Note:** The test set reflects a realistic distribution where irony is the minority class (~40%). Therefore, **F1-Score on the Irony class** is the primary evaluation metric, not accuracy.

### Preprocessing

Cardiff RoBERTa models require Twitter-specific preprocessing:
- `@username` → `@user`
- `https://...` → `http`

Other models (SetFit, Jina embeddings, Qwen) use raw text.

## Models

### 1. Cardiff RoBERTa-2021-124M (SOTA)

- **Type:** Fine-tuned Transformer
- **Pre-training:** 124M tweets
- **Source:** [cardiffnlp/twitter-roberta-base-2021-124m-irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-irony)

### 2. Cardiff RoBERTa-Base

- **Type:** Fine-tuned Transformer
- **Pre-training:** Standard RoBERTa + Twitter fine-tuning
- **Source:** [cardiffnlp/twitter-roberta-base-irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony)

### 3. SetFit (MPNet)

- **Type:** Few-Shot Learning (Contrastive + Logistic Regression)
- **Base Model:** `sentence-transformers/paraphrase-mpnet-base-v2`
- **Training:** 34 minutes, no hyperparameter tuning

### 4-7. Classic ML (XGBoost, Random Forest)

- **Embeddings:** Jina v3 (`jinaai/jina-embeddings-v3`)
- **Dimensions:** 256d and 768d (Matryoshka truncation)
- **Hyperparameter Search:** RandomizedSearchCV with 3-fold CV

### 8. Qwen 2.5-7B-Instruct

- **Type:** LLM with Few-Shot In-Context Learning
- **Inference:** Local via Ollama
- **Few-Shot:** 10 balanced examples sampled dynamically from train set

## Methodology

### Evaluation Protocol

1. All models evaluated on the **same test set** (784 samples)
2. Preprocessing applied only where required (Cardiff models)
3. Metrics computed using scikit-learn

### Metrics

| Metric | Description |
|--------|-------------|
| **F1-Score (Irony)** | Primary metric — harmonic mean of precision and recall for the Irony class |
| Accuracy | Secondary metric — overall correctness |
| Precision (Irony) | Proportion of correct Irony predictions |
| Recall (Irony) | Proportion of actual Irony cases detected |
| Inference Time | Time to classify all 784 test samples |

### Training Details

| Model | Training Time | Hyperparameter Tuning |
|-------|---------------|----------------------|
| Cardiff RoBERTa | Pre-trained | N/A |
| SetFit | ~34 min | None (default config) |
| XGBoost | ~20-28 min | RandomizedSearchCV (100-200 iter) |
| Random Forest | ~14-16 min | RandomizedSearchCV (50-70 iter) |
| Qwen | N/A | N/A (few-shot) |

## Results

### Performance Benchmark

| Model | Accuracy | F1 (Irony) | Precision | Recall | Inference Time | Parameters |
|-------|----------|------------|-----------|--------|----------------|------------|
| **Cardiff RoBERTa-2021-124M (SOTA)** | **78.57%** | **75.07%** | 70% | 81% | 1.72s | 125M |
| SetFit (MPNet) | 74.00% | 72.00% | 63% | 83% | 0.96s | 109M |
| Cardiff RoBERTa-Base | 73.47% | 62.72% | 71% | 56% | 9.54s | 125M |
| **XGBoost (Jina v3 256d)** | **71.43%** | **67.44%** | 62% | 75% | 0.03s | <1M |
| XGBoost (Jina v3 768d) | 66.84% | 61.08% | 57% | 66% | 0.04s | <1M |
| Random Forest (Jina v3 768d) | 65.82% | 59.15% | 56% | 62% | 0.02s | <1M |
| Random Forest (Jina v3 256d) | 65.69% | 58.16% | 56% | 60% | 0.06s | <1M |
| Qwen 2.5-7B (Few-Shot 10)* | 70.79% | 72.64%* | 58% | 98% | 2674s | 7,000M |

> **\*Warning:** Qwen's F1 score is misleading. With 98% recall but only 58% precision, the model systematically over-predicts IRONY. This is bias, not performance.

### Speed Comparison

| Model | Throughput | Speedup vs LLM |
|-------|------------|----------------|
| Random Forest (768d) | 39,200 samples/sec | 133,700x |
| XGBoost (256d) | 26,133 samples/sec | 89,140x |
| SetFit (MPNet) | 817 samples/sec | 2,785x |
| Cardiff RoBERTa-2021-124M | 456 samples/sec | 1,555x |
| Qwen 2.5-7B | 0.3 samples/sec | 1x |

## Key Insights

### 1. SOTA Performance Requires Domain-Specific Pre-training

Cardiff RoBERTa-2021-124M achieves the best F1 (75.07%) due to pre-training on 124M tweets. This domain knowledge cannot be replicated without massive computational resources.

### 2. SetFit is an Excellent Trade-off

SetFit achieves **96% of SOTA performance** with:
- 34 minutes training time
- No hyperparameter tuning
- 1.8x faster inference

### 3. XGBoost 256d Beats the Older Cardiff Model

| Metric | XGBoost 256d | Cardiff Base | Delta |
|--------|--------------|--------------|-------|
| F1 (Irony) | 67.44% | 62.72% | **+4.7 pts** |
| Inference | 0.03s | 9.54s | **318x faster** |
| Parameters | <1M | 125M | **125x smaller** |

### 4. Curse of Dimensionality Affects Classic ML

256 dimensions outperform 768 dimensions for tree-based models:
- 768 features / 3,817 samples = 5:1 ratio → Overfitting
- 256 features / 3,817 samples = 15:1 ratio → Better generalization

### 5. LLMs Are Not Suitable for This Task

Despite Qwen 2.5 being a capable foundation model (used by Jina v4 embeddings), general-purpose LLMs are inefficient for binary classification:
- 89,000x slower than XGBoost
- Biased toward IRONY class (98% recall, 58% precision)
- Impractical for production

### 6. Classic ML Has a Performance Ceiling

Tree-based models plateau around 67% F1. Transformers are necessary for capturing the linguistic nuances required for irony detection.

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU acceleration)
- Ollama (for Qwen inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/irony-detection-nlp-benchmark.git
cd irony-detection-nlp-benchmark

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Requirements

```
# requirements.txt
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
transformers
datasets
setfit
sentence-transformers
accelerate
einops
ipykernel
tqdm
```

### Ollama Setup (for Qwen)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull qwen2.5:7b-instruct
```

## Usage

### Run Complete Benchmark

```bash
jupyter notebook benchmark.ipynb
```

### Run Individual Experiments

```python
# Load dataset
from datasets import load_dataset
dataset = load_dataset("tweet_eval", "irony")

# Cardiff SOTA
from transformers import pipeline
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-2021-124m-irony")

# SetFit
from setfit import SetFitModel
model = SetFitModel.from_pretrained("models/setfit-mpnet-irony")

# XGBoost
import joblib
model = joblib.load("models/xgb_jina_v3_256.joblib")
```

## Hardware

Experiments conducted on:

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 4060 (8GB VRAM) |
| CPU | AMD/Intel (for RF) |
| RAM | 8GB+ |
| CUDA | 12.1 |

## License

Apache-2.0 License

## References

### Dataset

- Barbieri, F., Camacho-Collados, J., Neves, L., & Espinosa-Anke, L. (2020). [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://arxiv.org/abs/2010.12421). EMNLP 2020.

### Models

- Cardiff NLP: [twitter-roberta-base-2021-124m-irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-irony)
- SetFit: [Tunstall et al., 2022](https://arxiv.org/abs/2209.11055)
- Jina Embeddings v3: [jinaai/jina-embeddings-v3](https://huggingface.co/jinaai/jina-embeddings-v3)
- Qwen 2.5: [Qwen Team, 2024](https://huggingface.co/Qwen)

---

**Author:** El Basraoui Anass  
**Date:** January 2026
```