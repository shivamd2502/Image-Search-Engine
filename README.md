# Scalable Land Cover Classification using Parameter-Efficient Vision-Language Models

Official implementation accompanying the paper *"Scalable Land Cover Classification using Parameter-Efficient Vision-Language Models"* (Department of Instrumentation and Control Engineering, Vishwakarma Institute of Technology, Pune).

**Authors:** Dr. Kapil Mundada, Raghav Deshpande, Shivam Dapkekar

---

## Overview

Land Cover Classification (LCC) is a cornerstone task in remote sensing, underpinning climate monitoring, resource management, and ecological modeling. The continuous influx of high-resolution Earth Observation data demands models that are both highly accurate and computationally efficient — a combination that full fine-tuning of large pre-trained models struggles to deliver.

This project introduces a **Parameter-Efficient Fine-Tuning (PEFT)** framework built on top of a **frozen CLIP backbone**. Instead of updating the entire model, we train only a lightweight **Adapter module** and dedicated **classification heads**, achieving state-of-the-art accuracy while updating less than **1% of total parameters**.

## Key Results

| Metric | Value |
|---|---|
| Test Accuracy | **99.88%** |
| Trainable Parameters | 528,385 (0.63% of total) |
| Frozen CLIP Parameters | 86,400,000 (99.37%) |
| Training Time (5 epochs) | ~30 minutes |
| Full Fine-Tuning Equivalent | ~82 hours (163× reduction) |
| Test Set Size | 846 samples |

## Method

### Architecture

- **Backbone:** Frozen CLIP (ViT-B/32, 512-dim embedding space)
- **Adapter Module:** Lightweight bottleneck network (`Wdown`, `Wup` ∈ ℝ^512×512) inserted between the frozen image encoder and the classification heads
- **Classification Heads:** Dual heads (image + text) projecting embeddings to 4 land cover class logits
- **Learnable Temperature (τ):** Controls similarity sharpness in the contrastive objective

### Loss Function

A combined objective balances semantic alignment with discriminative classification:

```
L_total = L_contrastive + λ_ce · L_classification
```

- **Contrastive Loss** — symmetric cross-entropy over image-text similarity, preserving CLIP's pre-trained semantic alignment
- **Classification Loss** — categorical cross-entropy on both image and text classification heads

### Text Prompts

Each land cover class (Cloudy, Desert, Green Area, Water) is paired with 10 diverse natural-language prompt variants (e.g., *"a satellite image of a desert region"*, *"an aerial view of sandy desert terrain"*). Random prompt sampling during training acts as implicit data augmentation.

### Embedding Cache System

An embedding cache manager enables fast, incremental image retrieval:
- Detects new/removed images via set difference against the cache
- Encodes only new images, avoiding redundant computation
- Uses a Watchdog-based file system monitor to trigger updates automatically
- Achieves sub-100ms query inference and <20ms top-10 retrieval

## Dataset

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| Cloudy | 161 | 34 | 230 | 425 |
| Desert | 121 | 26 | 173 | 320 |
| Green Area | 157 | 34 | 224 | 415 |
| Water | 154 | 33 | 219 | 406 |
| **Total** | **593** | **127** | **846** | **1,566** |

- Stratified 70/15/15 train-validation-test split
- All images resized to 224×224 and normalized using CLIP's standard preprocessing

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 2 × 10⁻⁴ |
| Weight Decay | 0.01 |
| Scheduler | Cosine annealing (T_max = 10) |
| Batch Size | 32 |
| Epochs | 5 |
| Classification Loss Weight (λ_ce) | 1.0 |
| Hardware | NVIDIA GPU (CUDA) |

## Results

### Per-Class Test Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Cloudy | 1.00 | 1.00 | 1.00 | 230 |
| Desert | 1.00 | 0.994 | 0.997 | 173 |
| Green Area | 1.00 | 1.00 | 1.00 | 224 |
| Water | 1.00 | 1.00 | 1.00 | 219 |
| **Weighted Avg** | **1.00** | **1.00** | **1.00** | **846** |

Only a single misclassification occurred across the entire test set — a desert sample predicted as green area, likely a borderline case of sparse vegetation in semi-arid terrain.

### Embedding Cache Performance

| Metric | Value |
|---|---|
| Embedding Dimension | 512 |
| Per-Image Memory | 2 KB |
| Cache Size (846 images) | 1.7 MB |
| Initial Computation | ~45s |
| Per-Image Encode | ~50ms |
| Query Inference | <5ms |
| Top-10 Retrieval | <20ms |

## Repository Structure

```
.
├── data/                   # Dataset directory (train/val/test splits)
├── models/                 # Adapter, classification heads, CLIP wrapper
├── losses/                 # Contrastive + classification loss implementations
├── cache/                  # Embedding cache manager (incl. Watchdog monitor)
├── prompts/                # Text prompt templates for each land cover class
├── train.py                # Training script
├── evaluate.py             # Test set evaluation script
├── configs/                # Training configuration files
└── README.md
```

> Adjust this structure to match your actual repository layout.

## Limitations & Future Work

- Currently uses RGB imagery only; multi-spectral data (e.g., near-infrared) could improve discrimination of borderline cases
- Fixed 224×224 input resolution may lose detail in high-resolution imagery
- Evaluated on four fundamental classes over 846 test samples; larger and more fine-grained datasets would strengthen generalization claims

Planned directions include multi-spectral data integration, temporal sequence modeling for land cover change detection, hierarchical fine-grained classification, active learning, and federated deployment across distributed satellite gateways.

## Citation

If you use this work, please cite:

```bibtex
