# MI-AIGIQA: Mutual Information-Guided Multimodal AI-Generated Image Quality Assessment

> **ECCV 2026** | [Paper](#) | [arXiv](#) | [Project Page](#)

---

## Overview

**MI-AIGIQA** is a multimodal image quality assessment (IQA) framework for AI-generated images. It leverages the mutual information (MI) between visual and textual features to learn compact, information-rich representations for quality prediction. Built on top of the pretrained [ImageReward](https://github.com/THUDM/ImageReward) (BLIP) backbone, it introduces a two-stage optimization strategy that jointly maximizes the MI between modalities and minimizes quality prediction error.

<p align="center">
  <!-- Replace with your architecture figure -->
  <img src="assets/framework.png" width="85%" alt="MI-AIGIQA Framework">
</p>

### Key Components

| Module | File | Description |
|--------|------|-------------|
| **MMIM** | `model.py` | Main model: BLIP backbone + cross-attention fusion + quality regressor |
| **MMILB** | `modules/encoders.py` | Forward MI lower bound estimator (Gaussian variational) |
| **CPC** | `modules/encoders.py` | Backward MI estimator via Noise Contrastive Estimation |
| **ImageTextRegression** | `model.py` | Cross-attention fusion module and MLP quality predictor |

---

## Requirements

```bash
pip install torch torchvision          # PyTorch (>=1.12 recommended)
pip install image-reward               # ImageReward backbone
pip install einops scipy tqdm pillow   # Core dependencies
pip install thop                       # FLOPs computation (test_efficiency.py)
pip install matplotlib scikit-learn    # Gaussian visualization (optional)
```

We recommend creating a dedicated conda environment:

```bash
conda create -n miaigiqa python=3.9
conda activate miaigiqa
pip install -r requirements.txt
```

---

## Datasets

The following four AIGIQA benchmarks are supported:

| Dataset | # Images | Annotation Types | Reference |
|---------|----------|-----------------|-----------|
| **AGIQA-1k** | 1,080 | quality | [Li et al., 2023](https://arxiv.org/abs/2306.04990) |
| **AGIQA-3k** | 2,982 | quality, consistency | [Li et al., 2023](https://arxiv.org/abs/2306.04990) |
| **AIGCIQA2023** | 2,400 | quality, authenticity, consistency | [Wang et al., 2023](https://arxiv.org/abs/2307.00211) |
| **PKU-AIGIQA** | 2,800 | quality, authenticity, consistency | [Yuan et al., 2023](https://arxiv.org/abs/2307.01911) |

### Data Directory Structure

Place the dataset images under the `Data/` folder following this structure:

```
Data/
├── AGIQA-1k/
│   └── Image/          # All AGIQA-1k images (.jpg / .png)
├── AGIQA-3k/
│   └── Image/          # All AGIQA-3k images
├── AIGCIQA2023/
│   ├── <subfolder_A>/  # Subfolders as referenced in the CSV
│   └── <subfolder_B>/
└── PKU-AIGIQA/
    └── Image/          # All PKU-AIGIQA images
```

> **Note:** The `Data/` directory is not included in this repository. Please download each dataset from the links above and arrange the images as shown.

### 10-Fold Cross-Validation Splits

Pre-generated 10-fold splits are provided in the `Database/` folder:

```
Database/
├── AGIQA-1k/
│   ├── 0/              # Fold 0 (index starts from 0 for AGIQA-1k)
│   │   ├── train.csv
│   │   └── val.csv
│   └── 1-9/            # Folds 1–9
├── AGIQA-3k/
│   ├── 1/              # Fold 1 (index starts from 1 for other datasets)
│   │   ├── train.csv
│   │   └── val.csv
│   └── 2-10/           # Folds 2–10
├── AIGCIQA2023/
│   └── 1-10/
└── PKU-AIGIQA/
    └── 1-10/
```

For cross-dataset experiments (TABLE 2), a `full.csv` is used for each dataset — covering the entire dataset without fold splitting. Place these files directly under each dataset's `Database/` subdirectory (e.g., `Database/AGIQA-3k/full.csv`).

---

## Training

### TABLE 1 — Single-Dataset 10-Fold Cross-Validation

Edit the configuration block at the top of `MI-AIGIQA_train.py`:

```python
dataset  = 'AGIQA-3k'   # 'AGIQA-1k' | 'AGIQA-3k' | 'AIGCIQA2023' | 'PKU-AIGIQA'
mos_type = 'quality'     # 'quality' | 'consis' | 'authn' (authn not available for AGIQA)
main_lr  = 1e-5          # Learning rate for main model
mmilb_lr = 5e-6          # Learning rate for MI estimator (Stage 0)
alpha    = 0.1           # Weight for CPC / NCE loss
beta     = 0.1           # Weight for LLD (MI lower bound) loss
ROUND    = 10            # Number of folds
num_epoch = 20           # Epochs per fold
train_batch = 5          # Batch size
```

Then run:

```bash
python MI-AIGIQA_train.py
```

Checkpoints and logs are saved to:
```
weights_final/MI-AIGIQA_{dataset}_alpha{alpha}_beta{beta}_batch{train_batch}_mainlr{main_lr}_{ROUND}fold_{mos_type}/
```

Each run produces a `train_test.log` with per-fold SRCC / PLCC and the final 10-fold averages.

#### Training Objective

MI-AIGIQA uses a **two-stage training loop** at each fold:

| Stage | Optimized Modules | Loss |
|-------|-------------------|------|
| **Stage 0** (MI warm-up) | MMILB only | $-\text{LLD}$ |
| **Stage 1** (full network) | All parameters | $\mathcal{L}_{\text{MSE}} + \alpha \cdot \text{NCE} - \beta \cdot \text{LLD}$ |

where NCE is the CPC-based backward MI estimate, and LLD is the MMILB variational lower bound.

---

### TABLE 2 — Cross-Dataset Generalization

Edit the configuration block at the top of `CrossDataset-training1.py`:

```python
Train_dataset = 'AIGCIQA2023'   # Source dataset
Test_dataset  = 'PKU-AIGIQA'    # Target dataset
mos_type      = 'quality'
main_lr       = 1e-5
mmilb_lr      = 5e-6
num_epoch     = 10
alpha         = 0.1
beta          = 0.1
```

Then run:

```bash
python CrossDataset-training1.py
```

This trains on the **full** source dataset and evaluates on the **full** target dataset (no cross-validation). Checkpoints are saved to:
```
weights_final/MI-AIGIQA_BLIP_Cross_Train{Train_dataset}_Test{Test_dataset}_batch{train_batch}_main-lr{main_lr}_mmilb-lr{mmilb_lr}_{mos_type}/
```

---

## Evaluation

Training scripts report **SRCC** (Spearman Rank Correlation Coefficient) and **PLCC** (Pearson Linear Correlation Coefficient) on the validation split at the end of each fold. The final reported metric is the average across all 10 folds.

Official training logs reproducing Table 1 and Table 2 results from the paper are stored in `train_log_official/`:

```
train_log_official/
├── TABLE1/
│   ├── MI-AIGIQA_AGIQA-3k_quality_final/    train_test.log
│   ├── MI-AIGIQA_AGIQA-3k_consis_final/     train_test.log
│   ├── MI-AIGIQA_AIGCIQA2023_quality_final/ train_test.log
│   ├── MI-AIGIQA_AIGCIQA2023_consis_final/  train_test.log
│   ├── MI-AIGIQA_PKU-AIGIQA_quality_final/  train_test.log
│   └── MI-AIGIQA_PKU-AIGIQA_consis_final/   train_test.log
└── TABLE2/
    ├── MI-AIGIQA_BLIP_Cross_TrainAGIQA-3k_TestAIGCIQA2023_quality/
    ├── MI-AIGIQA_BLIP_Cross_TrainAGIQA-3k_TestPKU-AIGIQA_quality/
    └── ...  (all 12 cross-dataset combinations × 2 mos_types)
```

---

## Gaussian Visualization (Supplementary)

To reproduce the text-feature Gaussianity analysis (used to justify the MMILB assumption on AGIQA-3k prompts):

```bash
cd Gaussian-visualization
python visualize_text_gaussian.py \
    --csv ../Database/AGIQA-3k/1/train.csv \
    --prompt_col prompt \
    --out_dir ./output \
    --device cuda:0 \
    --pool cls
```

Outputs saved to `output/`:
- `blip_text_gaussian_check.pdf` — 2D PCA scatter + Q-Q plots
- `blip_text_gaussian_check.png`
- `summary.txt` — Filliben r-statistics and normality pass rate per dimension

---

## Efficiency Benchmarking

To measure the model's parameter count, FLOPs, and inference throughput:

```bash
python test_efficiency.py
```

This script reports:
- Total / trainable parameters for each sub-module
- Approximate fp32 model size (MB)
- GFLOPs for a single 224×224 image
- Average inference time and FPS on GPU

---

## Results

### Table 1 — In-Distribution Performance (10-Fold CV)

| Dataset | Metric | SRCC | PLCC |
|---------|--------|------|------|
| AGIQA-3k | Quality | — | — |
| AGIQA-3k | Consistency | — | — |
| AIGCIQA2023 | Quality | — | — |
| AIGCIQA2023 | Consistency | — | — |
| PKU-AIGIQA | Quality | — | — |
| PKU-AIGIQA | Consistency | — | — |

### Table 2 — Cross-Dataset Generalization

| Train → Test | Metric | SRCC | PLCC |
|-------------|--------|------|------|
| AGIQA-3k → AIGCIQA2023 | Quality | — | — |
| AGIQA-3k → PKU-AIGIQA | Quality | — | — |
| AIGCIQA2023 → AGIQA-3k | Quality | — | — |
| AIGCIQA2023 → PKU-AIGIQA | Quality | — | — |
| PKU-AIGIQA → AGIQA-3k | Quality | — | — |
| PKU-AIGIQA → AIGCIQA2023 | Quality | — | — |

> Full results including consistency and authenticity metrics are available in `train_log_official/`.

---

## Repository Structure

```
MI-AIGIQA_final/
├── model.py                    # MMIM architecture (CrossAttention, ImageTextRegression)
├── modules/
│   └── encoders.py             # MMILB (forward MI) and CPC (backward MI) estimators
├── ImageDataset.py             # PyTorch Dataset classes for all four benchmarks
├── utils_self.py               # Preprocessing, logging, DataLoader factory
│
├── MI-AIGIQA_train.py          # TABLE 1: single-dataset 10-fold training
├── CrossDataset-training1.py   # TABLE 2: cross-dataset train/test
│
├── test_efficiency.py          # Parameter count, FLOPs, and inference speed
│
├── Database/                   # 10-fold CSV splits for all datasets
├── Data/                       # Dataset images (not included — see Datasets section)
│
├── Gaussian-visualization/
│   └── visualize_text_gaussian.py  # Supplementary Gaussianity analysis
│
├── train_log_official/         # Official logs reproducing Table 1 & Table 2
│   ├── TABLE1/
│   └── TABLE2/
│
└── weights_final/              # Checkpoints saved during your training runs
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{miaigiqa2026,
  title     = {MI-AIGIQA: Mutual Information-Guided Multimodal AI-Generated Image Quality Assessment},
  author    = {YOUR_AUTHORS},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

---

## Acknowledgements

This work builds upon [ImageReward](https://github.com/THUDM/ImageReward) (BLIP backbone) and the mutual information estimation framework from [MMIM](https://github.com/declare-lab/MMIM). We thank the authors for their open-source contributions.
