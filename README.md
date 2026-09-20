# The Pareidolia Paradox

### Physics-Informed Deep Learning for Lunar Surface Topography Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-EE4C2C)
![Metric](https://img.shields.io/badge/Metric-Balanced%20Accuracy-success)
![License](https://img.shields.io/badge/License-MIT-green)

A physics-informed computer vision system for classifying **256×256 lunar surface images** into two topographic categories:

- **Class 0 — Depth:** Craters, depressions, holes, and concave structures
- **Class 1 — Rise:** Mounds, hills, rocks, boulders, and convex structures

This project was developed for *The Pareidolia Paradox*, a lunar surface image classification challenge where the direction of illumination is provided through the `sun_azimuth_angle` metadata.

---

## Table of Contents

- [Overview](#overview)
- [Challenge](#challenge)
  - [Objective](#objective)
  - [Evaluation Metric](#evaluation-metric)
  - [Dataset](#dataset)
  - [Key Challenges](#key-challenges)
- [Physics-Informed Approach](#physics-informed-approach)
  - [Sun-Azimuth Normalization](#sun-azimuth-normalization)
  - [Physics-Informed Gradient](#physics-informed-gradient)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
  - [Data Augmentation](#data-augmentation)
  - [Validation Strategy](#validation-strategy)
- [Experiments and Development](#experiments-and-development)
  - [Validation Results](#validation-results)
  - [Threshold Optimization](#threshold-optimization)
  - [Test-Time Augmentation](#test-time-augmentation)
- [Final Pipeline](#final-pipeline)
  - [Final Training](#final-training)
  - [Final Inference](#final-inference)
  - [Submission Verification](#submission-verification)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Dataset Setup](#dataset-setup)
  - [Training & Inference](#training--inference)
  - [Model Weights](#model-weights)
- [Reproducibility](#reproducibility)
- [Technical Details & Design Decisions](#technical-details--design-decisions)
- [Limitations](#limitations)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

*The Pareidolia Paradox* is a lunar surface image classification problem where the objective is to distinguish between **Depths** and **Rises** from grayscale lunar surface imagery.

The central difficulty is that lunar terrain appearance depends strongly on the direction of illumination. A crater and a mound can produce visually similar intensity patterns when the illumination direction changes. Conversely, the same physical structure can look substantially different under different solar azimuths.

The dataset provides a `sun_azimuth_angle` for every image. Instead of treating this value as an ordinary numerical feature, this project **incorporates it directly into the spatial image representation**.

```text
                 Lunar Image
                      │
                      ▼
             Sun-Azimuth Metadata
                      │
                      ▼
          Rotate by -sun_azimuth_angle
                      │
                      ▼
            Sun-Normalized Image
                 │          │
                 ▼          ▼
          Image Channel   Sobel Gradient
                 │          │
                 └────┬─────┘
                      ▼
              2-Channel ResNet18
                      │
                      ▼
             Binary Classification
                      │
                      ▼
             Threshold = 0.37
                      │
                      ▼
                Depth / Rise
```

---

## Challenge

### Objective

Classify each 256×256 lunar surface image into one of two categories:

| Label | Class | Description |
|:-----:|:------|:------------|
| 0 | **Depth** | Depressions, craters, holes, and concave structures |
| 1 | **Rise** | Mounds, hills, rocks, boulders, and convex structures |

### Evaluation Metric

The final evaluation metric is **Balanced Accuracy**, which gives equal importance to both classes by averaging their individual recall values:

$$
\text{Balanced Accuracy} = \frac{\text{Recall}_{\text{Depth}} + \text{Recall}_{\text{Rise}}}{2}
$$

### Dataset

#### Training Dataset

- **7,854** labeled images
- Resolution: **256×256** grayscale imagery
- Metadata: `image_id`, `sun_azimuth_angle`, `label`

#### Class Distribution

| Class | Meaning | Count | Percentage |
|:-----:|:--------|------:|-----------:|
| 0 | Depth | 2,854 | 36.34% |
| 1 | Rise | 5,000 | 63.66% |

The training dataset exhibits moderate class imbalance favoring the positive (Rise) class.

#### Evaluation Dataset

- **2,000** evaluation images
- Resolution: 256×256
- Metadata: `image_id`, `sun_azimuth_angle`
- Evaluation labels are hidden during development.

### Key Challenges

1. **Illumination Dependence** — The visual appearance of a lunar formation changes radically with the solar illumination angle. Raw image intensity alone contains misleading orientation-dependent features.
2. **Topographic Ambiguity** — Disambiguating concave from convex structures using a single grayscale image requires precise recovery of directional intensity gradient relationships.
3. **Solar-Azimuth Distribution Shift** — Exploratory analysis revealed shifts in the solar-azimuth distribution between training and evaluation splits, as well as class label correlations with illumination angle. Unhandled, models risk learning spurious illumination shortcuts rather than physical topography.
4. **Class Imbalance** — Optimizing standard accuracy risks biasing predictions toward the majority class (Rise). Loss functions must account for class weighting to align with Balanced Accuracy.
5. **Limited Computational Resources** — Models were trained on consumer hardware. Architectures were constrained to efficient variants (ResNet18, mixed precision, lightweight preprocessing), maintaining fast iterability without performance loss.

---

## Physics-Informed Approach

Instead of feeding raw numerical metadata into a secondary tabular branch, the pipeline maps solar azimuth directly into **spatial coordinate transformations**.

The final image representation consists of two channels:

| Channel | Content |
|:-------:|:--------|
| 0 | Sun-normalized grayscale image |
| 1 | Signed directional gradient along the illumination axis |

### Sun-Azimuth Normalization

Each image is rotated counter-clockwise by:

```text
rotation_angle = -sun_azimuth_angle
```

using OpenCV, retaining the original 256×256 dimensions. This aligns the primary solar illumination vector uniformly across the entire dataset prior to feature extraction.

### Physics-Informed Gradient

Following rotation, a **signed directional Sobel derivative** is computed along the x-axis:

$$
\text{gradient} = \frac{\partial I}{\partial x}
$$

where $I$ is the normalized image intensity. Preserving the gradient **sign** (rather than taking absolute magnitude) provides explicit signals regarding light-to-dark vs. dark-to-light transitions along crater walls and mound boundaries.

- **Calculated Scale Factor:** `1.039898`
- The scaled gradient is clipped to a stable numerical range before input stacking.

---

## Model Architecture

The core classifier uses a **ResNet18** backbone (pretrained via `torchvision`).

```text
Input [2 x 256 x 256]
  │
  ├── Channel 0: Sun-Normalized Image
  └── Channel 1: Signed Directional Gradient
          │
          ▼
   Modified ResNet18 (First Conv Adapted)
          │
          ▼
    Fully Connected Layer
          │
          ▼
      Binary Logit ──► Sigmoid ──► Threshold (0.37) ──► Class Output
```

- **First Convolution Layer:** Adapted from 3-channel (RGB) to 2-channel input by averaging pretrained RGB filter weights across the input channels.
- **Output:** Single logit converted via Sigmoid activation.
- **Total Parameters:** 11,173,889

---

## Training Strategy

| Setting | Value |
|:--------|:------|
| Framework | PyTorch & torchvision |
| Optimizer | AdamW (LR = 1×10⁻⁴, Weight Decay = 1×10⁻⁴) |
| Batch Size | 32 |
| Precision | Automatic Mixed Precision (AMP) |
| Loss Function | `BCEWithLogitsLoss` with class-imbalance positive weight |

The positive weight (`pos_weight` in PyTorch) compensates for the Rise-heavy class distribution:

$$
w_{\text{pos}} = \frac{2854}{5000} = 0.570800
$$

### Data Augmentation

Conservative spatial and intensity transformations were selected to avoid disturbing the physical illumination frame established by azimuth normalization.

| | Transformations |
|:--|:--|
| ✅ **Allowed** | Mild scale (90%–110%), small translations, subtle brightness/contrast tweaks, lightweight Gaussian noise |
| ❌ **Avoided** | Flips (horizontal/vertical), arbitrary rotations, strong perspective shifts |

### Validation Strategy

Validation was conducted using a **stratified 80/20 split**:

- **Training:** 6,283 images
- **Validation:** 1,571 images

Stratification was enforced across both the **class label** and **30° solar-azimuth bins** to ensure equal representation of illumination geometries and prevent shortcut leakage.

---

## Experiments and Development

The model evolved across distinct iterations:

```text
Phase 1: Raw Image + Azimuth Numerical Branch
   │
Phase 2: Sun Normalization + Gradient Verification
   │
Phase 3: Physics-Informed 2-Channel Network (ResNet18)
   │
Phase 3E / 3K: Azimuth-Aware Stratified Validation
   │
Phase 4 Physics: Class-Weighted Loss + Scaled Gradient + Conservative Augmentation
   │
Phase 5: Decision Threshold Sweep & TTA Evaluation
   │
Final Physics Pipeline
```

### Validation Results

Evaluating the physics-informed model on the controlled validation split:

| Metric | Default Threshold (0.50) | Optimized Threshold (0.37) |
|:-------|:------------------------:|:--------------------------:|
| **Balanced Accuracy** | 0.7146 | **0.7262** |
| Depth Recall | — | 75.61% |
| Rise Recall | — | 69.63% |

> **Note:** The validation score of 0.7262 reflects internal validation performance and does **not** represent official competition leaderboard results.

### Threshold Optimization

A decision boundary sweep was performed on validation probability outputs to maximize Balanced Accuracy:

| Threshold | Balanced Accuracy |
|:---------:|:-----------------:|
| 0.36 | 0.7213 |
| **0.37** | **0.7262** |
| 0.38 | 0.7245 |
| 0.39 | 0.7224 |
| 0.40 | 0.7255 |
| 0.41 | 0.7260 |
| 0.42 | 0.7234 |
| 0.43 | 0.7254 |
| 0.44 | 0.7240 |
| 0.45 | 0.7203 |

**Selected Threshold: `0.37`**

### Test-Time Augmentation

Test-Time Augmentation (TTA) using 5-point spatial shifts (with re-computed gradients per shift) was evaluated:

| Configuration | Balanced Accuracy |
|:--------------|:-----------------:|
| Without TTA | **0.7262** |
| With TTA | 0.7212 |

Because TTA reduced performance on validation data, it was **disabled** for final inference.

---

## Final Pipeline

### Final Training

The production model was trained on the **complete dataset of 7,854 labeled images** using the selected optimal hyperparameters:

```yaml
Architecture: Physics-Informed ResNet18
Input Channels: 2
Training Images: 7,854
Epochs: 2
Batch Size: 32
Learning Rate: 0.0001
Weight Decay: 0.0001
Pos Weight: 0.570800
Gradient Scale: 1.039898
Inference Threshold: 0.37
TTA: Disabled
Output Checkpoint: outputs/final_physics_model.pth
```

### Final Inference

For each evaluation sample:

1. Load the grayscale image (256×256).
2. Retrieve the corresponding `sun_azimuth_angle` from `Test/test_metadata.csv`.
3. Rotate the image counter-clockwise by `-sun_azimuth_angle`.
4. Rescale image intensities to `[0, 1]`.
5. Compute the signed Sobel x-gradient and divide by `1.039898`.
6. Stack into a 2-channel tensor `[2, 256, 256]`.
7. Compute the class probability via a ResNet18 forward pass.
8. Classify as **1 (Rise)** if `p ≥ 0.37`, else **0 (Depth)**.

### Submission Verification

Output file generated at `outputs/submission.csv`.

- ✅ Total rows: **2,000**
- ✅ Columns: `image_id`, `label`
- ✅ Missing / duplicate entries: **0**
- ✅ Valid output classes: only `0` or `1`

---

## Repository Structure

```text
Pareidolia-Paradox/
├── README.md
├── requirements.txt
├── train.py
├── inference.py
│
├── src/
│   ├── final_train.py
│   ├── final_inference.py
│   ├── phase4_physics_dataset.py
│   ├── phase4_physics_model.py
│   └── sun_utils.py
│
└── outputs/
    └── submission.csv
```

---

## Getting Started

### Installation

## 1. Clone the Repository

```bash
git clone https://github.com/AryanDhingan/Pareidolia-Paradox.git
cd Pareidolia-Paradox
```

## 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

## 3. Install PyTorch

The final model was developed and tested with:

- Python 3.10.11
- PyTorch 2.14.0 + CUDA 13.0
- Torchvision 0.29.0 + CUDA 13.0

For the CUDA 13.0 build:

```bash
pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

For CPU-only inference, install a compatible CPU build of PyTorch and torchvision instead.

## 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

# Inference

Before running inference, make sure the evaluation dataset is available at:

```text
Test/
├── eval_images/
└── test_metadata.csv
```

Download the final trained model:
[Download Final Model Weights](https://drive.google.com/file/d/1j9VAAi2YInKcv9QpqW9twX0fGG7bv8Em/view?usp=sharing)

Place the downloaded checkpoint at:

```text
outputs/final_physics_model.pth
```

Then run:

```bash
python inference.py
```

The script performs the complete preprocessing and inference pipeline and generates:

```text
outputs/submission.csv
```

The generated submission contains exactly the columns:

```text
image_id,label
```

for all 2,000 evaluation images.

### Dataset Setup

Organize dataset directories as follows:

```text
Train/
├── train_images/
└── train_metadata.csv

Test/
├── eval_images/
└── test_metadata.csv
```

### Training & Inference

**Train the model:**

```bash
python train.py
```

**Run inference:**

```bash
python inference.py
```

### Model Weights

Trained model checkpoints are hosted externally.

- **Download Link:** [Download Final Model Weights](https://drive.google.com/file/d/1j9VAAi2YInKcv9QpqW9twX0fGG7bv8Em/view?usp=sharing)
- **Destination Path:** `outputs/final_physics_model.pth`

---

## Reproducibility

Reproducibility is maintained by locking seeds and key parameters:

```python
SEED = 42
GRADIENT_SCALE = 1.039898
THRESHOLD = 0.37
```

---

## Technical Details & Design Decisions

| Question | Answer |
|:---------|:-------|
| **Why rotate by `sun_azimuth_angle`?** | Normalizes arbitrary directional lighting sources into a standardized frame across all inputs. |
| **Why signed gradients?** | Preserves positive vs. negative intensity transitions essential for identifying convex vs. concave geometry. |
| **Why 2-channel input?** | Combines raw spatial surface details with explicit illumination derivative features. |
| **Why Balanced Accuracy optimization?** | Directly aligns decision boundaries against dataset class imbalance. |

---

## Limitations

- **Validation vs. Leaderboard:** Validation metrics were measured locally and may deviate from test distribution evaluation.
- **Azimuth Shift:** Unseen test samples might exhibit unobserved lighting variations.
- **Empirical Thresholding:** The 0.37 operating point was fit specifically to validation predictions.
- **Single Model Execution:** No ensembling was utilized in the final submission pipeline.

---

## Contributors

- [Aryan Dhingan](https://github.com/AryanDhingan)
- [Bhavik Mungekar](https://github.com/BhaviK1247)

---

## Acknowledgments

Developed as part of *The Pareidolia Paradox* lunar surface classification competition. Special thanks to open-source computer vision tools and transfer learning frameworks.

---

## License

This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2026 Aryan Dhingan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```