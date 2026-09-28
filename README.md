# PET-Prostate-Cancer-AI
Deep learning project for PET image harmonization and prostate cancer classification across multiple imaging domains.
# PET Image Harmonization for Prostate Cancer Classification Using Deep Learning

## Overview

Deep learning models for medical imaging can experience substantial
performance degradation when applied to images acquired from different
centres or imaging systems. This domain shift can limit the
generalizability of models across clinical environments.

This project investigates the effectiveness of multi-source PET image
harmonization for deep-learning-based prostate cancer classification.

Two image harmonization strategies were experimentally evaluated to
investigate whether reducing differences between PET images from
different sources could improve model generalization.

The project was conducted at the **Turku PET Centre, Finland**.

---

## Research Objective

The main objective was to investigate the effectiveness of multi-source
PET image harmonization on the performance and generalizability of deep
learning models for prostate cancer classification.

Specifically, the project aimed to:

- Preprocess 3D PET images and extract relevant 2D prostate slices.
- Investigate two PET image harmonization strategies.
- Develop deep learning models for prostate cancer classification.
- Evaluate model performance on unseen data.
- Compare the two harmonization approaches in terms of model
  generalizability.

---

## Harmonization Strategies

Two harmonization approaches were investigated.

### H1 — Constant Resolution

PET images were harmonized while maintaining the image resolution and
allowing the scanned area to change.

### H2 — Constant Scanned Area

PET images were harmonized while maintaining the scanned area while
standardizing the image resolution.

The objective was to determine which approach produced better
generalization when the trained models were evaluated on images from a
different source.

---

## Medical Image Preprocessing

The preprocessing pipeline included:

- Processing 3D PET images.
- Anatomical segmentation using **TotalSegmentator**.
- Extraction of the prostate region using segmentation masks.
- Conversion of 3D PET volumes into 2D prostate slices.
- Image harmonization using the H1 and H2 strategies.
- Intensity normalization using min-max scaling.
- Patient-level train/test splitting to reduce data leakage.

---

## Deep Learning Models

Two pretrained convolutional neural network architectures were evaluated:

### ResNet50

Transfer learning was applied using a pretrained ResNet50 model.

The final layers were adapted for binary prostate cancer classification,
with dense and dropout layers added to reduce overfitting.

### DenseNet121

A pretrained DenseNet121 architecture was also evaluated using a similar
transfer-learning strategy.

The classification task distinguished between:

- Cancerous PET images
- Non-cancerous PET images

---

## Training Strategy

The models were developed using **TensorFlow/Keras**.

Several techniques were used during training:

- Transfer learning
- Patient-level dataset splitting
- Class weighting for class imbalance
- Early stopping
- Learning-rate reduction
- Dropout regularization
- Binary cross-entropy loss
- Adam optimization

---

## External Validation

A key component of the project was evaluating whether the models could
generalize beyond the data used for model development.

Models trained using the primary PET dataset were therefore evaluated
using an independent external PET dataset.

This provided a way to investigate the effect of domain shift and the
potential benefits of PET image harmonization.

---

## Results

The experiments showed differences between the two harmonization
strategies.

For the **ResNet50** model:

| Dataset | Harmonization | Recall | Precision | Accuracy |
|---|---|---:|---:|---:|
| AUTOPET test | H1 | 61.7% | 68.5% | 80.0% |
| External test | H1 | 67.7% | 97.8% | 75.4% |
| AUTOPET test | H2 | 55.3% | 66.8% | 78.4% |
| External test | H2 | 65.9% | 97.8% | 74.1% |

For the **DenseNet121** model:

| Dataset | Harmonization | Recall | Precision | Accuracy |
|---|---|---:|---:|---:|
| AUTOPET test | H1 | 53.5% | 57.5% | 74.2% |
| External test | H1 | 71.5% | 95.9% | 77.0% |
| AUTOPET test | H2 | 57.6% | 57.6% | 74.3% |
| External test | H2 | 71.1% | 95.4% | 76.5% |

Overall, H1 produced better results than H2 for ResNet50, while the
DenseNet121 results for the two harmonization strategies were relatively
similar.

The results also demonstrate the challenge of developing generalized
deep learning models across PET imaging domains.

---

## Technologies

- Python
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Matplotlib
- NiBabel
- TotalSegmentator
- ResNet50
- DenseNet121
- Medical image processing
- Transfer learning

---

## Repository Structure

    PET-Prostate-Cancer-AI/
    │
    ├── README.md
    ├── requirements.txt
    │
    ├── preprocessing/
    │   └── harmonization.py
    │
    ├── models/
    │   ├── resnet50.py
    │   └── densenet121.py
    │
    ├── train.py
    ├── evaluate.py
    │
    └── results/
        └── figures/

---

## Data Availability and Privacy

Medical imaging data used during this research are **not distributed
through this repository**.

Patient data, private datasets, institutional data, and other restricted
research materials are excluded to protect patient privacy and comply
with applicable data-use and institutional requirements.

This repository is intended to contain only code, documentation, and
results that are permitted for public release.

---

## Research Context

This work was carried out during a research internship at:

**Turku PET Centre, Finland**

as part of the:

**Erasmus Mundus Joint Master in Intelligent Photonics for Security,
Reliability, Sustainability and Safety (iPSRS)**

### Researcher

**James Adah Ameh**

Research interests:

- Artificial Intelligence
- Medical Imaging
- Deep Learning
- Computer Vision
- Image Processing
- Photonics
