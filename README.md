# Ocular Disease Recognition with ResNet-50

A deep learning project that classifies ocular (eye) diseases from fundus photographs using transfer learning with ResNet-50 and PyTorch — built as part of a university group project.

---

## Team

- Basmala Soliman
- Nouran Alaa
- Basmala Samy

---

## Project Overview

This project uses the ODIR (Ocular Disease Intelligent Recognition) dataset containing 6,392 patient records with fundus eye photographs. A pre-trained ResNet-50 model was fine-tuned using transfer learning to classify patients into 8 disease categories — simulating a real-world medical AI diagnostic system.

---

## Disease Categories

| Label | Disease |
|-------|---------|
| N | Normal |
| D | Diabetes |
| G | Glaucoma |
| C | Cataract |
| A | Age-related Macular Degeneration |
| H | Hypertension |
| M | Pathological Myopia |
| O | Other diseases/abnormalities |

---

## Project Pipeline

### Data Collection & Exploration
- Loaded ODIR dataset with 6,392 patient records
- Explored patient age distribution (range: 1:91, mean: 58 years)
- Analyzed disease label distribution across 8 categories
- Identified and handled 2,146 duplicate records

### Data Cleaning & Preprocessing
- Removed irrelevant columns (`filepath`, `filename`)
- Handled duplicate records
- Applied image transformations (resize to 512x512, normalization)
- Split dataset into training and validation sets

### Model Architecture
- Used **ResNet-50** pre-trained on ImageNet (Transfer Learning)
- Replaced the final fully connected layer with a custom head:
  - Linear(2048 : 2048) + ReLU + Dropout(0.4)
  - Linear(2048 : 2048) + ReLU + Dropout(0.4)
  - Linear(2048 : 8) + LogSigmoid
- Distributed model across GPUs using `nn.DataParallel`

### Training
- **Optimizer:** SGD with momentum=0.9, lr=3e-4
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 32
- **Mixed Precision Training:** GradScaler for memory efficiency
- **Checkpointing:** Saved best model based on validation loss

### Results
| Epoch | Train Accuracy | Val Accuracy | Train Loss | Val Loss |
|-------|---------------|--------------|------------|----------|
| 1 | 42.15% | 46.31% | 1.9396 | 1.7959 |
| 2 | 45.67% | 46.31% | 1.6771 | 1.5786 |
| 3 | 45.67% | 46.31% | 1.5431 | 1.5159 |

Training Duration: ~21 minutes on GPU

---

## Dataset

**Source:** [ODIR Dataset - Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k?resource=download)

Dataset not included in this repository due to size (2GB). Please download it from the link above and place it in the `data/` folder.

| Property | Value |
|----------|-------|
| Patients | 5,000 |
| Records | 6,392 |
| Image Size | 512x512 |
| Classes | 8 |
| Domain | Medical Ophthalmology |

---

## Key Learnings

- Implementing transfer learning with a pre-trained ResNet-50
- Fine-tuning deep learning models for medical image classification
- Handling large-scale image datasets with PyTorch DataLoaders
- Using mixed precision training (GradScaler) for GPU memory efficiency
- Multi-label classification for medical diagnostics
- Collaborative team-based machine learning project development

---
