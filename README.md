# Traffic Sign Classification using LeNet-5 Architecture (Keras)

A deep learning project that trains a Convolutional Neural Network (CNN) based on the classic **LeNet-5 architecture** to classify traffic signs across **43 categories** from the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

---

## Overview

Road safety systems and autonomous vehicles depend on the ability to accurately recognize traffic signs in real time. This project explores how a relatively compact CNN architecture — LeNet-5 — can be applied to this multi-class image classification problem, achieving strong performance with minimal computational overhead.

The model was built and trained entirely from scratch using Keras with TensorFlow backend, without any pre-trained weights or transfer learning.

---

## Model Architecture

The network follows the classic LeNet-5 structure adapted for 32×32 grayscale input images:

| Layer | Type | Output Shape |
|---|---|---|
| Input | — | 32 × 32 × 1 |
| Conv2D (6 filters, 5×5, ReLU) | Convolutional | 28 × 28 × 6 |
| AveragePooling2D (2×2) | Pooling | 14 × 14 × 6 |
| Conv2D (16 filters, 5×5, ReLU) | Convolutional | 10 × 10 × 16 |
| AveragePooling2D (2×2) | Pooling | 5 × 5 × 16 |
| Flatten | — | 400 |
| Dense (120, ReLU) | Fully Connected | 120 |
| Dense (84, ReLU) | Fully Connected | 84 |
| Dense (43, Softmax) | Output | 43 |

**Training configuration:**
- Optimizer: Adam (learning rate = 0.001)
- Loss: Sparse Categorical Crossentropy
- Epochs: 50 | Batch Size: 500

---

## Results

| Metric | Score |
|---|---|
| Training Accuracy | ~99.67% |
| Validation Accuracy | ~87.46% |
| **Test Accuracy** | **87.02%** |

The training curve shows strong convergence from epoch 1, with validation accuracy stabilizing around the 87% range — reasonable performance given the model's simplicity and no augmentation.

---

## Dataset

The dataset is sourced from the **German Traffic Sign Recognition Benchmark (GTSRB)** and contains pickle-serialized image arrays split into training, validation, and test sets.

**Dataset is hosted on Google Drive** (too large for GitHub):
👉 [Download Dataset](https://drive.google.com/drive/folders/1NXVIcBdvkeMry2B8o6bLqWmrSTS-rLka?usp=sharing)

After downloading, place the files in the following structure:
```
project/
├── traffic-signs-data/
│   ├── train.p
│   ├── valid.p
│   └── test.p
└── image_classification_lenet_keras_traffic_signs.ipynb
```

Alternatively, run the first cell in the notebook to auto-download via `gdown`.

**43 Classes include:**
Speed limits (20–120 km/h), No passing zones, Right-of-way, Yield, Stop, Pedestrians, Children crossing, Roundabout, and more.

---

## Project Structure

```
├── image_classification_lenet_keras_traffic_signs.ipynb   # Main notebook
├── README.md
└── traffic-signs-data/          # Dataset directory (not tracked in git)
    ├── train.p
    ├── valid.p
    └── test.p
```

---

## Steps in the Notebook

1. **Problem Statement** — Overview of the 43-class traffic sign classification task
2. **Import Libraries & Dataset** — Load pickle files, explore data shapes
3. **Image Exploration** — Visualize sample training images
4. **Data Preparation** — Grayscale conversion, normalization, shuffling
5. **Model Training** — Build and train the LeNet CNN
6. **Model Evaluation** — Accuracy metrics, training/validation curves
7. **Predictions** — Confusion matrix, classification report, visual prediction grid

---

## Requirements

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn gdown
```

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Model building and training |
| NumPy | Array operations |
| Pandas | Data handling |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | Metrics and evaluation |
| Pickle | Dataset loading |
| gdown | Google Drive dataset download |

**Python version:** 3.10+ recommended
