# 🐦 BirdCLEF 2025 - EfficientNetB0 Audio Classification

This repository contains a complete pipeline for **bird species identification** from sound recordings, based on **EfficientNetB0** and advanced audio preprocessing. It includes training, validation, inference, and postprocessing techniques specifically tailored for the **BirdCLEF 2025 competition**.

---

## 📚 Table of Contents

- [🧠 Overview](#-overview)
- [🔊 Audio Preprocessing](#-audio-preprocessing)
- [🎨 Spectrogram Augmentation](#-spectrogram-augmentation)
- [🧮 Model Architecture](#-model-architecture)
- [⚙️ Loss and Optimization](#-⚙️-Loss-and-Optimization)
- [🧪 Postprocessing](#-postprocessing)
- [📁 Files](#-files)

---

## 🧠 Overview

The model is trained to detect bird calls using **precomputed mel spectrograms** generated from 5-second audio clips sampled at 32 kHz. The entire training pipeline supports:

- EfficientNetB0 with pretrained weights via Timm
- Stratified 5-fold cross-validation
- Fast audio-to-melspec conversion via NumPy or `librosa`
- Reproducible training with seeded RNG and CuDNN configuration

---

## 🔊 Audio Preprocessing

Audio clips are converted into mel spectrograms using:

- Sampling rate: `32,000 Hz`
- FFT size: `1024`
- Hop length: `512`
- Mel bins: `128`
- Frequency range: `50 Hz – 14,000 Hz`
- Final shape: `(256 × 256)`

Precomputed spectrograms are stored as `.npy` files for fast training.

---

## 🎨 Spectrogram Augmentation

During training, the spectrograms are randomly augmented to improve generalization:

- **Time masking**: Randomly masks horizontal bands (time axis)
- **Frequency masking**: Randomly masks vertical bands (frequency axis)
- **Brightness jitter**: Slight shifts in amplitude (magnitude scaling)
- **Mixup**: Blends spectrograms and labels using `mixup_alpha = 0.5`

Augmentations are applied with a configurable probability (`aug_prob = 0.5`).

---

## 🧮 Model Architecture

- Backbone: `efficientnet_b0` from [Timm](https://rwightman.github.io/pytorch-image-models/)
- Input channels: `1` (grayscale spectrogram)
- Output: 264 bird classes with multilabel activation
- Training hardware: GPU (`cuda` if available)

Model outputs are fed through a `Sigmoid` activation for multilabel classification.

---

## ⚙️ Loss and Optimization

- **Loss**: `BCEWithLogitsLoss` (binary cross-entropy for multilabel tasks)
- **Optimizer**: `AdamW` with `weight_decay=1e-5`
- **Scheduler**: `CosineAnnealingLR`
  - `lr = 5e-4`, `min_lr = 1e-6`, `T_max = epochs`
- **Batch size**: 32, **Epochs**: 10 (adjustable)

---

## 🧪 Postprocessing

At inference time, the following steps are performed to convert raw model outputs into final predictions:

1. **Sigmoid Activation**  
   The model outputs a raw score (logit) per class. These are passed through a sigmoid function to obtain **probabilities** in the range `[0, 1]`.

2. **Thresholding**  
   Each class probability is compared against a fixed threshold (e.g., `0.5`).  
   Classes with probabilities above this threshold are considered **active (present)** in the current audio segment.

3. **Chunk-Level Aggregation**  
   Soundscape recordings are divided into smaller chunks (e.g., 5s or 10s).  
   For each chunk, predictions are made independently.  
   Then, **predicted classes across all chunks for the same recording** are **aggregated** (e.g., via max or union) to produce one final set of labels for that recording.

4. **Formatting for Submission**  
   Final predictions are converted into the **BirdCLEF 2025 submission format**, where each line specifies:
   - `row_id`
   - `class_name` (bird ID)
   - `presence` (binary label: 0 or 1)

---

## 📁 Files

```
birdclef-2025/
├── efficientnet_b0_training.ipynb       # Full training pipeline
├── efficientnet_b0_inference.ipynb      # Prediction and postprocessing
├── data_preprocessing.ipynb             # Audio-to-melspec conversion
└── README.md                            # This documentation
```
