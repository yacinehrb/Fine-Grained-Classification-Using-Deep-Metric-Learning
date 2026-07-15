# Fine-Grained Classification Using Deep Metric Learning

## Overview

This project implements a **Deep Metric Learning** pipeline for fine-grained classification of mechanical component anomalies. Instead of directly predicting class labels, the model learns a discriminative embedding space where visually similar samples are located close together while different classes are pushed farther apart.

This approach is particularly effective for industrial inspection tasks where defects exhibit **high intra-class variation** and **low inter-class variation**, making conventional classification models less reliable.

---

## Key Features

- 🔹 Deep Metric Learning framework using **Triplet Loss**
- 🔹 Learns robust feature embeddings instead of class probabilities
- 🔹 Improved separation between visually similar defect categories
- 🔹 Suitable for fine-grained industrial inspection and anomaly classification
- 🔹 Embedding visualization before and after training

---

## Methodology

The network learns an embedding function

\[
f(x): x \rightarrow \mathbb{R}^d
\]

that maps each image into a feature space where:

- Images from the **same class** are close together.
- Images from **different classes** are far apart.

Training is performed using **Triplet Loss**.

### Triplet Loss

For every training triplet:

- **Anchor (A):** reference image
- **Positive (P):** image from the same class
- **Negative (N):** image from a different class

the objective is

\[
L(A,P,N)=\max\left(0,\ ||f(A)-f(P)||^2-||f(A)-f(N)||^2+\alpha\right)
\]

where:

- **f(·)** is the embedding network
- **α** is the margin enforcing class separation

The model minimizes the distance between the anchor and positive sample while maximizing the distance between the anchor and negative sample.

---

## Embedding Space Visualization

### Before Training

The embeddings are randomly distributed, resulting in poor class separability.

![Embedding Before Training](Before_training.PNG)

---

### After Training

After optimization with Triplet Loss, samples belonging to the same category naturally cluster together while different categories become clearly separated.

![Embedding After Training](After_training.PNG)


---

## Applications

This project can be applied to:

- Industrial quality inspection
- Mechanical defect classification
- Surface anomaly detection
- Product similarity search
- Image retrieval systems

---

## Results

Deep Metric Learning significantly improves the discriminative power of the learned feature space by producing compact intra-class clusters and larger inter-class margins. This makes the approach particularly suitable for fine-grained visual recognition problems where conventional softmax classifiers struggle.
