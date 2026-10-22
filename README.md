# 😷 Face Mask Detection System

A Computer Vision binary image classification system designed to detect face mask compliance for public health monitoring and campus access control. Powered by a **Convolutional Neural Network (CNN)** with **MobileNetV2 Transfer Learning** and served via an interactive **Streamlit** dashboard.

---

## 📌 Project Overview
* **Topic:** 51. Face Mask Detection (Section-401 A)
* **Problem Type:** Supervised Binary Image Classification ($y \in \{0, 1\}$)
* **Dataset:** Kaggle – Face Mask Detection Dataset (Cropped facial regions)
* **Objective:** Detect whether an individual is wearing a protective face mask or not.
* **Input Features:** Facial image pixels reshaped to $224 \times 224 \times 3$ (RGB)
* **ML Techniques:** Deep CNN, Transfer Learning (MobileNetV2 backbone)
* **Evaluation Metrics:** Accuracy, Precision, Recall
* **Outcome:** Automated compliance verification for public health safety enforcement.

---

## ⚙️ Deep Learning Architecture

### 1. Transfer Learning Backbone (MobileNetV2)
* **Pre-trained Weights:** Initialized on ImageNet ($1.4\text{M}+$ images).
* **Frozen Base:** Pre-trained feature extraction layers are set to `trainable = False` to retain low-level edge, gradient, and facial contour detectors without overfitting.
* **Depthwise Separable Convolutions:** Lightweight and fast inference on CPU environments without dedicated GPUs.

### 2. Custom Classification Head
1. **Input Normalization:** Preprocessed using MobileNetV2's pipeline, scaling raw pixel values $[0, 255]$ into the range $[-1.0, 1.0]$.
2. **GlobalAveragePooling2D:** Collapses spatial feature maps into a 1D vector (1280 feature values).
3. **Dropout ($p = 0.2$):** Randomly deactivates 20% of neurons during training to prevent overfitting.
4. **Dense Layer (Sigmoid):** Outputs continuous probability score:

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:
* $\hat{y} < 0.5 \implies$ **With Mask ($0$)**
* $\hat{y} \ge 0.5 \implies$ **Without Mask ($1$)**

### 3. Optimization & Loss
* **Loss Function:** Binary Cross-Entropy Loss:

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

* **Optimizer:** Adam ($\text{learning\_rate} = 10^{-3}$)

---

## 📊 Evaluation Metrics

Model performance is evaluated on unseen test data across three standard metrics:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

---

## 📂 Repository Structure

```text
├── dataset/
│   ├── with_mask/              # Facial crops wearing masks
│   └── without_mask/           # Facial crops without masks
├── train.py                    # MobileNetV2 training pipeline & metrics
├── app.py                      # Interactive Streamlit testing dashboard
├── mask_detector_model.keras   # Saved trained model weights
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes venv and cache
└── README.md                   # Technical system documentation