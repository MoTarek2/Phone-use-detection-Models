# 🚗 Distracted Driver & Phone Usage Detection

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Project Overview
This project is a computer vision pipeline designed to detect distracted driving behaviors, specifically focusing on phone usage behind the wheel. Originally developed as an artificial intelligence course project, the system uses deep learning to classify driver behavior into three core categories:
1. **Safe Driving**
2. **Texting** (Right/Left hand)
3. **Calling** (Right/Left hand)

The repository contains two different modeling approaches—**EfficientNetB2** and **MobileNetV2**—allowing for a comparison between high-accuracy feature extraction and lightweight, edge-optimized performance (ideal for real-time OpenCV integrations).

---

## 📊 Dataset & Preprocessing

The models are trained using the [State Farm Distracted Driver Detection dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection). 

* **Class Mapping:** The original dataset contains 10 highly specific classes. To create a more robust and generalized phone-detection model, these 10 classes were mapped down to the 3 core behaviors listed above.
* **Data Splitting (Crucial Step):** Instead of a random train/test split, the data is split by **Driver ID (`subject`)**. This prevents data leakage. If the same driver appeared in both the training and validation sets, the model would memorize the driver's clothing or face rather than learning the actual distracted behavior.

---

## 🧠 Model Architectures

### 1. EfficientNetB2 (`EfficientNetB2_FINAL.ipynb`)
* **Architecture:** Uses EfficientNetB2 pre-trained on ImageNet.
* **Strengths:** High accuracy and excellent feature extraction. The compound scaling of EfficientNet makes it highly efficient for its parameter count (~9 million).
* **Implementation:** Includes an inference block that scans a target folder (`moaaz-testing`), predicts the class, and overlays confidence percentages directly onto the output images.

### 2. MobileNetV2 (`phone-use-detection-model-Final.ipynb`)
* **Architecture:** Uses MobileNetV2, freezing early layers and fine-tuning the top 30 layers.
* **Strengths:** Extremely lightweight (~2.2 million parameters) and optimized for edge devices or real-time video feed processing.
* **Implementation:** Employs `ThreadPoolExecutor` for multithreaded image loading and applies data augmentation (Rotation, Zoom, Translation) to improve model generalization.

---

## ⚙️ Setup & Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/yourusername/driver-phone-detection.git](https://github.com/yourusername/driver-phone-detection.git)
cd driver-phone-detection
pip install -r requirements.txt
