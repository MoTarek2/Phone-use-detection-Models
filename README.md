# 🚗 Distracted Driver & Phone Usage Detection

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Project Overview
This project is a computer vision pipeline developed for an artificial intelligence course, designed to detect distracted driving behaviors—specifically phone usage behind the wheel. The system uses deep learning and OpenCV to classify driver behavior from a live feed or static images into three core categories:
1. **Safe Driving**
2. **Texting** (Right/Left hand)
3. **Calling** (Right/Left hand)

This repository contains two different modeling approaches—**EfficientNetB2** and **MobileNetV2**—allowing for a comparison between high-accuracy feature extraction and lightweight, edge-optimized performance. This project also serves as a technical demonstration for professional AI and computer vision development.

---

## 📊 Dataset & Preprocessing

The models are trained using the [State Farm Distracted Driver Detection dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection). 

* **Class Mapping:** The original dataset contains 10 highly specific classes. To create a more robust and generalized phone-detection model, these 10 classes were mapped down to the 3 core behaviors listed above.
* **Data Splitting (Crucial Step):** In the EfficientNet model, instead of a random train/test split, the data is split by **Driver ID (`subject`)**. This is critical. If the same driver appeared in both the training and validation sets, the model would memorize the driver's clothing or face rather than learning the actual distracted behavior, leading to massive data leakage.

---

## 🧠 Model Architectures

### 1. EfficientNetB2 (`EfficientNetB2_FINAL.ipynb`)
* **Architecture:** Uses EfficientNetB2 pre-trained on ImageNet.
* **Strengths:** High accuracy and excellent feature extraction. The compound scaling of EfficientNet makes it highly efficient for its parameter count (~9 million).
* **Implementation:** Includes an inference block that scans a target folder (`moaaz-testing`), predicts the class, and overlays confidence percentages directly onto the output images.

### 2. MobileNetV2 (`phone-use-detection-model-Final.ipynb`)
* **Architecture:** Uses MobileNetV2, freezing early layers and fine-tuning the top 30 layers.
* **Strengths:** Extremely lightweight (~2.2 million parameters) and optimized for edge devices or real-time OpenCV video feed processing.
* **Implementation:** Employs `ThreadPoolExecutor` for multithreaded image loading and applies data augmentation (Rotation, Zoom, Translation) to improve model generalization.

---

## ⏱️ Time and Space Complexity

### Time Complexity
* **Training:** Bound by O(E × B × F), where E is epochs, B is the number of batches, and F is the forward/backward pass time of the base architecture. MobileNetV2 processes epochs significantly faster than EfficientNetB2.
* **Inference:** O(1) per image. MobileNetV2 is optimized for fast, real-time inference, whereas EfficientNetB2 carries higher computational latency per frame.

### Space Complexity & Known Bottlenecks
* **EfficientNetB2:** Space complexity is primarily dictated by the model weights (~9M parameters) and standard batch-loading memory overhead.
* **MobileNetV2 (Memory Bottleneck):** The current implementation in the notebook loads the *entire* training and validation dataset into RAM simultaneously as NumPy arrays using `np.concatenate`. Even using `float16` at 192x192 resolution, this O(N × W × H × C) approach is highly memory-inefficient and will cause severe Out-Of-Memory (OOM) errors on machines with limited RAM. 
  * *Future Optimization:* Rewrite the data pipeline to use `tf.data.Dataset` or `ImageDataGenerator` for disk-to-batch loading, which will reduce memory complexity to O(Batch_Size).

---

## ⚠️ Limitations, Failure Cases & Bias

Deep learning models for image classification are strictly bound by their training distributions. In a real-world deployment, this system has specific vulnerabilities:

1. **Occlusion:** If the driver's hand or the phone is blocked by the steering wheel or a passenger, the model will struggle to differentiate between holding a phone and resting an empty hand.
2. **Lighting Conditions:** The models will fail entirely during night driving, under harsh glare, or in tunnels, as the dataset only contains well-lit daytime images.
3. **Class Confusion:** A driver scratching their ear, resting their head on their hand, or looking down at their lap will likely trigger false positives for "Calling" or "Texting" due to the similar spatial geometry.
4. **Camera Angles (Overfitting):** The models are fit to the specific right-dashboard camera angle of the training set. Moving the camera to the rearview mirror or the left pillar will drastically drop accuracy to near-random guessing.
5. **Demographic Bias:** The dataset features a very limited pool of 26 unique drivers. If this pool lacks diversity, the model's feature extraction (like gaze direction or skin contrast) will perform poorly on underrepresented demographics.
6. **Vehicle/Layout Bias:** The dataset consists exclusively of Left-Hand Drive (LHD) vehicles. The model inherently expects the driver on the left side of the frame. Deploying this in a Right-Hand Drive (RHD) vehicle will cause catastrophic failure unless the input feeds are flipped horizontally during preprocessing.
