# Distracted Driver / Phone Use Detection

This repository contains two deep learning models designed to detect distracted driving behavior, specifically focusing on phone usage. The models are trained on the State Farm Distracted Driver Detection dataset, with the original 10 classes mapped down to 3 core behaviors: **Safe Driving, Texting, and Calling**.

## 📁 Repository Contents

1. **`EfficientNetB2_FINAL.ipynb`**: 
   * Uses **EfficientNetB2** as the base architecture.
   * Splits the training and validation data by *driver ID* (`subject`) rather than at random. This prevents data leakage by ensuring the model learns the behavior, not the specific person or their clothing.
   * Includes an inference block that scans a target folder (`moaaz-testing`) and outputs predictions with confidence percentages overlaid on the images.

2. **`phone-use-detection-model-Final.ipynb`**:
   * Uses **MobileNetV2** as the base architecture, freezing the early layers and fine-tuning the top 30 layers.
   * Implements data augmentation (Rotation, Zoom, Translation) to improve generalization.
   * Uses multithreading (`ThreadPoolExecutor`) for image loading, resizing images to 192x192.

---

## ⏱️ Time and Space Complexity

### Time Complexity
* **Training:** Training time is bound by $O(E \times B \times F)$, where $E$ is epochs, $B$ is the number of batches, and $F$ is the forward/backward pass time of the base architecture. MobileNetV2 is computationally cheaper and faster per epoch than EfficientNetB2, though EfficientNetB2 typically converges to higher accuracy in fewer steps.
* **Inference (Prediction):** $O(1)$ per image. MobileNetV2 is heavily optimized for fast real-time inference on edge devices (like dashcams), whereas EfficientNetB2 will have notably higher latency.

### Space Complexity (Memory & Storage)
* **MobileNetV2 Notebook:** The space complexity here is a severe bottleneck and poorly optimized. The code loads the entire training and validation datasets into RAM simultaneously as NumPy arrays via `np.concatenate`. Even using `float16` at 192x192 resolution, this $O(N \times W \times H \times C)$ approach is a massive RAM hog and will cause Out-Of-Memory (OOM) crashes on most standard local machines. To fix this, switch to `tf.data.Dataset` or `ImageDataGenerator` to load images in batches from the disk, which drops the RAM space complexity from $O(N)$ to $O(\text{Batch\_Size})$.
* **EfficientNetB2 Notebook:** The space complexity of the model parameters is roughly ~9 million parameters for EfficientNetB2, making it significantly heavier than MobileNetV2's ~2.2 million parameters.

---

## ⚠️ Limitations & Failure Cases

Deep learning models for image classification are not foolproof. In a real-world deployment, expect the following failure cases:

1. **Occlusion:** If the driver's hand is blocked by the steering wheel, their body, or a passenger, the model will struggle to determine if they are holding a phone or just resting their hand.
2. **Lighting and Shadows:** The training data has decent lighting. The models will likely fail during night driving, in tunnels, or under harsh glare without retraining on low-light/IR camera data.
3. **Class Confusion:** The model relies heavily on spatial features. A driver scratching their ear or resting their head on their hand might trigger a false positive for "Calling". A driver looking down at the radio or their lap might trigger a false positive for "Texting."
4. **Camera Angles:** These models are heavily over-fit to the specific camera angle used in the dataset (mounted on the right side of the dashboard). Changing the camera perspective (e.g., rearview mirror mount) will drastically drop accuracy.

---

## 🔍 Potential Sources of Bias

* **Demographic Bias:** If the 26 unique drivers in the dataset do not represent a diverse spread of skin tones, ages, and genders, the model may perform poorly on underrepresented demographics. Facial feature extraction (like gaze direction) can fail on darker skin tones if the lighting is poor and the training data lacked diversity.
* **Vehicle/Layout Bias:** The dataset features Left-Hand Drive (LHD) vehicles, meaning the model implicitly learns that the driver is on the left side of the frame. If tested on a Right-Hand Drive (RHD) vehicle, the spatial features will be mirrored, and the model will fail unless the input images are flipped horizontally during preprocessing.
* **Device Bias:** The models were trained on data from several years ago. The way people grip modern, massive smartphones might differ from the phones used when the dataset was originally created, potentially skewing inference.
