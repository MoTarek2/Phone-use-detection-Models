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
   * Uses multithreading (`ThreadPoolExecutor`) for image loading, resizing images to 19
