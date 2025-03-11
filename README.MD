# Evaluating CNN Architectures for Noisy Sign Language Recognition  

## Overview  
This project explores the robustness of different Convolutional Neural Network (CNN) architectures in handling noisy and transformed sign language images. Several custom CNN architectures were tested on a dataset of American Sign Language (ASL) images to evaluate their performance under various augmentations that simulate real-world data corruption. The study examines how models degrade when tested on altered data and whether training with augmented data improves robustness.

## Dataset  
- **Source:** [Kaggle: Sign Language Detection Using Images](https://www.kaggle.com/datasets/harshvardhan21/sign-language-detection-using-images)  
- **Total Images:** 42,000  
- **Labels:** 35 classes (A-Z and 1-9)  
- **Resolution:** 128x128x3 (RGB)  

## Objective  
Computer vision models frequently encounter real-world challenges such as lighting variations, occlusions, motion blur, and background noise. This study evaluates how different CNN architectures handle these distortions by introducing controlled augmentations into the dataset and comparing performance against a clean baseline. Three transformation categories were selected to simulate corrupted data:  

1. **Affine Transformations (Skewing)** – Represents variations in camera angles, imperfect hand positioning, and perspective distortion.  
2. **Center Rotations** – Mimics misalignment caused by signer movement or camera shifts.  
3. **Random Noise Injection** – Simulates environmental noise such as shadows, low lighting, or minor occlusions.  

Multiple test sets were generated with increasing levels of distortion for each transformation category. Models were trained on clean images and evaluated on both clean and transformed test sets. A secondary evaluation measured whether training with augmented data improved resilience to distortions compared to models trained only on clean data.

## Model Architectures  
Four CNN architectures were designed and tested:

1. **Small Architecture (Model A)**  
   - Single convolutional layer  
   - 10 filters with a 2x2 kernel  
   - Average pooling for downsampling  

2. **Deep Architecture (Model B)**  
   - Six convolutional layers  
   - Filter sizes: 128 → 64 → 128  
   - Ends with a 4x4 kernel  

3. **Wide Architecture (Model C)**  
   - Single convolutional layer with a large number of filters  
   - 2048 filters with 2x2 kernels  
   - Average pooling  

4. **Narrowing Architecture (Model D)**  
   - Four convolutional layers with progressively fewer filters  
   - Starts with 128 filters (3x3 kernel), then decreases to 64, 32, and 16  

## Data Preprocessing  
- **Exploratory Data Analysis (EDA):** Verified image consistency and balance.  
- **Test Set Transformations:** Created three categories of transformed images:  
  - **Affine (Skew)**
  - **Center Rotation**
  - **Random Noise**  
- **Each category included:**  
  - 75-90 test datasets  
  - 350 images per test set  
  - Increasing degrees of transformation  

## Experimental Setup  
1. Train all models on the original dataset.  
2. Evaluate on both the clean test set and the transformed test sets.  
3. Measure accuracy degradation across transformations.  
4. Compare performance between models trained only on clean data vs. models trained with augmented data.  

## Results  
- **Baseline vs. Augmented Data Training:**  
  - Models trained with augmented data exhibited greater robustness, showing a slower decline in performance.  

- **Impact of Transformations:**  
  - Affine transformations and center rotations resulted in the highest performance degradation.  
  - Models were more resistant to random noise compared to spatial distortions.  

- **Model Performance:**  
  - **Model B (Deep Architecture)** demonstrated the best overall resilience across all test conditions.  
  - **Model D (Narrowing Architecture)** showed no improvement over the baseline.  
  - **Model C (Wide Architecture)** maintained accuracy against random noise but performed poorly under skew and rotation.  

## Conclusion  
- CNNs exhibit varying degrees of robustness when subjected to real-world distortions.  
- Deeper architectures provide more stable performance compared to shallower or narrower models.  
- Training with augmented data significantly reduces accuracy degradation.  
- Real-world applications of sign language recognition require hybrid models capable of handling multimodal input and dynamic environments.  

