# Evaluating CNN Architectures for Noisy Sign Language Recognition  

## Overview  
This project explores the robustness of different Convolutional Neural Network (CNN) architectures when handling noisy and transformed sign language images. We tested several custom CNN architectures on a dataset of American Sign Language (ASL) images and evaluated their performance under different transformations, including skew (affine), center rotation, and random noise.

## Dataset  
- **Source:** [Kaggle: Sign Language Detection Using Images](https://www.kaggle.com/datasets/harshvardhan21/sign-language-detection-using-images)  
- **Total Images:** 42,000  
- **Labels:** 35 classes (A-Z and 1-9)  
- **Resolution:** 128x128x3 (RGB)  

## Research Question  
- How do different CNN architectures handle noise, lighting variations, and background transformations in real-world conditions?  
- How does performance degrade as data transformations increase in intensity?  

## Model Architectures  
Architectures found in models folder:  

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

## Results  
- **Baseline vs. Augmented Data Training:**  
  - Models generally improved with exposure to augmented data.  
  - Slower accuracy degradation when trained on noisy data.  

- **Impact of Transformations:**  
  - Affine and center rotation transformations degraded performance the most.  
  - Models were more robust to random noise, which better mimics real-world lighting variations.  

- **Model Performance:**  
  - **Model B (Deep Architecture)** performed best overall, showing resilience to all transformations.  
  - **Model D (Narrowing Architecture)** did not improve over the baseline.  
  - **Model C (Wide Architecture)** was effective against noise but less so against skew and rotation.  

## Conclusion  
- CNNs remain a strong choice for Sign Language Recognition (SLR), but robustness varies by architecture.  
- Real-world applications require hybrid models that can handle multimodal input and environmental variations.  
- Future work includes incorporating real-time video recognition and expanding datasets to improve model generalization.  

## References  
- [Sign Language Detection Using Images (Kaggle)](https://www.kaggle.com/datasets/harshvardhan21/sign-language-detection-using-images)  
- Research papers on deep learning-based sign language recognition systems.  

## Repository  
- **GitHub:** [Project Repository](https://github.com/numbersrcool/207_final/tree/main)  

