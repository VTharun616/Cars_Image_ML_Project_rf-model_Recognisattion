# 🚗 Car Image Classification using Machine Learning
This project classifies car images into two categories: Audi and Toyota using Machine Learning and image processing techniques.
The model is trained on flattened image pixels and evaluates multiple classification algorithms to find the best-performing model.

## 1. Project Overview
In this project, I performed:
Image data loading from dataset folders
Image resizing and preprocessing
Feature extraction by flattening images
Label encoding based on folder names
Training multiple classification models
Model evaluation using accuracy, precision, recall, F1-score, and AUC-ROC
Final model selection based on best performance

## 2. Technologies Used

-Python
-NumPy
-Pandas
-OpenCV
-Matplotlib
-Scikit-learn
-imutils

## 3. Image Preprocessing

Loaded images using OpenCV
Resized images to 128 × 128
Converted images to NumPy arrays
Flattened images into 1D feature vectors
Normalized pixel values using MinMaxScaler (0–255 → 0–1)

## 4. Feature Engineering
Each image converted into a numerical vector
Labels extracted from folder names
Final dataset created using DataFrame

## 5. Algorithms Applied
I applied multiple classification algorithms for comparison:
Logistic Regression
Support Vector Machine (SVM)
Random Forest Classifier

## 6.🏆 Final Model Selection
✔ Random Forest Classifier
Selected 
because it achieved:
Highest Accuracy
Better Precision and Recall
Best F1-score
Strong AUC-ROC performance
More stable and robust predictions

## 7. Evaluation Metrics Used
Accuracy Score
Confusion Matrix
Classification Report
Precision
Recall
F1 Score
ROC Curve
AUC-ROC Score

## 8. Model Workflow
Image Dataset:
- Resize Images (128×128)
- Flatten Images
- Label Encoding
- Train-Test Split
- MinMax Scaling
- Model Training
- Evaluation
- Best Model Selection

## 9. Results Summary
Random Forest performed best among all models
SVM gave good results but slightly lower than RF
Logistic Regression was baseline model

## 10. Future Improvements
Use CNN (Deep Learning) for better accuracy
Apply data augmentation
Increase dataset size
Use transfer learning (ResNet, VGG16)
Deploy using Streamlit or Flask

## Author
**Tharun Kumar Vadde**

Aspiring AI/ML Engineer focused on:
-Machine Learning
-Deep Learning
-Computer Vision
-NLP
-Generative AI projects
