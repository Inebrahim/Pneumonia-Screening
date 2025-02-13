# Pneumonia Detection using Chest X-ray Images

## Overview
This project aims to develop a deep learning model to classify chest X-ray images and detect whether a patient has **pneumonia**. The model uses Convolutional Neural Networks (CNNs) for image classification and object recognition, with performance evaluated using various metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **ROC AUC**.

The project focuses on high sensitivity for pneumonia detection, ensuring that the model correctly identifies most of the pneumonia cases (with a high recall score).

## Key Features
- **Deep Learning Model**: CNN (Convolutional Neural Network)
- **Libraries Used**:
  - Python (TensorFlow/Keras, Numpy, Pandas)
  - Deep Learning (CNN)
  - Image Processing (OpenCV, PIL)
  - Scikit-learn (for metrics)
- **Performance Metrics**:
  - Accuracy: 89.26%
  - Precision: 85.49%
  - Recall: 99.74% (high sensitivity)
  - F1-Score: 92.07%
  - ROC AUC: 85.77%

- **Task**: Pneumonia detection from Chest X-ray images.
- **Dataset**: Chest X-ray images dataset containing labeled data for pneumonia and healthy cases.

## Requirements
To run this project, you will need the following libraries:
- Python 3.x
- TensorFlow 2.x (for model training and evaluation)
- Keras
- Numpy
- Pandas
- Scikit-learn
- OpenCV (optional for image processing)
- Matplotlib (for visualization)

You can install all dependencies using pip:
```bash
pip install tensorflow keras numpy pandas scikit-learn opencv-python matplotlib
```

## Dataset
The dataset used in this project consists of chest X-ray images labeled as **Pneumonia** or **Normal**. You can find the dataset [here](add-dataset-link-here).

## Models and Results

### Model: Convolutional Neural Network (CNN)
- **Architecture**: The model consists of convolutional layers followed by pooling layers and fully connected layers, designed to extract relevant features from the X-ray images.
  
### Performance Metrics:
- **Accuracy**: 89.26%
  - The model correctly classifies 89.26% of the images.
  
- **Precision**: 85.49%
  - Precision indicates the model's ability to avoid false positives, with a solid rate of correctly identifying pneumonia instances among all the predicted positives.

- **Recall**: 99.74%
  - High recall ensures the model is highly sensitive to detecting pneumonia, with a minimal chance of missing a positive case.

- **F1-Score**: 92.07%
  - The F1-score is high, indicating a good balance between precision and recall, reflecting the model's overall effectiveness.

- **ROC AUC**: 85.77%
  - The ROC AUC score suggests that the model has good discriminatory power, distinguishing between pneumonia and non-pneumonia cases.

## How to Run
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/pneumonia-detection-xray.git
   ```
2. Navigate to the project directory:
   ```bash
   cd pneumonia-detection-xray
   ```
3. Prepare the dataset:
   - Place the X-ray image dataset into the appropriate directory as required by the script (refer to the specific dataset structure in the project).

4. Run the model training script:
   ```bash
   python train_model.py
   ```

5. The model will be trained, and you will receive the results, including performance metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **ROC AUC**.

6. After training, the model will save the trained weights in the `models/` directory. You can use the saved weights to make predictions on new X-ray images:
   ```bash
   python predict.py --image path_to_new_image
   ```

## Evaluation Metrics
- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were correctly identified by the model.
- **F1-Score**: A balance between precision and recall, useful when there is class imbalance.
- **ROC AUC**: Measures the ability of the model to discriminate between the classes (pneumonia vs. normal).

## Acknowledgments
- The project uses TensorFlow/Keras for deep learning model training, Scikit-learn for performance evaluation, and other image processing libraries such as OpenCV.
- Dataset credits to the creators for providing the labeled chest X-ray images for pneumonia detection.
