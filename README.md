# 🌾 Rice Leaf Disease Prediction Using CNN

A deep learning-based web application that predicts rice leaf diseases from uploaded leaf images using a MobileNetV2 transfer learning model.

## 📌 Project Overview

This project uses computer vision and deep learning to classify rice leaves into three disease categories:

- Bacterial Leaf Blight
- Brown Spot
- Leaf Smut

The trained MobileNetV2 model is integrated with a Flask web application where users can upload a rice leaf image and receive the predicted disease along with the model confidence score.

## 🎯 Objective

The main objective is to develop an image classification system that can automatically identify common rice leaf diseases and provide quick predictions through a user-friendly web interface.

## 🧠 Model

Transfer Learning with **MobileNetV2**

- Image Size: 128 × 128
- Number of Classes: 3
- Optimizer: Adam
- Data Augmentation: Applied during training
- Best Validation Accuracy: **82.61%**

## 📂 Dataset

The dataset contains 119 rice leaf images distributed across three classes:

| Disease | Images |
|---|---:|
| Bacterial Leaf Blight | 40 |
| Brown Spot | 40 |
| Leaf Smut | 39 |
| **Total** | **119** |

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Flask
- NumPy
- Pillow
- HTML
- CSS
- JavaScript
- Google Colab
- VS Code

## 🌐 Web Application

The Flask application allows users to:

1. Upload a rice leaf image
2. Process the image automatically
3. Predict the disease using the trained model
4. Display the predicted disease
5. Display the confidence score

## 📁 Project Structure

```text
Riceleaf_CNN-webapp
│
├── app.py
├── class_names.json
├── rice_leaf_mobilenetv2.keras
├── requirements.txt
├── .gitignore
│
├── static
│   └── uploads
│
├── templates
│   └── index.html
│
└── venv