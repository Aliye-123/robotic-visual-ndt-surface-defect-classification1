# AI-Based Surface Defect Classification for Robotic Visual NDT Inspection

This repository contains the computer vision module of a robotic visual inspection system designed for surface-level NDT applications.

The project focuses on classifying surface conditions of industrial parts using a deep learning model. The classification output is planned to be used as a decision signal for a robotic arm, allowing the inspected part to be sorted into the correct category.

## Project Overview

Traditional visual inspection processes may depend heavily on operator experience. This can cause inconsistent results, especially in repetitive industrial quality control tasks.

This project aims to support an automated robotic inspection workflow by using an AI-based image classification model. In the planned system, the part is placed in an inspection area, captured by a camera, classified by the model, and then sorted by a robotic arm according to the predicted class.

## Dataset

A custom dataset was created for this project.

The dataset includes 7 classes:

1. `boyali_hasarsiz` - Painted, no defect  
2. `boyali_orta_hasarli` - Painted, medium defect  
3. `boyali_yuksek_hasarli` - Painted, severe defect  
4. `boyasiz_hasarsiz` - Unpainted, no defect  
5. `boyasiz_orta_hasarli` - Unpainted, medium defect  
6. `boyasiz_yuksek_hasarli` - Unpainted, severe defect  
7. `unknown` - Unknown / out-of-distribution class  

## Data Collection

A custom OpenCV-based data collection script was developed.

The script captures images from a camera and saves the selected region of interest into the corresponding class folder. This helped create a controlled dataset for surface defect classification.

## Model Architecture

The model is based on transfer learning using EfficientNetB0.

Main components:

- EfficientNetB0 backbone
- ImageNet pre-trained weights
- Data augmentation
- Global Average Pooling
- Batch Normalization
- Dropout
- Dense softmax classification layer
- Fine-tuning stage

## Training Pipeline

The training pipeline includes:

- Dataset folder validation
- Train / validation / test split
- Image resizing to 224x224
- Data augmentation
- Transfer learning
- Fine-tuning
- Model checkpointing
- Early stopping
- Learning rate reduction
- Classification report
- Confusion matrix

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- EfficientNetB0
- Transfer Learning

## Robotic System Integration

This project is designed as the vision and decision-making module of a robotic inspection system.

The planned full workflow is:

1. The part is placed in the inspection area.
2. The camera captures the surface image.
3. The AI model classifies the surface condition.
4. The predicted class is sent to the robotic control system.
5. The robotic arm sorts the part into the corresponding box.

## Future Work

- Increase dataset size
- Add more industrial surface categories
- Improve lighting consistency
- Test the model with real-time camera input
- Integrate model output with robotic arm control
- Add ESP32-CAM or industrial camera support
- Deploy the model in a real-time inspection station

## Repository Structure

```text
src/
  collect_dataset.py
  train_model_advanced.py

requirements.txt
README.md
LICENSE
