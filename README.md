# AI-Driven Oocyte Quality Assessment for IVF

## Overview

This project focuses on developing an AI-assisted system for oocyte segmentation and quality assessment in In Vitro Fertilization (IVF) procedures.

The goal is to automate parts of the oocyte evaluation process using Deep Learning and Machine Learning techniques. The system combines image segmentation, feature extraction, and quality prediction to analyze microscopic oocyte images.

---

## Problem Statement

Manual assessment of oocyte quality can be time-consuming and may vary between observers. This project explores how AI can assist in analyzing oocyte images by automatically identifying important structures and predicting quality categories based on extracted features.

---

## Dataset Preparation

* Oocyte images were collected from publicly available sources.
* Images were manually annotated using Label Studio.
* Annotation data was exported and processed for model training.
* Separate datasets were prepared for YOLOv8 and U-Net segmentation models.
* Data augmentation techniques were applied to improve model generalization.

---

## Project Workflow

Input Oocyte Image

↓

Image Annotation (Label Studio)

↓

Dataset Preparation & Augmentation

↓

YOLOv8 / U-Net Segmentation

↓

Feature Extraction

↓

Random Forest Quality Classification

↓

Flask Web Application

↓

Quality Prediction Output

---

## Models Used

### YOLOv8

Used for segmentation and localization of important oocyte structures from microscopic images.

### U-Net

Used for semantic segmentation and extraction of oocyte regions from images.

### Random Forest

Used for quality assessment based on features extracted from segmented images.

---

## Technologies Used

* Python
* PyTorch
* YOLOv8
* U-Net
* Scikit-learn
* Flask
* OpenCV
* NumPy
* Pandas
* Label Studio

---

## Project Structure

```text
features/
python_codes/
sample_results/
static/
templates/

app.py
dataset.yaml
README.md
```

---

## Features

* Oocyte image segmentation
* Automated feature extraction
* Quality prediction pipeline
* Flask-based user interface
* End-to-end machine learning workflow
* Visualization of prediction results

---

## Results

The project successfully demonstrates:

* Automated segmentation of oocyte images
* Feature extraction from segmented regions
* Machine learning-based quality prediction
* Deployment through a Flask web application

Sample outputs and evaluation results can be found in the `sample_results` folder.

---

## Limitations

This project was developed using publicly available image datasets and does not use real clinical outcome data. Therefore, the system should be considered a proof-of-concept research project rather than a clinical decision-support tool.

---

## Future Improvements

* Larger and more diverse datasets
* Clinical validation with expert annotations
* Additional segmentation architectures
* Improved quality scoring methods
* Cloud deployment and scalability

---

