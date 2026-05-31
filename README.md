# AI-Driven Oocyte Quality Assessment for IVF

## Overview

This project develops an AI-assisted system for oocyte segmentation and quality assessment in IVF procedures using Deep Learning and Machine Learning techniques.

The system combines YOLOv8 segmentation, U-Net segmentation, feature extraction, and Random Forest classification to automate parts of the oocyte evaluation process.

---

## Project Workflow

Input Oocyte Image

↓

YOLOv8 / U-Net Segmentation

↓

Feature Extraction

↓

Quality Classification

↓

Flask Web Application Output

---

## Technologies Used

- Python
- YOLOv8
- U-Net
- Random Forest
- Flask
- OpenCV
- PyTorch
- NumPy
- Pandas

---

## Project Structure

python_codes/
features/
templates/
static/
sample_results/
app.py

---

## Features

- Oocyte segmentation using YOLOv8
- Oocyte segmentation using U-Net
- Morphological feature extraction
- Quality prediction using Random Forest
- Flask web application for image upload and prediction
- Visualization of segmentation results

---

## Results

- Successful oocyte segmentation
- Automated feature extraction pipeline
- Quality classification model developed
- Interactive Flask deployment

---

## Note

Dataset was collected from publicly available IVF image sources and manually annotated for segmentation tasks.

Model weight files are not included due to GitHub storage limitations.
