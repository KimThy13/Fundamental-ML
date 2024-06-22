# Mini-Project for Fundamentals of Machine Learning Course
![background](./materials/ai_wp.jpg)
This repository contains the code and data for a mini-project on facial expression recognition using machine learning algorithms.

## 📑 Project Policy
- Team: group should consist of 3-4 students.

    |No.| Student Name    | Student ID |
    | --------| -------- | ------- |
    |1|Huỳnh Lưu Vĩnh Phong|21280103|
    |2|Tạ Hoàng Kim Thy|21280083|
    |3|Nguyễn Hải Ngọc Huyền|21280091|
    |4|Trần Hoài Bắc|21280006|

- The submission deadline is strict: **11:59 PM** on **June 22nd, 2024**. Commits pushed after this deadline will not be considered.

## 📦 Project Structure

The repository is organized into the following directories:

- **/data**: This directory contains the facial expression dataset. You'll need to download the dataset and place it here before running the notebooks. (Download link provided below)
- **/notebooks**: This directory contains the Jupyter notebook ```EDA.ipynb```. This notebook guides you through exploratory data analysis (EDA) and classification tasks.

## ⚙️ Usage

This project is designed to be completed in the following steps:

1. **Fork the Project**: Click on the ```Fork``` button on the top right corner of this repository, this will create a copy of the repository in your own GitHub account. Complete the table at the top by entering your team member names.

2. **Download the Dataset**: Download the facial expression dataset from the following [link](https://mega.nz/file/foM2wDaa#GPGyspdUB2WV-fATL-ZvYj3i4FqgbVKyct413gxg3rE) and place it in the **/data** directory:

3. **Complete the Tasks**: Open the ```notebooks/EDA.ipynb``` notebook in your Jupyter Notebook environment. The notebook is designed to guide you through various tasks, including:
    
    1. Prerequisite
    2. Principle Component Analysis
    3. Image Classification
    4. Evaluating Classification Performance 

    Make sure to run all the code cells in the ```EDA.ipynb``` notebook and ensure they produce output before committing and pushing your changes.

5. **Commit and Push Your Changes**: Once you've completed the tasks outlined in the notebook, commit your changes to your local repository and push them to your forked repository on GitHub.


Feel free to modify and extend the notebook to explore further aspects of the data and experiment with different algorithms. Good luck.

## Purpose

The purpose of this dataset is to facilitate the development and evaluation of machine learning and deep learning models for recognizing and classifying human emotions from facial images. This standardized dataset aids researchers and developers in:

- Enhancing model accuracy
- Creating practical applications such as virtual assistants and security systems
- Advancing the field of deep learning by providing consistent and comprehensive data for rigorous testing and benchmarking

## Dataset

The dataset comprises 35,887 grayscale images of faces, each sized 48x48 pixels. These images are pre-processed with automatic alignment, ensuring that the faces are approximately centered and occupy a consistent area within each image. The goal of the dataset is to classify each face according to the emotion it expresses, with seven possible categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

![image of data](./materials/data_images.png)
## Methodology

### Feature Engineering

- **Histogram of Oriented Gradients (HOG):**
  - Convert images to grayscale, compute gradients, create histograms of gradient orientations, and normalize them.
  - Concatenate these histograms to form feature vectors.

- **Principal Component Analysis (PCA):**
  - Standardize the data to have a mean of 0 and a variance of 1.
  - Compute the covariance matrix of the standardized data.
  - Perform eigenvalue decomposition of the covariance matrix to obtain eigenvectors and eigenvalues.
  - Project the data onto the top principal components derived from the eigen decomposition.

### Principal Models

- **K-Nearest Neighbors (KNN):**
  - Classifies data points based on the labels of their k-nearest neighbors.

- **XGBoost:**
  - Gradient boosting framework that sequentially builds weak learners to optimize model performance.

- **LightGBM:**
  - Efficient gradient boosting framework with a focus on leaf-wise tree growth, designed for enhanced accuracy.

- **Multilayer Perceptron (MLP) using PyTorch:**
  - Deep learning model comprising multiple layers of neurons, implemented using the PyTorch framework, capable of learning complex patterns.

## Model Result

### Original Data

|   Model   | Accuracy | Precision | Recall | F1-score |
|:---------:|:--------:|:---------:|:------:|:--------:|
|  XGBoost  |   0.503  |   0.503   |  0.503 |  0.496   |
|    KNN    |   0.515  |   0.503   |  0.515 |  0.501   |
|  LightGBM |   0.520  |   0.523   |  0.520 |  0.513   |
|    MLP    |   0.486  |   0.479   |  0.486 |  0.476   |

### Apply PCA

|   Model     | Accuracy | Precision | Recall | F1-score |
|:-----------:|:--------:|:---------:|:------:|:--------:|
|  XGBoost_PCA |   0.478  |   0.477   |  0.478 |  0.469   |
|    KNN_PCA   |   0.508  |   0.499   |  0.508 |  0.493   |
|  LightGBM_PCA|   0.488  |   0.485   |  0.488 |  0.475   |
|    MLP_PCA   |   0.509  |   0.510   |  0.509 |  0.505   |



