# Robotic Predictive Maintenance with Machine Learning

This project focuses on developing a predictive maintenance system for robots, using sensor data and machine learning techniques to determine when a robot might need maintenance. The main application is industrial maintenance, where robots are used in critical processes, and their maintenance must be accurate and timely.

## Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation and Execution](#installation-and-execution)
- [Usage](#usage)
- [System Architecture](#system-architecture)
- [Learning Strategies](#learning-strategies)

## Introduction

The project uses sensor data (such as RMS, skewness, kurtosis, etc.) and analyzes the data using machine learning techniques (such as Random Forest) to predict when a robot will need maintenance. Principal Component Analysis (PCA) is also applied for dimensionality reduction of the data.

## Requirements

To run the project, you need to install the following libraries:

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

You can install the required libraries using the command:

```bash
pip install -r requirements.txt

System Architecture:
The system includes the following steps:
Data Loading: Data is loaded from a CSV file and stored in a DataFrame.
Data Preparation: Columns containing the 'fault' label are isolated and used for model training.
Model Training: The model is trained using the Random Forest algorithm to predict maintenance needs.
Evaluation and Visualizations: The confusion matrix, feature importances, and histograms are generated to evaluate the model.


Learning Strategies:
The project employs the following machine learning techniques:
Random Forest Classifier: Used for classifying the data and predicting robot failures.
Principal Component Analysis (PCA): Applied to reduce the dimensionality of the data and improve model performance.
