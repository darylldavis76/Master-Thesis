# Machine Learning for Induction Motor Fault Diagnosis

This repository contains the machine learning component of my master thesis, which focuses on diagnosing faults in induction motors using operational data. 
The project leverages advanced feature extraction and machine learning techniques to classify and identify different fault conditions in induction motors.

## Overview
The primary goal of this project is to classify the operational state of a three-phase induction motor under various fault conditions. The faults considered include:

- Healthy condition
- 1 broken bar
- 2 broken bars

These conditions are evaluated at low, medium, and high RPM ranges.

## Data Description
The dataset consists of stator current signals collected from a three-phase induction motor operating under different fault conditions. The signals were captured under the following scenarios:

* Healthy condition
* 1 broken bar
* 2 broken bars


Each condition tested at low, medium, and high RPM ranges

## Feature Extraction

The stator current signals were analyzed to extract features in the time, frequency, and time-frequency domains. The following techniques were employed:

- Fast Fourier Transform (FFT): Used for frequency domain analysis.
- Short-Time Fourier Transform (STFT): Used for time-frequency domain analysis.

## Feature Selection

To select the most relevant features, we employed feature selection techniques, which included:

* Recursive Feature Elimination (RFE): An iterative method that removes the least important features based on model accuracy.
* SelectKBest: Selects the top k features based on statistical tests.

Both techniques were integrated with Grid Search Cross-Validation (CV) to optimize the feature selection process.

## Machine Learning Models

The following machine learning models were used for classification:

* Support Vector Machine (SVM): A supervised learning model that analyzes data for classification and regression analysis.
* k-Nearest Neighbors (kNN): A simple, instance-based learning algorithm used for classification.

Both models were fine-tuned using Grid Search CV to find the optimal hyperparameters for the classification task.

## Results

The trained models are capable of accurately classifying the operational state of the induction motor, identifying faults, and distinguishing between healthy and faulty conditions.
