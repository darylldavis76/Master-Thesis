#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
from scipy.signal import stft, welch
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


# In[5]:


# Function to compute time features
def compute_time_features(data, sample_rate):
    features = {}
    features['mean'] = np.mean(data)
    features['RMS'] = np.sqrt(np.mean(np.square(data)))
    features['StandardDeviation'] = np.std(data)
    features['ShapeFactor'] = np.sqrt(np.mean(np.square(data))) / np.mean(np.abs(data))
    features['SNR'] = np.mean(data) / np.std(data)
    features['THD'] = np.sqrt(np.sum(np.square(data[1:])) / np.square(data[0])) # Total Harmonic Distortion
    features['SINAD'] = np.mean(data) / np.sqrt(np.mean(np.square(data - np.mean(data))))
    features['PeakValure'] = np.max(np.abs(data))
    features['CrestFactor'] = np.max(np.abs(data)) / np.sqrt(np.mean(np.square(data)))
    features['ClearanceFactor'] = np.max(np.abs(data)) / np.mean(np.sqrt(np.abs(data)))
    features['ImpulseFactor'] = np.max(np.abs(data)) / np.mean(np.abs(data))
    return features

# Function to compute frequency features
def compute_frequency_features(data, sample_rate):
    freq_domain = np.abs(fft(data))
    freqs = fftfreq(len(data), 1 / sample_rate)
    features = {}
    features['MeanFrequency'] = np.mean(freq_domain)
    features['MedianFrequency'] = np.median(freq_domain)
    features['BandPower'] = np.sum(freq_domain ** 2)
    features['OccupiedBandwidth'] = np.sum(freq_domain > 0.05 * np.max(freq_domain))
    features['PowerBandwidth'] = np.sum(freq_domain > 0.5 * np.max(freq_domain))
    features['PeakAmplitude'] = np.max(freq_domain)
    features['PeakLocation'] = freqs[np.argmax(freq_domain)]
    return features

# Function to compute time-frequency features using STFT
def compute_time_freq_features(data, sample_rate):
    f, t, Zxx = stft( data, fs=sample_rate, nperseg=256)
    magnitude = np.abs(Zxx)

    features={}
    features['SpectralKurtosis'] = kurtosis(magnitude, axis=None)
    features['SpectralSkewness'] = skew(magnitude, axis=None)
    features['SpectralCrest'] = np.max(magnitude) / np.mean(magnitude)
    features['SpectralFlatness'] = np.exp(np.mean(np.log(magnitude))) / np.mean(magnitude)
    features['SpectralEntropy'] = -np.sum(magnitude * np.log2(magnitude), axis=None)
    features['SpectralCentroid'] = np.sum(f[:, np.newaxis] * magnitude, axis=0) / np.sum(magnitude, axis=0)
    features['SpectralSpread'] = np.sqrt(np.sum((f[:, np.newaxis] - features['SpectralCentroid'])**2 * magnitude, axis=0) / np.sum(magnitude, axis=0))
    features['SpectralRolloff'] = np.sum(magnitude, axis=0)[np.newaxis] * 0.85
    features['TFRidges'] = np.argmax(magnitude, axis=0)
    features['InstantaneousBandwidth'] = np.std(magnitude, axis=0)
    features['InstantaneousFrequency'] = np.mean(magnitude, axis=0)
    features['MeanEnvelopeEnergy'] = np.mean(np.abs(magnitude), axis=0)
    features['WaveletEntropy'] = -np.sum(np.square(magnitude) * np.log2(np.square(magnitude)), axis=None)
    return features

# Function to aggregate features
def aggregate_features(vector_data):
    aggregated_features = {}
    for key, vec in vector_data.items():
        aggregated_features[f'{key}_mean'] = np.mean(vec)
        aggregated_features[f'{key}_std'] = np.std(vec)
        aggregated_features[f'{key}_min'] = np.min(vec)
        aggregated_features[f'{key}_max'] = np.max(vec)
        aggregated_features[f'{key}_range'] = np.ptp(vec)
    return aggregated_features

# Function to normalize features
def normalize_features(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Main function to extract features from all CSV files in a folder
def extract_m_features(folder_path):
    feature_table = pd.DataFrame()
    sample_rate = 1428.57

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = np.loadtxt(file_path, delimiter=',')

            time_features = compute_time_features(data, sample_rate)
            frequency_features = compute_frequency_features(data, sample_rate)
            time_frequency_features = compute_time_freq_features(data, sample_rate)

            aggregated_time_frequency_features = aggregate_features(time_frequency_features)

            combined_features = {**time_features, **frequency_features, **aggregated_time_frequency_features}

            combined_feature_table = pd.DataFrame([combined_features])
            
            feature_table = pd.concat([feature_table, combined_feature_table], ignore_index=True)

    feature_table = feature_table.apply(normalize_features, axis=0)
    feature_table['HealthState'] = 0

    return feature_table

# Usage
folder_path = "D:\MT dataset\mcsadc-IM motor-rotorbarfailure-2023\mcsadc-IM-motor-rotorbarfailure-OG\processed_CSV\SC_abc_healthy"
healthy_feature_table = extract_m_features(folder_path)



# In[6]:


combined_feature_table = pd.concat([healthy_feature_table, BB1_feature_table, BB2_feature_table], ignore_index=True)
X = combined_feature_table.drop(columns=['HealthState'])
y = combined_feature_table['HealthState']


# In[7]:


# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Debug statement to print the normalized data
print('Normalized X data:')
print(X_normalized)

print('Y_numeric data:')
print(y)

# now 'X_normalized' contains the normalized features and 'y' contains the labels


# In[10]:


# Define data augmentation functions
def add_noise(data, noise_level=0.01):
    noise = noise_level * np.random.normal(size=data.shape)
    return data + noise

def scale(data, scaling_factor=1.1):
    return data * scaling_factor

def time_shift(data, shift_max=2):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(data, shift)

def augment_data(X, Y, augmentations=5):
    augmented_X, augmented_Y = [], []
    for _ in range(augmentations):
        for x, y in zip(X, Y):
            augmented_X.append(add_noise(x))
            augmented_X.append(scale(x))
            augmented_X.append(time_shift(x))
            augmented_Y.extend([y, y, y])
    return np.array(augmented_X), np.array(augmented_Y)

# Assuming X_normalized and y are already defined
# Augment the dataset
augmented_X, augmented_Y = augment_data(X_normalized, y)

# Combine original and augmented data
X_final = np.vstack((X_normalized, augmented_X))
y_final = np.hstack((y, augmented_Y))

# Shuffle the data
X_final, y_final = shuffle(X_final, y_final, random_state=42)

# Debug statement to print the normalized data
print('Final X data:')
print(X_final)

print('Final y:')
print(y_final)


# In[19]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# Define the model and parameter grid for SVM
svm_model = SVC()
svm_param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}

# Define the model and parameter grid for kNN
knn_model = KNeighborsClassifier()
knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Create the RFE (Recursive Feature Elimination) model with cross-validation for SVM with a linear kernel
linear_svm = SVC(kernel='linear')
rfecv_svm = RFECV(estimator=linear_svm, step=1, cv=5, scoring='accuracy', min_features_to_select=15)

# Create the pipeline with imputer and feature selection for SVM
pipeline_svm = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('rfe', rfecv_svm),
])

# Perform grid search on the pipeline
grid_search_svm = GridSearchCV(pipeline_svm, param_grid={
    'rfe__estimator__C': [0.1, 1, 10, 100]}, cv=5, scoring='accuracy')

# Create the RFE model with cross-validation for kNN
rfecv_knn = RFECV(estimator=knn_model, step=1, cv=5, scoring='accuracy', min_features_to_select=15)

# Create the pipeline with imputer and feature selection for kNN
pipeline_knn = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('rfe', rfecv_knn)
])

# Perform grid search on the pipeline for kNN
grid_search_knn = GridSearchCV(pipeline_knn, param_grid=knn_param_grid, cv=5, scoring='accuracy')


# In[20]:


# Fit the pipeline and perform hyperparameter tuning for SVM
grid_search_svm.fit(X_train, y_train)
svm_best_estimator = grid_search_svm.best_estimator_

# Get the selected features for SVM
selected_features_svm = X.columns[rfecv_svm.support_]

# Evaluate SVM model on the test set
X_test_selected_svm = rfecv_svm.transform(X_test)
svm_predictions = svm_best_estimator.predict(X_test_selected_svm)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("SVM selected features:", selected_features_svm)
print("SVM test accuracy:", svm_accuracy)


# In[ ]:


# Fit the pipeline and perform hyperparameter tuning for kNN
grid_search_knn.fit(X_train, y_train)
knn_best_estimator = grid_search_knn.best_estimator_

# Get the selected features for kNN
selected_features_knn = X.columns[rfecv_knn.support_]

# Evaluate kNN model on the test set
X_test_selected_knn = rfecv_knn.transform(X_test)
knn_predictions = knn_best_estimator.predict(X_test_selected_knn)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print("kNN selected features:", selected_features_knn)
print("kNN test accuracy:", knn_accuracy)


# 
