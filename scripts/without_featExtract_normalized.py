#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from multiprocessing import Pool, cpu_count


# In[2]:


# Import the necessary classifiers 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


# In[3]:


# Define the folder containing your csv files
folder_path = "D:\MT dataset\mcsadc-IM motor-rotorbarfailure-2023\combined files_training set\Training data_with label"

# Get a list of all csv files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')] # creates a list of all filenames in the directory folder_path that end with .csv

feat_vars = ['Time', 'SC_a', 'SC_b', 'SC_c', 'Speed']
class_var = 'Health State'

# initialize empty lists to store data from all the files
all_X = []
all_Y_numeric = []

# loop through each csv file
for i, file_name in enumerate(csv_files):
    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)

    # Read data from the current csv file
    data = pd.read_csv(file_path)

    # Debug statement to print the contents of the data
    print(f'Data from file {i + 1}:')
    print(data.head())

    # Extract the predictor variables (features) and response variables (classes)
    X = data[feat_vars].values # features
    Y_str = data[class_var].values # String labels

    # Initialize an empty array to store the encoded labels
    Y_numeric = np.zeros(len(Y_str))

    # Loop through each row of Y_str
    for j in range(len(Y_str)):
        # Debug statement to print the current value of Y_str
        # print(f'Y_str[{j}]: {Y_str[j]}')

        # Convert Y_str to lowercase for case-sensitive matching 
        lower_Y_str = Y_str[j].lower()

        # Check for the presence of specific health states
        if '1 broken bar' in lower_Y_str:
            Y_numeric[j] = 1 # Encode as 1 for 1 broken bar
        elif '2 broken bars' in lower_Y_str:
            Y_numeric[j] = 2 # Encode as 2 for 2 broken bars
        else:
            Y_numeric[j] = 0 # Encode as 0 for healthy state

    # Append data from the current file to the lists
    all_X.append(X)
    all_Y_numeric.append(Y_numeric)

    # Debug statement to indicate successful processing
    print(f'Data from file{i + 1} processed successfully.')
    
# Convert lists to numpy arrays
all_X = np.vstack(all_X)
all_Y_numeric = np.hstack(all_Y_numeric)

# Normalize the data
scaler = StandardScaler()
all_X_normalized = scaler.fit_transform(all_X)

# Debug statement to print the normalized data
print('Normalized data:')
print(all_X_normalized)

print('all_Y_numeric:')
print(all_Y_numeric)

# now 'all_X_normalized' contains the normalized features and 'all_Y_numeric' contains the labels


# In[4]:


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

# Assuming all_X_normalized and all_Y_numeric are already defined
# Augment the dataset
augmented_X, augmented_Y = augment_data(all_X_normalized, all_Y_numeric)

# Combine original and augmented data
final_X = np.vstack((all_X_normalized, augmented_X))
final_Y = np.hstack((all_Y_numeric, augmented_Y))

# Shuffle the data
final_X, final_Y = shuffle(final_X, final_Y, random_state=42)


# In[5]:


# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(final_X, final_Y, test_size=0.2, random_state=42)

# Initialize the models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Bilayered Neural Network': MLPClassifier(hidden_layer_sizes=(10,),max_iter=1000),
    'Trilayered Neural Network': MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000),
    'Random Forest': RandomForestClassifier(n_jobs=-1)
}


# In[11]:


DTmodel = DecisionTreeClassifier()
DTmodel.fit(X_train, y_train)
y_pred_DT = DTmodel.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred_DT)

# Print the accuracy
print(f'Decision Tree Accuracy: {acc * 100:.2f}%')


# In[6]:


# Train and evaluate each model using a for loop
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print the accuracy
    print(f'{model_name} Accuracy: {accuracy * 100:.2f}%')
    


# In[7]:


# Function to train and evaluate the models - using multiprocessing 
def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return model_name, accuracy

# Train and evaluate each model in parallel
with Pool(cpu_count()) as pool:
    results = pool.starmap(train_and_evaluate, [(name, model, X_train, y_train, X_test, y_test) for name, model in models.items()])

# Print the accuracy of each model
for model_name, accuracy in results:
    print(f'{model_name} Accuracy: {accuracy * 100:.2f}%')

