#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import os
import joblib
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB


# In[4]:


options = ["Nitrogen", "Potassium", "Phosphorous", "Temperature", "Humidity", "pH", "Rainfall"]

all_combinations = []

for i in range(1, len(options)+1):
    for combination in combinations(options, i):
        all_combinations.append(combination)


# In[6]:


indices = []

for item in all_combinations:
    str_item = ''
    for option in options:
        if option in item: str_item += '1'
        else: str_item += '0'
    indices.append(str_item)


# In[14]:


df = pd.read_csv("data.csv")

columns = ["N", "K", "P", "temperature", "humidity", "ph", "rainfall", "label"]
df = df[columns]

df = df.sample(frac=1).reset_index(drop=True)


# In[19]:


def preprocessData(df):
    
    X = df.drop("label", axis=1)
    y = df.label
    
    ordinal_enc = OrdinalEncoder()
    y_temp = ordinal_enc.fit_transform(y.values.reshape(-1, 1))
    
    num_attributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    num_pipeline = Pipeline([
                ("std_scaler", StandardScaler())
                ])
    
    X = num_pipeline.fit_transform(X)
    return X, y_temp


# In[20]:


options_to_column = {'Nitrogen': 'N', 'Potassium': 'P', 'Phosphorous': 'K', 'Temperature': 'temperature', 'Humidity': 'humidity',
                     'pH': 'ph', 'Rainfall': 'rainfall'}

for i, combination in enumerate(all_combinations):
    columns = ["label"]
    
    for item in combination:
        columns.append(options_to_column[item])
    temp_df = df[columns].copy()
    X, y_temp = preprocessData(temp_df)
    
    clf = GaussianNB()
    clf.fit(X, y_temp.ravel())
    
    joblib.dump(clf, "indexed/model-{}.joblib".format(indices[i]))

