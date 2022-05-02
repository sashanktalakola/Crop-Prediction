import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
import joblib

df = pd.read_csv("data.csv")
columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
df = df[columns]
df = df.sample(frac=1).reset_index(drop=True)

X = df.drop("label", axis=1)
y = df.label

ordinal_enc = OrdinalEncoder()
y_temp = ordinal_enc.fit_transform(y.values.reshape(-1, 1))

y = pd.get_dummies(y)

num_attributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
num_pipeline = Pipeline([
            ("std_scaler", StandardScaler())
            ])

X = num_pipeline.fit_transform(X)
clf = GaussianNB()
clf.fit(X, y_temp.ravel())

joblib.dump(clf, "model.joblib")
joblib.dump(ordinal_enc, "objects/ordinal_enc.joblib")
joblib.dump(num_pipeline, "objects/num_pipeline.joblib")