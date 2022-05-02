import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("data.csv")
columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
df = df[columns]
df = df.sample(frac=1).reset_index(drop=True)

X = df.drop("label", axis=1)
y = df.label

ordinal_enc = OrdinalEncoder()
y_ord = ordinal_enc.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y_ord, test_size=.2)

num_attributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
num_pipeline = Pipeline([
            ("std_scaler", StandardScaler())
            ])

X_train = num_pipeline.fit_transform(X_train)
clf = GaussianNB()
clf.fit(X_train, y_train)

X_test = num_pipeline.transform(X_test)
y_pred = clf.predict(X_test)

class_names = ordinal_enc.categories_[0]
print(classification_report(y_test, y_pred, target_names=class_names))