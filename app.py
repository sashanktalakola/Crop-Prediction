import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
import time
import joblib

st.title("Crop Prediction:rice::corn:")
ordinal_enc = joblib.load("model/objects/ordinal_enc.joblib")
num_pipeline = joblib.load("model/objects/num_pipeline.joblib")

class_names = ordinal_enc.categories_[0]

st.header("\nEnter Values to Predict the Optimal Crop")
#clf = joblib.load("model/model.joblib")

options = ["Nitrogen", "Potassium", "Phosphorous", "Temperature", "Humidity", "pH", "Rainfall"]

available_features = st.multiselect("Select available features", options=options, default="Temperature")
model_index = ''
for option in options:
	if option in available_features:
		model_index += "1"
	else:
		model_index += "0"

try:
    clf = joblib.load("model/indexed/model-{}.joblib".format(model_index))
except FileNotFoundError:
    st.warning("Select atleast one feature(Preferably 3 or more)")
    clf = joblib.load("model/model.joblib")
n = st.number_input("Nitrogen Ratio", min_value = 0.00)
k = st.number_input("Potassium Ratio", min_value = 0.00)
p = st.number_input("Phosphorous Ratio", min_value = 0.00)
temperature = st.number_input("Temperature(Celcius)", min_value = 0.00)
humidity = st.number_input("Humidity", min_value = 0.00)
ph = st.number_input("pH of the Soil", min_value = 1.000000, max_value = 14.000000)
rainfall = st.number_input("Rainfall", min_value = 0.00)

inputs  = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
new_inputs = []

for i, input_value in zip(model_index, inputs[0]):
	if i == "1":
		new_inputs.append(input_value)

new_inputs = np.array(new_inputs).reshape(-1, 1)
prediction = clf.predict(new_inputs)[0]
index = int(prediction)
predicted_crop = class_names[index]

if st.button("Predict"):
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.03)
        progress.progress(i+1)
    st.success("Predicted crop : " + predicted_crop.capitalize())
    info_probabilities = ''

    i = 0
    for predicted_probability in clf.predict_proba(new_inputs)[0]:
    	info_probabilities = "{} - {}\n".format(class_names[i].capitalize(), predicted_probability)
    	i += 1
    	st.write(info_probabilities)
