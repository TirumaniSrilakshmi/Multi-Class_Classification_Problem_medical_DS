import streamlit as st
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense



def load_model(model_name):
    with open(f'models/{model_name}.pkl','rb') as file:
        model = pickle.load(file)
    return model
    

st.title("Medical Test Result Classification")
st.write("Select a model and input your data:")

# model_option = st.selectbox("Choose a model", ["Logistic Regression", "Decision Tree","Random Forest",
#                                                "K Nearnest Neighbour","Naive Bayes","XGBoost"])

input_models = ({
    "Logistic Regression": load_model('Logistic_Regression'),
    "Random Forest": load_model('Random_Forest'),
    "XGBoost": load_model('XGBoost'),
    "Naive Bayes": load_model('Naive_Bayes'),
    "K Nearest Neighbors": load_model('K_Nearest_Neighbors'),
    "Decision Tree": load_model('Decision_Tree')
})

model_option = st.selectbox("Choose a model", list(input_models.keys()))

st.subheader("Enter patient details:")

age = st.number_input("Age",min_value = 0,max_value = 120)
days  = st.number_input("Days",min_value = 0, max_value = 100)
gender = st.selectbox("Gender",['Male','Female'])
blood_type = st.selectbox("Blood Type", ['A+','A-','B+','B-','AB+','AB-','O+','O-'])
medical_condition = st.selectbox("Medical Condition",['Cancer','Obesity','Diabetes','Asthma','Hypertension','Arthritis'])
admission_type = st.selectbox("Admission Type",['Urgent','Emergency','Elective'])
medication = st.selectbox("Medication",['Paracetamol','Ibuprofen','Aspirin','Penicillin','Lipitor'])

input_data = pd.DataFrame({
'Age': [age],
'Days_numeric' : [days],
'Gender_Male'	: [1 if gender=="Male" else 0],
'Blood_Type_A_minus' :[1 if blood_type =="A-" else 0],
'Blood_Type_AB_plus' : [1 if blood_type =="AB+" else 0],	
'Blood_Type_AB_minus' : [1 if blood_type =="AB-" else 0],	
'Blood_Type_B_plus' :	[1 if blood_type =="B+" else 0],
'Blood_Type_B_minus' : [1 if blood_type =="B-" else 0],
'Blood_Type_O_plus' : [1 if blood_type =="O+" else 0],
'Blood_Type_O_minus' : [1 if blood_type =="O-" else 0],
'Medical_Condition_Asthma' :[1 if medical_condition == "Asthma" else 0 ],
'Medical_Condition_Cancer' : [1 if medical_condition == "Cancer" else 0 ],
'Medical_Condition_Diabetes' : [1 if medical_condition == "Diabetes" else 0 ],	
'Medical_Condition_Hypertension' : [1 if medical_condition == "Hypertension" else 0 ],
'Medical_Condition_Obesity' : [1 if medical_condition == "Obesity" else 0 ],
'Admission_Type_Emergency' : [1 if admission_type == "Emergency" else 0 ],
'Admission_Type_Urgent' : [1 if admission_type == "Urgent" else 0 ],
'Medication_Ibuprofen' : [1 if medication == "Ibuprofen" else 0 ],
'Medication_Lipitor' : [1 if medication == "Lipitor" else 0 ],
'Medication_Paracetamol' : [1 if medication == "Paracetamol" else 0 ],
'Medication_Penicillin'      :[1 if medication == "Penicillin" else 0 ]
})

expected_order=['Age', 'Days_numeric', 'Gender_Male', 'Blood_Type_A_minus',
       'Blood_Type_AB_plus', 'Blood_Type_AB_minus', 'Blood_Type_B_plus',
       'Blood_Type_B_minus', 'Blood_Type_O_plus', 'Blood_Type_O_minus',
       'Medical_Condition_Asthma', 'Medical_Condition_Cancer',
       'Medical_Condition_Diabetes', 'Medical_Condition_Hypertension',
       'Medical_Condition_Obesity', 'Admission_Type_Emergency',
       'Admission_Type_Urgent', 'Medication_Ibuprofen', 'Medication_Lipitor',
       'Medication_Paracetamol', 'Medication_Penicillin']

input_data = input_data.reindex(columns = expected_order)

selected_model = input_models[model_option]
print(f"Selected model type: {type(selected_model)}")  


if st.button("Predict"):
    print(f"Input Data: /n {input_data}")  
    print(f"Predicting with model: {model_option}")
    #model = input_models[model_option]
    prediction = selected_model.predict(input_data)
    result_mapping  = {0:"Normal",1:"Abnormal",2:"Inconclusive"}
    result = result_mapping.get(prediction[0], "Unknown result")
    st.write(f"The Preditced test result is : {result}")
