import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('model_with_threshold.sav','rb') as f:
    model=pickle.load(f)

st.title("Churn Shield: Predicting and Preventing Customer Attrition to Minimize Revenue Loss in the Telco Industry")
st.sidebar.header("Customer Information")

def user_input():
    Dependents=st.sidebar.selectbox('Dependents',['Yes','No'])
    tenure=st.sidebar.number_input('Tenure (months)',min_value=0,max_value=100,value=12)
    InternetService=st.sidebar.selectbox('Internet Service',['DSL','Fiber optic','No'])
    OnlineSecurity=st.sidebar.selectbox('Online Security',['Yes', 'No','No internet service'])
    OnlineBackup=st.sidebar.selectbox('Online Backup',['Yes', 'No','No internet service'])
    DeviceProtection=st.sidebar.selectbox('Device Protection',['Yes','No','No internet service'])
    TechSupport=st.sidebar.selectbox('Tech Support',['Yes','No','No internet service'])
    Contract=st.sidebar.selectbox('Contract',['Month-to-month','One year','Two year'])
    PaperlessBilling=st.sidebar.selectbox('Paperless Billing',['Yes','No'])
    MonthlyCharges=st.sidebar.number_input('Monthly Charges',min_value=0.0,max_value=200.0,value=70.0)
    
    df=pd.DataFrame({
        'Dependents':[Dependents],
        'tenure':[tenure],
        'InternetService':[InternetService],
        'OnlineSecurity':[OnlineSecurity],
        'OnlineBackup':[OnlineBackup],
        'DeviceProtection':[DeviceProtection],
        'TechSupport':[TechSupport],
        'Contract':[Contract],
        'PaperlessBilling':[PaperlessBilling],
        'MonthlyCharges':[MonthlyCharges],
    })
    return df

data=user_input()
st.subheader("Customer Data")
st.write(data.transpose())

threshold=0.35
proba=model.predict_proba(data)[0][1]
pred=int(proba>=threshold)

st.subheader("Prediction Result")
st.write(f"Probability of Churn: {proba:.2f}")
st.write(f"Predicted Class: {pred} ({'Churn' if pred==1 else 'Not Churn'})")

st.subheader("Batch Prediction with CSV")
uploaded_file=st.file_uploader("Upload CSV File",type=["csv"])
if uploaded_file:
    df_batch=pd.read_csv(uploaded_file)
    st.write("Preview:",df_batch.head())
    probas=model.predict_proba(df_batch)[:,1]
    preds=(probas>=threshold).astype(int)
    df_batch['Churn_Probability']=probas
    df_batch['Churn_Prediction']=preds
    st.write("Predictions:",df_batch.head(20))