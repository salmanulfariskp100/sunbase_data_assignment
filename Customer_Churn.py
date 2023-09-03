# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 00:33:05 2023

@author: Sagar N.R
"""

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import os
import base64

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://www.questionpro.com/blog/wp-content/uploads/2018/02/Customer-Churn.jpg");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stSidebar"] > div:first-child {{

background-position: center; 
background-no-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
html_temp = """
 <div style ="background-color:lightseagreen;padding:17px">
 <h1 style ="color:black;text-align:center;">Customer Churn Predictor</h1>
 </div>
 """
st.markdown('##')
st.markdown(page_bg_img,unsafe_allow_html=True)
st.markdown(html_temp, unsafe_allow_html = True)


st.sidebar.header('User Input Parameters')

def user_input_features():
    Age  = st.sidebar.selectbox("Age of patient",(range(20,95,1)))
    st.sidebar.header('Select Gender 0 for female and 1 for male')
    Gender  = st.sidebar.selectbox("Gender",(0,1))
    st.sidebar.header('Select Location 0 for Chicago,1 for Houston,2 for Los Angeles,3 for Miami,4 for New York ')
    Location = st.sidebar.selectbox('Select the Location',(0,1,2,3,4))
    Subscription_Length_Months = st.sidebar.selectbox('Subscription Number of Months',(range(1,24,1)))
    Monthly_Bill = st.sidebar.slider('Monthly Bill Charges in $ dollars',30,100)
    Total_Usage_GB  = st.sidebar.slider("Total Usage in GB",50,500)
    
    data = {
            "Age":Age,
            "Gender":Gender,
            'Location':Location,
            'Subscription_Length_Months':Subscription_Length_Months,
            'Monthly_Bill':Monthly_Bill,
            'Total_Usage_GB':Total_Usage_GB 
            }
    features = pd.DataFrame(data,index = [0])
    return features 

df_1 = user_input_features()
st.subheader('User Input parameters')
st.write(df_1)

#Reading the Customer churn 
df=pd.read_excel('customer_churn_large_dataset.xlsx')
df=df.set_index('CustomerID')
#Droping the Name Variable
df=df.drop('Name',axis=1)

from sklearn.preprocessing import LabelEncoder

df['Gender']=LabelEncoder().fit_transform(df.Gender)
df['Location']=LabelEncoder().fit_transform(df.Location)

#train test split
x=df.iloc[:,:-1]
y=df.iloc[:,-1]

# Model Building

#builing Random Forest model
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=500,max_depth=15,min_samples_split=5)

RFC.fit(x,y)

st.write(RFC.predict(df_1))


if st.button("Predict"):   
    prediction = RFC.predict(df_1)
    st.write(f"<div style ='background-color:transparent;padding:2px'><h2 style ='color:cyan;text-align:left;'>Predicted Result of the Model is {prediction} class </h2> </div>",unsafe_allow_html = True)
    
    if prediction==0:
        st.write(f"<h2 style ='color:cyan;text-align:left;'> Customer Not Churn </h2>",unsafe_allow_html = True)
    elif prediction==1:
        st.write(f"<h2 style ='color:cyan;text-align:left;'> Customer Churn </h2>",unsafe_allow_html = True)

