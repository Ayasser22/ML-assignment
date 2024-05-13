import streamlit as st
import requests
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit_lottie as st_lottie
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import PIL as Image 



st.set_page_config(
    page_title='lung cancer',
    page_icon=':gem:',
    initial_sidebar_state='collapsed'
)

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

model=joblib.load(open("lungcancer",'rb'))

def predict(age, gender, smoking, alcohol_use):
    features = np.array([age, gender, smoking, alcohol_use] + [0]*20).reshape(1, -1)
    prediction = model.predict(features)
    return prediction



with st.sidebar:
    choose = option_menu( None, ["Home", "Graphs", "About", "Contact"],
        icons=["house", "kanban", "book", "person_lines"],
        menu_icon="app-indicator",default_index=0,
        styles={
            "container": {"padding": "5px !important", "background-color": "#fafafa"},
            "icon": {"color": "#EEEFFF", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#02ab21"}
        }
    )


if choose == "Home":
    st.write('# Lung cancer prediction')
    st.subheader('Enter your details to lung cancer prediction')
    # User input
    age= st.number_input("Enter your age:", min_value=0)
    gender= st.radio("Select your gender:", ('male', 'female'))
    gender_encoder = 1 if gender == 'male' else 0
    smoking= st.number_input("Enter your smoking times per day:", min_value=0)
    alcohol_use= st.number_input("Enter your alcohol use per day:", min_value=0)
    #predict
    sample=predict(age,gender_encoder,smoking,alcohol_use)

    if st.button("predict"):
        if sample <= 4:
            st.write('This indicates a low level of lung cancer.')
    else:
        st.write('This indicates a high level of lung cancer.')


elif choose == "Graphs":
    st.write("# Graphs")
    st.image("download.png")

elif choose == "About":
    st.write("# About")
    st.write("This app provides a lung cancer prediction based on user input.")

elif choose == "Contact":
    st.write("# Contact")
    st.write("For inquiries, please contact us at a.yasser2160@nu.edu.eg")


data=pd.read_csv('cancer patient data sets.csv')
