from cmath import nan
from tkinter import W
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import base64

from dashboard_functions import *
from sections_of_app import *

# settings for streamlit page



# import the data contained in csv downloaded from Kaggle in a dataframe
df_diabetes = pd.read_csv('data/diabetes_dataset__2019.csv')


st.title('Diabetes Analysis')

# list of different sections of the app
list_app_sections = ['Data Exploration and Data Cleaning', 'Data Analysis']
select_section = st.sidebar.selectbox('Select a section you want to see', list_app_sections)

st.write(
    'This dashboard was created to present the analysis of some data with the goal to understand what are the main causes of diabetes.' + 
    'The data used here can be found on Kaggle, at the following link: \n _https://www.kaggle.com/tigganeha4/diabetes-dataset-2019_. ' +
    'The dataset about diabetes is composed by ' + str(df_diabetes.shape[0]) + ' rows and ' + str(df_diabetes.shape[1]) + ' columns.')


if select_section == 'Data Exploration and Data Cleaning':
    exploration_data_cleaning(df_diabetes)
if select_section == 'Data Analysis':
    data_analysis_section(df_diabetes)
