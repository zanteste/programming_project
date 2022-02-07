""" In this page there is the code for the machine learning algorithm """
import pandas as pd
import streamlit as st

from machinelearning_functions import *


original_dataset = pd.read_csv('data/diabetes_dataset__2019.csv')
# applying of data cleaning functions to the original dataset
df_diabetes = dataset_datacleaning(original_dataset)

def app():
    st.title('Diabetes Prediction')