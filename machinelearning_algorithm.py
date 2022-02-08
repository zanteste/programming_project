""" In this page there is the code for the machine learning algorithm """
import pandas as pd
import streamlit as st

from machinelearning_functions import *


original_dataset = pd.read_csv('data/diabetes_dataset__2019.csv')
# applying of data cleaning functions to the original dataset
df_diabetes = dataset_datacleaning(original_dataset)

def app():
    st.title('Diabetes Prediction')

    st.write("The goal of the project presented here is to predict whether or not a person has diabetes: so, it's necessary to implement a machine learning algorithm." + 
            " As described in the previous sections, the *target feature* is the *Diabetic* column: since this feature can only have two values, no (not diabetic) and yes (diabetic), it is a *classification problem*.")

    st.write("Some data operations are required before running the algorithm, that are: ")
    st.markdown("- **feature scaling** for numeric features;")
    st.markdown("- **one hot encoding** for every categorical feature with more than two values;")
    st.markdown("- **creating a binary column** for every categorical feature with only two values.")

    #creation of a list for every operations needed before running the machine learning algorithm
    list_of_operations = ['Feature scaling', 'One hot encoding', 'Creating a binary column']

    # expander to get informations about every operations needed before running the machine learning algorithm
    with st.expander('Get info about every operations needed before running the algorithm'):
        choosen_operation = st.selectbox('', list_of_operations)
        text_operation_needed(choosen_operation)
    
    st.header('Data preprocessing')
    
    # creating a copy of original dataset. This copy is used in the machines learning algorithms
    df_diabetes_for_ml = df_diabetes.copy()

    