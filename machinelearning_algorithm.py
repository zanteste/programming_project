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

    
    st.header('Data preprocessing')
    
    features_to_be_scaled,features_one_hot,features_binary_columns = list_of_columns_for_every_data_preprocessing_operation(df_diabetes)

    st.write("Some data operations are required before running the algorithm, that are: ")
    st.markdown("- **feature scaling** for numeric features. This operation has been applied to the following features: *" + ', '.join(features_to_be_scaled) + '*.')
    st.markdown("- **one hot encoding** for every categorical feature with more than two values. This operaton has been applied to the following featurs: *" + ', '.join(features_one_hot) + '*.')
    st.markdown("- **creating a binary column** for every categorical feature with only two values. This operaton has been applied to the following featurs: *" + ', '.join(features_binary_columns) + '*.')

    # applyng the need preprocessing operations
    # the new dataframe is used in the machine learning algorithms
    df_copy = df_diabetes.copy()
    df_diabetes_for_ml = preprocessing_data_operations(df_copy)

    #creation of a list for every operations needed before running the machine learning algorithm
    list_of_operations = ['Feature scaling', 'One hot encoding', 'Creating a binary column']
    
    # expander to get informations about every operations needed before running the machine learning algorithm
    with st.expander('Get info about every operations needed before running the algorithm and about the operation results'):
        choosen_operation = st.selectbox('', list_of_operations)
        text_operation_needed(choosen_operation)
        if choosen_operation == 'Feature scaling':
            col1, col2 = st.columns(2)
            with col1:
                st.write('**Original Dataset**')
                st.table(df_diabetes[features_to_be_scaled].head())
            with col2:
                st.write('**Dataset after Data Scaling**')
                st.table(df_diabetes_for_ml[features_to_be_scaled].head())
        if choosen_operation == 'One hot encoding':
            column_one_hot_encoded = st.selectbox('Select a colum to see the results of one hot encoding', features_one_hot)
            st.write('After one hot encoding operation the following columns were obtained: ')
            # getting list of columns in df_diabetes_for_ml (after data preprocessing operations)
            list_of_columns = df_diabetes_for_ml.columns.to_list()
            # Print the names of columns obtained after one hot encoding
            for col in list_of_columns:
                if column_one_hot_encoded in col:

                    st.markdown('- *'+col+'*')
            
            st.write('Every column can have **0** or **1** as values: the value *1* indicates that a participant age is in the range indicated in the name of the new column.')
        if choosen_operation == 'Creating a binary column':
            column_binary = st.selectbox('Select a column to see how its values changed', features_binary_columns)
            display_old_values_new_values_binary_op(column_binary, df_diabetes, df_diabetes_for_ml)

    st.header('Machine Learning Algorithm Results')

    