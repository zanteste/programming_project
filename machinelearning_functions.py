import pandas as pd
import numpy as np
import streamlit as st

# function to replace null values in the columns BMI, Pregnancies, Pdiabetes and Diabetic
def replacing_null_values(df_diabetes):
    # replace null values in the column BMI with the mode value
    BMI_mode_value = df_diabetes.BMI.mode()[0]
    df_diabetes.BMI = df_diabetes.BMI.fillna(BMI_mode_value)

    # replace null values in the column Pregnancies with the most frequent value
    Pregnancies_most_frequent_value = df_diabetes.Pregnancies.value_counts().index.to_list()[0]
    df_diabetes.Pregnancies = df_diabetes.Pregnancies.fillna(Pregnancies_most_frequent_value)

    # replace null values in the column Pdiabetes with the most frequent value
    Pdiabetes_most_frequent_value = df_diabetes.Pdiabetes.value_counts().index.to_list()[0]
    df_diabetes.Pdiabetes = df_diabetes.Pdiabetes.fillna(Pdiabetes_most_frequent_value)

    # remove the row that has nan value in the column Diabetic
    df_diabetes = df_diabetes[df_diabetes.Diabetic.notnull()]

    return df_diabetes

# function to clean errors in columns RegularMedicine, BPLevel, Pdiabetes and Diabetic
def errors_datacleaning(df_diabetes):
    # In the column RegularMedicine there is one field valued with 'o'. The column must have values 'yes/no'.
    # The error 'o' is replaced with 'no'
    df_diabetes.loc[df_diabetes.RegularMedicine == 'o', 'RegularMedicine'] = 'no'

    # The column BPLevel can have three type of value: high, normal and low. In some row the values have first capital letter.
    # All values are modified in order to have only lower case letters. In one row there is a space: the space must be removed
    df_diabetes.BPLevel = df_diabetes.BPLevel.str.lower()
    df_diabetes.BPLevel = df_diabetes.BPLevel.str.replace(' ', '')

    # The column Pdiabetes can have only two values: yes or no. There are a lot of rows with 'o' instead of 'No'.
    # In that rows the values 'o' are substitued with 'no'.
    df_diabetes.loc[df_diabetes.Pdiabetes == '0', 'Pdiabetes'] = 'no'

    # In the column Diabetic there is a row with ' no' instead of 'no'. So the space at the beginning must be removed.
    df_diabetes.Diabetic = df_diabetes.Diabetic.str.replace(' ', '')

    return df_diabetes

# a unique function for data cleaning is created for having a unique function to call in every page of the app
def dataset_datacleaning(df):
    df_cleaned = df.rename(columns={'Pregancies':'Pregnancies', 'UriationFreq':'UrinationFreq'})

    df_cleaned = replacing_null_values(df_cleaned)

    df_cleaned = errors_datacleaning(df_cleaned)

    # changing Pregnancies type from float to object
    df_cleaned['Pregnancies'] = df_cleaned['Pregnancies'].astype(str)

    return df_cleaned

# function to create describing text for every operation neeed before running the algorithm 
def text_operation_needed(operation):
    if operation == 'Feature scaling':
        st.write("**Feature scaling** is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.")
    if operation == 'One hot encoding':
        st.write("**One hot encoding** is a common way of preprocessing categorical features for machine learning models. Thysi type of encoding creates a new binary feature for each possible category and assigns a value of 1 to the feature of " +
                "each sample that corresponds to its ordinal category.")
    if operation == 'Creating a binary column':
        st.write("It's not useful to apply the one hot encoding to categorical features with only two possible values, since after the one hot encoding process we would have some useless columns. So, we substite the original column with " +
                "a binary one that contains only 1 or 0 values, for every categorical column with only two allowed values. ")
