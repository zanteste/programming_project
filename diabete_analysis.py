from cmath import nan
from tkinter import W
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from dashboard_functions import text_for_nan_cleaning

# settings for streamlit page
st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto")



# import the data contained in csv downloaded from Kaggle in a dataframe
df_diabetes = pd.read_csv('data/diabetes_dataset__2019.csv')


st.title('Diabetes Analysis')


st.write(
    'This dashboard was created to present the analysis of some data with the goal to understand what are the main causes of diabetes.' + 
    'The data used here can be found on Kaggle, at the following link: \n _https://www.kaggle.com/tigganeha4/diabetes-dataset-2019_. ' +
    'The dataset about diabetes is composed by ' + str(df_diabetes.shape[0]) + ' rows and ' + str(df_diabetes.shape[1]) + ' columns.')

# all dataset columns in a list
columns_list = df_diabetes.columns.to_list()
# all columns except the last one are merged in a string
all_columns_except_last = ', '.join(columns_list[:-1])

st.header('Data Exploration and Data Cleaning')

# in the following text (st.write) the name of each column is shown
st.write(
    'The columns of the diabetes dataset are the following: *' + all_columns_except_last + '* and *' + columns_list[len(columns_list)-1] + '*.' +
    "It's possible to see that two columns have some spelling errors. They have been corrected as follows: "
)
st.markdown("- *Pregancies* became *Pregnancies*; \n - *UriationFreq* became *UrinationFreq*.")
# renaming of Pregancies and UriationFreq
df_diabetes.rename(columns={'Pregancies':'Pregnancies', 'UriationFreq':'UrinationFreq'}, inplace=True)
# dataset columns in list after renaming the two columns before
columns_list = df_diabetes.columns.to_list()


# -------------------------- DATA CLEANING OF NULL VALUES ---------------------------

# list of the dataset columns with null values inside
columns_with_nan = df_diabetes.columns[df_diabetes.isna().any()].to_list()
# columns with nan to an unique str, in order to concatenate with the following text
string_columns_with_nan = ", ".join(columns_with_nan)
st.write("While browsing the data, it has been discovered that there were some null values in the original dataset." + 
" In particular the null values were in the following columns: " + string_columns_with_nan + ". " + 
" \n If you are interested in the cleaning of the null values click the box below.")


# Expander for getting information about how the null values have been treated
with st.beta_expander('Get info about cleaning of null values'):
    col1, col2 = st.beta_columns(2)
    with col1:
        choosen_column_with_nan = st.selectbox('', columns_with_nan)
    with col2:
        for nan_col in columns_with_nan:
            if nan_col == choosen_column_with_nan:
                number_of_nan_values= df_diabetes[nan_col].isnull().sum()
                st.write(' \n ' + str(number_of_nan_values) + ' null values in the selected column.')
                text_for_nan_cleaning(nan_col)
                if nan_col == 'BMI':
                    if st.button('Show ' + nan_col + ' distribution'):
                        fig, ax = plt.subplots(1,1, figsize=(2,2))
                        df_diabetes[nan_col].hist(ax=ax)
                        st.pyplot(fig)
                if (nan_col == 'Pregnancies') | (nan_col == 'Pdiabetes'):
                    most_frequent_value = df_diabetes[nan_col].mode()[0]
                    st.write('The most frequent value in the selected columns is ' + str(most_frequent_value) + ". So all the nan values have been filled with it.") 

# --------------------------------- DATA CLEANING OF ALL COLUMNS ------------------------------------

st.write('Once null values were replaced, it became necessary to correct some errors in the data. In particular, errors have been found in the following columns:' +
' *RegularMedicine*, *BPLevel*, *Pdiabetes* and *Diabetic*. If you want to discover what type of errors have been found and how they have been managed click the box below.')

# Expander for getting information about how errors have been handled

                


# dictionary contain the meaning of each columns
meaning_of_columns = {'Age':'Age of the participants', 'Gender': 'Sex of the participants', 
                        'Family_Diabetes':'It indicates if there were diabetes cases in the family of the participants.', 
                        'highBP':'It indicates if a participant was diagnosed with high blook pressure', 'PhysicallyActive': 'It indicates how much physical activity a participant does',
                       'BMI':'Body mass (fat) index of the participant', 'Smoking': "It denotes if a participant smokes or doesn't", 'Alcohol':'It indicates if a participant consumes alcohol or not', 
                       'Sleep':'It indicates how much a participant sleeps', 'Soundsleep':'It indicates the amount of hours of sound sleep', 
                       'RegularMedicine':'It indicates if a participant takes medicines regularly or not',
                       'JunkFood':'It indicates if a participant consumes a junk food or not','Stress':'It indicates the level of stress of the participants', 
                       'BPLevel':'Blood pressure level of participants', 'Pregnancies':'Number of pregnanancis for each participant', 
                       'Pdiabetes':'Gestation diabetes', 
                       'UrinationFreq': 'Urination frequency', 'Diabetic':'It indicates if a participant is diabetic or not'}


# creation of two columns in order to have the selectbox and the meaning of the selected column side by side

with st.beta_expander('Get info about columns'):
    col1, col2 = st.beta_columns(2)
    with col1:
        choosen_column = st.selectbox('',columns_list)
    with col2:
        for key in meaning_of_columns:
            if key == choosen_column:
                st.text("")
                st.write("*" + meaning_of_columns[key] + '*')
                if key == 'Pregancies':
                    list_of_possible_values = list(set(df_diabetes[key]))
                    st.write('This column is a numerical column.'+ 
                    ' Altough, the selected column can be considered as categorical since it can have only four values.')
                    st.write('So, the selected column can have the following values:')
                    for value in list_of_possible_values:
                        st.markdown("- " + str(value))
                if df_diabetes.dtypes[key] == np.object:
                    list_of_possible_values = list(set(df_diabetes[key]))
                    st.write('The selected column can have the following values:')
                    for value in list_of_possible_values:
                        st.markdown("- " + str(value))
                if df_diabetes.dtypes[key] != np.object:
                    minimum_value = df_diabetes[key].min()
                    maximum_value = df_diabetes[key].max()
                    medium_value = round(df_diabetes[key].mean(),2)
                    st.write('The selected column contains numeric values with the following properties:')
                    st.markdown("- Minimum value: " + str(minimum_value))
                    st.markdown("- Maximum value: " + str(maximum_value))
                    st.markdown("- Medium value: " + str(medium_value))
                

