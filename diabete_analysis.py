from cmath import nan
from tkinter import W
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import base64

from dashboard_functions import datacleaning, replacing_null_values, text_for_nan_cleaning

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

# create copy of df_diabetes before replacing null values in order to show info about nan values
df_diabets_with_nan = df_diabetes.copy()
# replacing null values in the original dataset df_diabetes
df_diabetes = replacing_null_values(df_diabetes)

# Expander for getting information about how the null values have been treated
with st.beta_expander('Get info about cleaning of null values'):
    col1, col2 = st.beta_columns(2)
    with col1:
        choosen_column_with_nan = st.selectbox('', columns_with_nan)
    with col2:
        for nan_col in columns_with_nan:
            if nan_col == choosen_column_with_nan:
                number_of_nan_values= df_diabets_with_nan[nan_col].isnull().sum()
                st.write(' \n ' + str(number_of_nan_values) + ' null values in the selected column.')
                text_for_nan_cleaning(nan_col)
                if nan_col == 'BMI':
                    BMI_mode_value = df_diabetes[nan_col].mode()[0]
                    st.write('The mode value of BMI column is: ' + str(BMI_mode_value) + '. So the null values have been replaced with that value.')
                    if st.button('Show ' + nan_col + ' distribution'):
                        fig, ax = plt.subplots(1,1, figsize=(2,2))
                        df_diabets_with_nan[nan_col].hist(ax=ax)
                        st.pyplot(fig)
                if (nan_col == 'Pregnancies') | (nan_col == 'Pdiabetes'):
                    most_frequent_value = df_diabets_with_nan[nan_col].value_counts().index.to_list()[0]
                    st.write('The most frequent value in the selected columns is ' + str(most_frequent_value) + ". So all the nan values have been filled with it.") 

# --------------------------------- DATA CLEANING OF ALL COLUMNS ------------------------------------

st.write('Once null values were replaced, it became necessary to correct some errors in the data. In particular, errors have been found in the following columns:' +
' *RegularMedicine*, *BPLevel*, *Pdiabetes* and *Diabetic*. If you want to discover what type of errors have been found and how they have been managed click the box below.')
st.write("All changes to the original dataset have been made according to  what is defined in the pdf visible by clicking on 'Show pdf' (at page 6 (711 of the original document) and 7 (712))." +
        " This pdf was written by *Neha Prerna Tigga* and *Shruti Garg*.")

if st.checkbox("Show PDF Document"):
        def show_pdf(file_path):
            with open(file_path,"rb") as f:
                  base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)
        st.write(show_pdf("1-s2.0-S1877050920308024-main.pdf"))
# create copy of df_diabetes before data cleaning in order to show why data cleaning is necessary
df_diabetes_no_datacleaning = df_diabetes.copy()
# data cleaning
df_diabetes = datacleaning(df_diabetes)


# list that contains the name of columns for which data cleaning is necessary
columns_for_data_cleaning = ['RegularMedicine', 'BPLevel', 'Pdiabetes', 'Diabetic']

# Expander for getting information about how errors have been handled (Data Cleaning)
with st.beta_expander('Get info about how errors have been handled'):
    choosen_column = st.selectbox('', columns_for_data_cleaning)
    list_of_original_values = df_diabetes_no_datacleaning[choosen_column].value_counts().index.to_list()
    list_of_values_after_data_cleaning = df_diabetes[choosen_column].value_counts().index.to_list()
    st.write('In the original dataset, the *' + choosen_column + '* had the following values: ')
    for val in list_of_original_values:
        st.markdown('- ' + str(val))
    if choosen_column == 'RegularMedicine':
        st.write("It's possible to see that there were some rows with value *'o'* instead of *'no'*. " + 
                "So, data cleaning operations for the selected column had the goal to substitute the *'o'* values with the expected value.")
    if choosen_column == 'BPLevel':
        st.write("It's possible to see that there were some rows with lowercase first letter and others with uppercase first letter." +
                'Since the original data (in the PDF above) are with lowercase first letter, data cleaning operations have modified the fields with uppercase first letter.' +
                ' Furthermore, there is a row in which the *' + choosen_column + '* column is enhanced with a space inside: that space has been removed.')
    if choosen_column == 'Pdiabetes':
        st.write('Accordin to the pdf, *' + choosen_column + "* column can only have two values: *'yes'* or 'no'. From the list above,"+ 
                " it's possible to see that in the original dataset there is an unwanted value: '0'. That value has been substituted with 'no'," + 
                "as '0' can be seen as the case in which a person has not diabetes gestation.")
    if choosen_column == 'Diabetic':
        st.write("It's possible to see that there are two kinds of 'no' in the original data: the reason is that there is a row with a space in it." +
                " That space has been removed, during data cleaning operations.")
    st.write('Once the data have been cleaned, the column *' + choosen_column + '* has the following values, as described in the pdf document:' )
    for val in list_of_values_after_data_cleaning:
        st.markdown('- ' + str(val))



                


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
                

