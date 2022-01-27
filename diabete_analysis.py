from numpy import number
import pandas as pd
import streamlit as st

# settings for streamlit page
st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto")



# import the data contained in csv downloaded from Kaggle in a dataframe
df_diabetes = pd.read_csv('data/diabetes_dataset__2019.csv')


st.title('Diabetes Analysis')


st.write(
    'This dashboard was created to present the analysis of some data with the goal to understand what are the main causes of diabetes.' + 
    'The data used here can be found on Kaggle, at the following link: \n _https://www.kaggle.com/tigganeha4/diabetes-dataset-2019_. ' +
    'The dataset about diabetes is composed by ' + str(df_diabetes.shape[0]) + ' rows and ' + str(df_diabetes.shape[1]) + ' columns.')

# all dataset columns are saved into a list
columns_list = columns_list = df_diabetes.columns.to_list()

# all columns except the last one are merged in a string
all_columns_except_last = ', '.join(columns_list[:-1])

# in the following text (st.write) the name of each column is shown
st.write(
    'The columns of the diabetes dataset are the following: *' + all_columns_except_last + '* and *' + columns_list[len(columns_list)-1] + '*.'
)


# dictionary contain the meaning of each columns
meaning_of_columns = {'Age':'Age of the people', 'Gender': 'Sex of the people', 'Family_Diabates':'', 'highBP':'', 'PhysicallyActive': 'It indicates how much physical activity a person does',
                       'BMI':'Body mass (fat) index', 'Smoking': 'It denotes if a person is a smoker or not', 'Alcohol':'', 'Sleep':'', 'Soundsleep':'', 'RegularMedicine':'',
                       'JunkFood':'','Stress':'', 'BPLevel':'', 'Pregancies':'', 'Pdiabetes':'', 'UriationFreq': '', 'Diabetic':''}

# creation of two columns in order to have the selectbox and the meaning of the selected column side by side
col1, col2 = st.beta_columns(2)
with col1:
    choosen_column = st.selectbox('',columns_list)
with col2:
    for key in meaning_of_columns:
        if key == choosen_column:
            st.text("")
            st.write("*" + meaning_of_columns[key] + '*')
