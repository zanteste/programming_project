import pandas as pd
import numpy as np
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
meaning_of_columns = {'Age':'Age of the participants', 'Gender': 'Sex of the participants', 
                        'Family_Diabetes':'It indicates if there were diabetes cases in the family of the participants.', 
                        'highBP':'It indicates if a participant was diagnosed with high blook pressure', 'PhysicallyActive': 'It indicates how much physical activity a participant does',
                       'BMI':'Body mass (fat) index of the participant', 'Smoking': "It denotes if a participant smokes or doesn't", 'Alcohol':'It indicates if a participant consumes alcohol or not', 
                       'Sleep':'It indicates how much a participant sleeps', 'Soundsleep':'It indicates the amount of hours of sound sleep', 
                       'RegularMedicine':'It indicates if a participant takes medicines regularly or not',
                       'JunkFood':'It indicates if a participant consumes a junk food or not','Stress':'It indicates the level of stress of the participants', 
                       'BPLevel':'Blood pressure level of participants', 'Pregancies':'Number of pregnanancis for each participant', 
                       'Pdiabetes':'Gestation diabetes', 
                       'UriationFreq': 'Frequency of urination', 'Diabetic':'It indicates if a participant is diabetic or not'}

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
                

