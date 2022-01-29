import pandas as pd 
import streamlit as st

def text_for_nan_cleaning(nan_column):
    if nan_column == 'BMI':
        return st.write('Since the distribution of the values of the column called ' + nan_column + 
                    ' is not a normal distribution, the null values were substituted with the mode value of the column.')
    if (nan_column == 'Pregnancies') | (nan_column == 'Pdiabetes'):
        return st.write('Since, the selected column can be considered as a categorical column with 5 (from 0 to 4) values,' + 
                    'the null values were substitued with the most frequent value. Before doing that, it was necessary to change the type of the column.')
    if nan_column == 'Diabetic':
        return st.write('Since, one goal of the project presented here is to predict if a person has diabetes or not, the selected column is the target of the project. ' + 
                    'So, the row that presented the only null value of the ' + nan_column + ' column has been removed from the dataset.')

        
    

    
