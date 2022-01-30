import pandas as pd 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# functions to describe how the null values were replaced in the 4 columns that presented that type of value
def text_for_nan_cleaning(nan_column):
    if nan_column == 'BMI':
        return st.write('Since the distribution of the values of the column called ' + nan_column + 
                    ' is not a normal distribution, the null values were substituted with the mode value of the column.')
    if (nan_column == 'Pregnancies') :
        return st.write('The selected column can be considered as a categorical column with 5 values(from 0 to 4),' + 
                    'the null values were substitued with the most frequent value.')
    if nan_column == 'Diabetic':
        return st.write('Since, one goal of the project presented here is to predict if a person has diabetes or not, the selected column is the target of the project. ' + 
                    'So, the row that presented the only null value of the ' + nan_column + ' column has been removed from the dataset.')
    if nan_column == 'Pdiabetes':
        return st.write('The selected column is a categorical variable, so we can substitute the null values with its most frequent value.')
       
# function to clean errors in columns RegularMedicine, BPLevel, Pdiabetes and Diabetic
def datacleaning(df_diabetes):
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

def text_for_description_of_columns(key, df_diabetes):
    if key == 'Pregancies':
        list_of_possible_values = list(set(df_diabetes[key]))
        st.write('This column is a numerical column.'+ 
        ' Altough, the selected column can be considered as categorical since it can have only four values.')
        st.write('So, the selected column can have the following values:')
        for value in list_of_possible_values:
            st.markdown("- " + str(value))
    if (df_diabetes.dtypes[key] == np.object) | (key == 'Pregnancies'):
        list_of_possible_values = list(set(df_diabetes[key]))
        st.write('The selected column can have the following values:')
        for value in list_of_possible_values:
            st.markdown("- " + str(value))
    elif df_diabetes.dtypes[key] != np.object:
        minimum_value = df_diabetes[key].min()
        maximum_value = df_diabetes[key].max()
        medium_value = round(df_diabetes[key].mean(),2)
        st.write('The selected column contains numeric values with the following properties:')
        st.markdown("- Minimum value: " + str(minimum_value))
        st.markdown("- Maximum value: " + str(maximum_value))
        st.markdown("- Medium value: " + str(medium_value))

# function to obtain list of all column types in the dataset
def get_list_of_columns_types(df_diabetes):
    cols_list = df_diabetes.columns.to_list()
    types_of_col_in_dataframe = []
    for col in cols_list:
        if df_diabetes[col].dtype == 'object' and 'object' not in types_of_col_in_dataframe:
            types_of_col_in_dataframe.append('object')
        if df_diabetes[col].dtype == 'float64' and 'float64' not in types_of_col_in_dataframe:
            types_of_col_in_dataframe.append('float64')
        if df_diabetes[col].dtype == 'int64' and 'int64' not in types_of_col_in_dataframe:
            types_of_col_in_dataframe.append('int64')
    
    return types_of_col_in_dataframe

# function to create distribution for each column of the dataset
def create_distribution_plot(col, df):
    if (df[col].dtype == 'object') | (col == 'Pregnancies'):
        df_values_in_selected_col = df[col].value_counts().to_frame().reset_index()
        df_values_in_selected_col.columns = [col, 'Number of Participants']
        fig = plt.figure(figsize=(10,6))
        graph = sns.barplot(x=df_values_in_selected_col[col], y=df_values_in_selected_col['Number of Participants'], palette='Greens')
        for p in graph.patches:
            graph.annotate(format(p.get_height(), '.0f'),
                        (p.get_x()+p.get_width() /2., p.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0,9),
                        textcoords = 'offset points')
        graph.set_title(col + ': participants distribution', size=12)
        graph.set_xlabel('')
        graph.set_ylabel('Number of participants', size=12)
        st.pyplot(fig)
    else:
        number_of_values_in_col = len(df[col].value_counts().index.to_list())
        fig = plt.figure(figsize=(10,6))
        sns.distplot(df[col], color='green', bins=number_of_values_in_col)
        st.pyplot(fig)