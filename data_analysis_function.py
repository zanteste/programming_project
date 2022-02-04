import pandas as pd 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# function to define a sequence of colors for plots
def color_sequence_for_graph(col):
    if col == 'Age':
        colors_based_on_value = {'less than 40':'#82E0AA', '40-49':'#2ECC71', '50-59':'#239B56', '60 or older':'#186A3B'}
    if col == 'Gender':
        colors_based_on_value = {'Male':'#AED6F1', 'Female':'#C39BD3'}
    if col in ['Family_Diabetes', 'Pdiabetes', 'Diabetic']:
        colors_based_on_value = {'no': '#E74C3C', 'yes': '#2ECC71'}
    if col in ['Alcohol', 'Smoking', 'highBP', 'RegularMedicine']:
        colors_based_on_value = {'no': '#2ECC71', 'yes': '#E74C3C'}
    if col == 'PhysicallyActive':
        colors_based_on_value = {'none':'#E74C3C', 'less than half an hr': '#E67E22', 'more than half an hr':'#F1C40F', 'one hr or more':'#2ECC71'}
    if col == 'JunkFood':
        colors_based_on_value = {'occasionally':'#2ECC71', 'often':'#F1C40F', 'very often':'#F39C12', 'always':'#E74C3C'}
    if col == 'Stress':
        colors_based_on_value = {'not at all':'#2ECC71', 'sometimes':'#F1C40F', 'very often':'#F39C12', 'always':'#E74C3C'}
    if col == 'BPLevel':
        colors_based_on_value = {'normal':'#2ECC71', 'low':'#F1C40F', 'high':'#E74C3C'}
    if col == 'Pregnancies':
        colors_based_on_value = {0.0:'#85C1E9', 1.0:'#5DADE2', 2.0:'#3498DB', 3.0:'#2E86C1', 4.0:'#2874A6'}
    if col == 'UrinationFreq':
        colors_based_on_value = {'quite often':'#2ECC71', 'not much':'#E74C3C'}
    return colors_based_on_value

# function to create a ordered list for the 'categorical' values of a column
def list_categorical_order(col):
    if col == 'Age':
        list_order_values = ['less than 40', '40-49', '50-59', '60 or older']
    if col == 'Gender':
        list_order_values = ['Female', 'Male']
    if col in ['Family_Diabetes', 'highBP', 'Smoking', 'Alcohol', 'RegularMedicine', 'Pdiabetes', 'Diabetic']:
        list_order_values = ['no', 'yes']
    if col == 'PhysicallyActive':
        list_order_values  = ['none', 'less than half an hr', 'more than half an hr', 'one hr or more']
    if col == 'JunkFood':
        list_order_values = ['occasionally', 'often', 'very often', 'always']
    if col == 'Stress':
        list_order_values = ['not at all', 'sometimes', 'very often', 'always']
    if col == 'BPLevel':
        list_order_values = ['low', 'normal', 'high']
    if col == 'UrinationFreq':
        list_order_values = ['not much', 'quite often']
    return list_order_values

def custom_order_based_on_values_one_col(col, df):
    list_values_in_order = list_categorical_order(col)
    
    # ordering dataframe based on the list in which the values are in the requested order
    df[col] = pd.Categorical(df[col], list_values_in_order)
    df = df.sort_values(by=col)
    return df

# function to order a dataframe based on categorical order of two differente columns
def custom_order_based_on_values_two_col(col1, col2, df):
    list_values_in_order_col1 = list_categorical_order(col1)
    list_values_in_order_col2 = list_categorical_order(col2)

    df[col1] = pd.Categorical(df[col1], list_values_in_order_col1)
    df[col2] = pd.Categorical(df[col2], list_values_in_order_col2)

    df = df.sort_values(by=[col1, col2])
    return df

# function to create distribution for each column of the dataset
def create_distribution_plot(col, df):
    #if (df[col].dtype == 'object') | (col == 'Pregnancies'):
    if (col not in ['BMI', 'Sleep', 'SoundSleep']):
        df_values_in_selected_col = df[col].value_counts().to_frame().reset_index()
        df_values_in_selected_col.columns = [col, 'Number of Participants']
        # ordering the dataframe just created according to the list of the requested order
        df_values_in_selected_col = custom_order_based_on_values_one_col(col, df_values_in_selected_col)
        fig = plt.figure(figsize=(10,6))
        # use colors defined in the function 'color_sequence_for_graph'
        colors_for_value = color_sequence_for_graph(col)
        graph = sns.barplot(x=df_values_in_selected_col[col], y=df_values_in_selected_col['Number of Participants'], palette=colors_for_value)
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