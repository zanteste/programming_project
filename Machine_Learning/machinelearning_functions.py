import pandas as pd
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as sp


from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC # support vector machine classifier
from sklearn.neighbors import KNeighborsClassifier as KNC 
from sklearn.ensemble import RandomForestClassifier as RF

import time


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
        st.write("Below, it's shown how data have changed after *feature scaling* (the first 5 rows of the datasets):")
    if operation == 'One hot encoding':
        st.write("**One hot encoding** is a common way of preprocessing categorical features for machine learning models. Thysi type of encoding creates a new binary feature for each possible category and assigns a value of 1 to the feature of " +
                "each sample that corresponds to its ordinal category.")
    if operation == 'Creating a binary column':
        st.write("It's not useful to apply the one hot encoding to categorical features with only two possible values, since after the one hot encoding process we would have some useless columns. So, we substite the original column with " +
                "a binary one that contains only 1 or 0 values, for every categorical column with only two allowed values. ")

# function to get list of columns for each data preprocessing operations: data scaling, one hot encoding and creting a binary column
def list_of_columns_for_every_data_preprocessing_operation(df):
    # data scaling
    numerics = ['int64', 'float64']
    features_to_be_scaled = df.select_dtypes(include=numerics).columns.to_list()

    # finding columns that need the one hot encoding (only features with more than 2 values)
    # and finding columns for which we only need to subsitute values with binary ones (0 and 1) --> only features with 2 values

    # dropping the numeric features in order to have only categorical ones
    df_object_columns = df.drop(columns=features_to_be_scaled)
    df_values_for_features = df_object_columns.nunique().to_frame().reset_index()
    df_values_for_features.columns = ['Columns', 'N_values']

    # features that need the one hot encoding
    features_one_hot = df_values_for_features[df_values_for_features.N_values > 2]['Columns'].to_list()
    # features in whihc substitute values with binary ones
    features_binary_columns = df_values_for_features[df_values_for_features.N_values == 2]['Columns'].to_list()

    return features_to_be_scaled,features_one_hot,features_binary_columns

# functions to apply the need operations in data preprocessing: data scaling, one hot encoding and creating a binary column
def preprocessing_data_operations(df):
    
    features_to_be_scaled,features_one_hot,features_binary_columns = list_of_columns_for_every_data_preprocessing_operation(df)
    # data scaling
    scaler = StandardScaler()

    df[features_to_be_scaled] = scaler.fit_transform(df[features_to_be_scaled])
    # applying one hot encoding to columns in features_one_hot
    # use drop_first to avoid the creation of an unuseful column
    df = pd.get_dummies(data=df, columns=features_one_hot, drop_first=False)

    # substituting values with binary ones for columns in features_binary_columns
    binary_values_for_features = {'Gender': {'Male':0, 'Female':1}, 'Family_Diabetes':{'no':0, 'yes':1}, 'highBP':{'no':0, 'yes':1}, 
                                'Smoking':{'no':0, 'yes':1}, 'Alcohol':{'no':0, 'yes':1}, 'RegularMedicine':{'no':0, 'yes':1}, 
                                'Pdiabetes':{'no':0, 'yes':1}, 'UrinationFreq':{'not much':0, 'quite often':1}, 'Diabetic':{'no':0, 'yes':1}}
    
    df.replace(binary_values_for_features, inplace=True)

    return df

# functions to display old values and new values after substituting the values with binary ones
def display_old_values_new_values_binary_op(column_binary, df_original, df_for_ml):
    
    # CSS to hide row index in st.table
    hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
        
    col1, col2 = st.columns(2)
    with col1:
        original_values = df_original[column_binary].value_counts().to_frame().reset_index()
        original_values.columns = [column_binary, 'N']
        original_values.drop(columns='N', inplace=True)
        # applyng the css
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(original_values)
    with col2: 
        new_values = df_for_ml[column_binary].value_counts().to_frame().reset_index()
        new_values.columns = [column_binary, 'N']
        new_values.drop(columns='N', inplace=True)
        # applying the css
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        st.table(new_values)

# function to get rank of fatures in machine learning algorithm
def rank_features(model, X_train, y_train):
    importances = permutation_importance(model, X_train, y_train, scoring='neg_mean_squared_error', n_repeats=10, 
                       random_state=0)
    indexes = importances.importances_mean.argsort()
    imp_indexes = importances.importances_mean[indexes]

    cols_indexes = X_train.columns[indexes]

    return cols_indexes, imp_indexes

# function to train machine learning algorithms
# df_type is a string to indicate which dataset is used for training: all features or only that one with features with highest correlation
def train_machine_learning_algorithm(X, y, algorithm, df_type):
    if algorithm == 'support_vector':
        model = SVC()
    if algorithm == 'k-nearest':
        model = KNC()
    if algorithm == 'random':
        model = RF()
    
    X_train, X_test, y_train, y_test = sp(X, y, test_size=0.20, random_state=0)

    start = time.time()
    model.fit(X_train, y_train)
    stop = time.time()

    training_time = stop - start

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # ranking indexes (cols_indexes and imp_indexes) are obtained only if we train machine learning algorithm with all features
    if df_type == 'all':
        # get indexes for ranking the features
        cols_indexes, imp_indexes = rank_features(model, X_train, y_train)
        # returning the accuracy and the indexes for ranking the features
        return accuracy, training_time, cols_indexes, imp_indexes
    else:
        return accuracy, training_time
    

# function to plot features rank
def plotting_rank_features(cols_indexes, imp_indexes):
    figure = plt.figure(figsize=(10,10))
    plt.barh(cols_indexes, imp_indexes)
    plt.xlabel('Importanza delle features') 

    st.pyplot(figure)
    
figure = plt.figure(figsize=(10,10))

#plt.bar_label(graph, size=12, padding=5, labels=['{:.2}'.format(a) for a in imp_indexes])


