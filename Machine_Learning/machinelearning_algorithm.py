""" In this page there is the code for the machine learning algorithm """
import pandas as pd
import streamlit as st

from Machine_Learning.machinelearning_functions import *

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
            
            st.write('Every column can have **0** or **1** as values.')

        if choosen_operation == 'Creating a binary column':
            column_binary = st.selectbox('Select a column to see how its values changed', features_binary_columns)
            display_old_values_new_values_binary_op(column_binary, df_diabetes, df_diabetes_for_ml)

    st.header('Machine Learning Algorithm Results')

    st.write('During data analysis, it has been shown that there are some features with a high correlation with the target feature *Diabetic*. For this reason, it has been decided to train machine learning algorithms considering all features at first and then only the features' +
             ' with that high correlation.')

    st.write('The features with the highest correlation with the *Diabetic* column are the following: *Age*, *Family_Diabetes*, *PDiabetes*, *Pregnancies*, *RegularMedicine*, *highBP*, *Stress* *BPLevel*. ')

    st.write('So, machine learning algorithms were trained considering: (1) all features of the original dataaset and (2) only features with highest correlation with the target column.')

    st.write('Three different classification algorithms were choosen:')
    st.markdown('- *Support Vector Machines for classification*: SVC in Scikit-Learn library.')
    st.markdown('- *Random Forest Classifier*: RandomForestClassifier in Scikit-Learn library.')
    st.markdown('- *K-nearest neighbors classifier*: KNeighborsClassifier in Scikit-Learn library.')

    # -----------------------  MACHINE LEARNING ALGORITHMS TRAINING ----------------------

    # two different datasets are needed: one with all feature (df_diabetes_for_ml) and an other one with only features with highest correlation
    # so next, the second dataframe is created
    features_with_highest_corr = ['Age', 'Family_Diabetes', 'PDiabetes', 'Pregnancies', 'RegularMedicine', 'highBP', 'Stress', 'BPLevel']

    # since data preprocessing operations (feature scaling, one hot encoding and creation of a binary column) changed the name of columns
    # we need to take the column from the full dataset (df_diabetes_for_ml) that contains the name of the features contained in the
    # list named 'features_with_highest_corr'
    columns_with_highest = []
    for col in features_with_highest_corr:
        for column_for_ml in df_diabetes_for_ml.columns.to_list():
            if col in column_for_ml:
                columns_with_highest.append(column_for_ml)
    
    # creation of the dataset containing only the columns referred to the features with highest correlation
    # the target feature 'Diabetic' is added to the list columns_with_highest
    df_highest = df_diabetes_for_ml[columns_with_highest + ['Diabetic']]

    # for the training of machine learning algorithm it's need to divide the dataset (df_highest or df_diabetes_for_ml) into two different datasets
    # one containing all features except the target one and one containing only the target feature

    # dividing full dataset (df_diabetes_for_ml)
    X_full = df_diabetes_for_ml.loc[:, df_diabetes_for_ml.columns != 'Diabetic']
    y_full = df_diabetes_for_ml['Diabetic']

    # dividing dataset with features with highest correlation (df_highest)
    X_highest = df_highest.loc[:, df_highest.columns != 'Diabetic']
    y_highest = df_highest['Diabetic']

    if st.button('Show results for Support Vector Classification'):
        # training with all features
        accuracy_svc_all, training_time_all, cols_indexes, imp_indexes = train_machine_learning_algorithm(X_full, y_full, 'support_vector', 'all')
        # training with only features with highest correlation
        accuracy_svc_highest, training_time_highest = train_machine_learning_algorithm(X_highest, y_highest,  'support_vector', 'highest')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Accuracy with all features: ')
            st.markdown('- ' + str(round(accuracy_svc_all,2)))
            st.write('Training time with all features: ')
            st.markdown('- ' + str(training_time_all))
        with col2:
            st.write('Accuracy with features with highest correlation: ')
            st.markdown('- ' + str(round(accuracy_svc_highest,2)))
            st.write('Training time with features with\ highest features: ')
            st.markdown('- ' + str(training_time_highest))

        st.write("In the graph below, it can be seen what are the features that better predict diabetes: ")
        plotting_rank_features(cols_indexes, imp_indexes)
    
    if st.button('Show results for K-nearest Neighbors'):
        # training with all features
        accuracy_svc_all, training_time_all, cols_indexes, imp_indexes = train_machine_learning_algorithm(X_full, y_full, 'k-nearest', 'all')
        # training with only features with highest correlation
        accuracy_svc_highest, training_time_highest = train_machine_learning_algorithm(X_highest, y_highest,  'k-nearest', 'highest')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Accuracy with all features: ')
            st.markdown('- ' + str(round(accuracy_svc_all,2)))
            st.write('Training time with all features: ')
            st.markdown('- ' + str(training_time_all))
        with col2:
            st.write('Accuracy with features with highest correlation: ')
            st.markdown('- ' + str(round(accuracy_svc_highest,2)))
            st.write('Training time with features with\ highest features: ')
            st.markdown('- ' + str(training_time_highest))

        st.write("In the graph below, it can be seen what are the features that better predict diabetes: ")
        plotting_rank_features(cols_indexes, imp_indexes)
    
    if st.button('Show results for Random Forest'):
        # training with all features
        accuracy_svc_all, training_time_all, cols_indexes, imp_indexes = train_machine_learning_algorithm(X_full, y_full, 'random', 'all')
        # training with only features with highest correlation
        accuracy_svc_highest, training_time_highest = train_machine_learning_algorithm(X_highest, y_highest,  'random', 'highest')
        col1, col2 = st.columns(2)
        with col1:
            st.write('Accuracy with all features: ')
            st.markdown('- ' + str(round(accuracy_svc_all,2)))
            st.write('Training time with all features: ')
            st.markdown('- ' + str(training_time_all))
        with col2:
            st.write('Accuracy with features with highest correlation: ')
            st.markdown('- ' + str(round(accuracy_svc_highest,2)))
            st.write('Training time with features with\ highest features: ')
            st.markdown('- ' + str(training_time_highest))

        st.write("In the graph below, it can be seen what are the features that better predict diabetes: ")
        plotting_rank_features(cols_indexes, imp_indexes)

    