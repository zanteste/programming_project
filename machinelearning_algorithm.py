""" In this page there is the code for the machine learning algorithm """
import pandas as pd
import streamlit as st

from machinelearning_functions import *
from sklearn.metrics import accuracy_score

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
            
            st.write('Every column can have **0** or **1** as values: the value *1* indicates that a participant age is in the range indicated in the name of the new column.')
        if choosen_operation == 'Creating a binary column':
            column_binary = st.selectbox('Select a column to see how its values changed', features_binary_columns)
            display_old_values_new_values_binary_op(column_binary, df_diabetes, df_diabetes_for_ml)

    st.header('Machine Learning Algorithm Results')

    st.write('During data analysis, it has been shown that there are some features with a high correlation with the target feature *Diabetic*. For this reason, it has been decided to train machine learning algorithms considering all features at first and then only the features' +
             ' with that high correlation.')

    st.write('The features with the highest correlation with the *Diabetic* column are the following: *Age*, *Family_Diabetes*, *PDiabetes*, *Pregnancies*, *RegularMedicine*, *highBP*, *Stress* *BPLevel*. In addition, there are two more features with a good correlation: *PhysicallyActive* and *Alcohol*.')

    st.write('So, machine learning algorithmss were trained considering: (1) all features, (2) only features with highest correlation features and (3) features with highest correlation plus the two with a good correlation.')

    st.write('Three different classification algorithms were choosen:')
    st.markdown('- *Support Vector Machines for classification*: SVC in Scikit-Learn library.')
    st.markdown('- *Random Forest Classifier*: RandomForestClassifier in Scikit-Learn library.')
    st.markdown('- *K-nearest neighbors classifier*: KNeighborsClassifier in Scikit-Learn library.')

    # -----------------------  MACHINE LEARNING ALGORITHM TRAINED WITH ALL FEATURES ----------------------
        # separating all features from the target one
    X = df_diabetes_for_ml.loc[:, df_diabetes_for_ml.columns != 'Diabetic']
    y = df_diabetes_for_ml['Diabetic']
        # splitting X and y in train and test df
    X_train, X_test, y_train, y_test = sp(X,y, test_size=0.20, random_state=0)

    # Support Vector Machine Classifier
    model_svc = SVC()
    model_svc.fit(X_train, y_train)
    y_pred_svc = model_svc.predict(X_test)

    accuracy_svc_all = accuracy_score(y_test, y_pred_svc)

    # K-nearest Neighbors 
    model_knc = KNC()
    model_knc.fit(X_train, y_train)
    y_pred_knc = model_knc.predict(X_test)

    accuracy_knc_all = accuracy_score(y_test, y_pred_knc)

    # Random Forest
    model_rf = RF()
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)

    accuracy_rf_all = accuracy_score(y_test, y_pred_rf)



    # ----------------------- FINDING BEST PARAMS FOR MACHINE LEARNING ALGORITHM TRAINED WITH FEATURES WITH HIGHEST CORR ----------------------
    features_with_highest_corr = ['Age', 'Family_Diabetes', 'PDiabetes', 'Pregnancies', 'RegularMedicine', 'highBP', 'Stress', 'BPLevel']
    columns_with_highest = []
    for col in features_with_highest_corr:
        for column_all in df_diabetes_for_ml.columns.to_list():
            if col in column_all:
                columns_with_highest.append(column_all)

        # dataframe with highest features plus diabetic
    df_highest = df_diabetes_for_ml[columns_with_highest + ['Diabetic']]
        # separating all features from the target one
    X = df_highest.loc[:, df_highest.columns != 'Diabetic']
    y = df_highest['Diabetic']
        # splitting X and y in train and test df
    X_train, X_test, y_train, y_test = sp(X,y, test_size=0.20, random_state=0)

     # Support Vector Machine Classifier
    model_svc = SVC()
    model_svc.fit(X_train, y_train)
    y_pred_svc = model_svc.predict(X_test)

    accuracy_svc_highest = accuracy_score(y_test, y_pred_svc)

    # K-nearest Neighbors 
    model_knc = KNC()
    model_knc.fit(X_train, y_train)
    y_pred_knc = model_knc.predict(X_test)

    accuracy_knc_highest = accuracy_score(y_test, y_pred_knc)

    # Random Forest
    model_rf = RF()
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)

    accuracy_rf_highest = accuracy_score(y_test, y_pred_rf)

    # ----------- FINDING BEST PARAMS FOR MACHINE LEARNING ALGORITHM TRAINED WITH FEATURES WITH HIGHEST CORR plus FEATURES WITH GOOD CORR ------
    features_with_good_corr = ['Age', 'Family_Diabetes', 'PDiabetes', 'Pregnancies', 'RegularMedicine', 'highBP', 'Stress', 'BPLevel', 'PhysicallyActive', 'Alcohol']
    columns_with_good = []
    for col in features_with_good_corr:
        for column_all in df_diabetes_for_ml.columns.to_list():
            if col in column_all:
                columns_with_good.append(column_all)
        # dataframe with highest features plus diabetic
    df_good = df_diabetes_for_ml[columns_with_good + ['Diabetic']]
        # separating all features from the target one
    X = df_good.loc[:, df_good.columns != 'Diabetic']
    y = df_good['Diabetic']
        # splitting X and y in train and test df
    X_train, X_test, y_train, y_test = sp(X,y, test_size=0.20, random_state=0)

    # Support Vector Machine Classifier
    model_svc = SVC()
    model_svc.fit(X_train, y_train)
    y_pred_svc = model_svc.predict(X_test)

    accuracy_svc_good = accuracy_score(y_test, y_pred_svc)

    # K-nearest Neighbors 
    model_knc = KNC()
    model_knc.fit(X_train, y_train)
    y_pred_knc = model_knc.predict(X_test)

    accuracy_knc_good = accuracy_score(y_test, y_pred_knc)

    # Random Forest
    model_rf = RF()
    model_rf.fit(X_train, y_train)
    y_pred_rf = model_rf.predict(X_test)

    accuracy_rf_good = accuracy_score(y_test, y_pred_rf)

    # creating dataframe creating all results
    result_features = {'Algorithm': ['Support Vector', 'K-nearest Neighbors', 'Random Forest'], 'Accuracy Results All':[accuracy_svc_all, accuracy_knc_all, accuracy_rf_all], 
                        'Accuracy Results Features Highest Correlation': [accuracy_svc_highest, accuracy_knc_highest, accuracy_rf_highest], 'Accuracy Results Features Highest Correlation Plus Good':[accuracy_svc_good, accuracy_knc_good, accuracy_rf_good]}

    df_results = pd.DataFrame.from_dict(result_features)

    st.table(df_results)