import types
import pandas as pd
import matplotlib.pyplot as plt
from data_analysis_function import *

original_dataset = pd.read_csv('data/diabetes_dataset__2019.csv')
# applying of data cleaning functions to the original dataset
df_diabetes = dataset_datacleaning(original_dataset)

def app():

    st.title('Data Analysis')
    st.write('After expoliring and cleaning the original dataset, the work continued with some data analysis with the following goals:')
    st.markdown('- analyze every features of the dataset and the possible correlations between them; ')
    st.markdown('- analyze the main causes of diabetes;')

    # get list of all possible types in the dataset
    types_of_col_in_dataframe = get_list_of_columns_types(df_diabetes)
    text_types_of_col_in_dataframe = ', '.join(types_of_col_in_dataframe)
    
    st.write('The type of the columns and the kind of values have been considered during data anlysis, in order to better analyse each feature of the dataset.' +
            ' There are three different types of columns in the dataset: ' + text_types_of_col_in_dataframe + '.')
    st.write('All columns of type *object* can be considered as **categorical** features, while columns of type **float64** and **int64** are numerical features. ' +
            'The column *Pregnancies* (type *float64*) during the all data analysis process is considered as a categorical variable since it admits only 5 values: from 0 to 5.')

    st.header('Analysis of all features and their correlation')
    st.write('First of all, we wanted to analyse the distribution of each columns and the most interesting correlations between the features of the dataset.')
    st.write("In this first part of data analysis, the target column of the project (the column *Diabetic*) it has not been considered, since every analysis with the target is presented in the next data analysis section.")
    

    # get the distribution for each feature
    with st.expander('Get distribution of data for each column'):
        st.write('**Notes about distribution plot**: for every features it has been specified a custom order of its values and a specific color for each of them.')
        choosen_column = st.selectbox('Choose a column to see its distribution', df_diabetes.columns.to_list())
        create_distribution_plot(choosen_column, df_diabetes)


    # list that contains the indicator of all interesting correlations
    # that list is used in the sidebar selectbox
    correlation_list = ['highBP with Age', 'PhysicallyActive with Age', 'Age with BMI', 'RegularMedicine with Age', 'BPLevel with Age', 'JunkFood with Age', 'Stress with Age',
                        'UrinationFreq with Age', 'Gender with Smoking', 'Gender with BMI', 'Gender with Alcohol', 'Gender with RegularMedicine', 'Gender with UrinationFreq', 
                        'PhysicallyActive with BMI', 'PhysicallyActive with Gender', 'PhysicallyActive with JunkFood', 'UrinationFreq with PhysicallyActive', 'Smoking with Alcohol',
                        'Stress with Smoking', 'Smoking with JunkFood', 'Alcohol with highBP', 'Alcohol with BMI', 'Alcohol with SoundSleep', 'Alcohol with BPLevel', 
                        'Sleep with SoundSleep and BMI', 'JunkFood with BMI', 'RegularMedicine with Family_Diabetes', 'RegularMedicine with JunkFood', 'RegularMedicine with Stress',
                        'RegularMedicine with BPLevel', 'RegularMedicine with UrinationFreq', 'highBP with BMI', 'highBP with Stress', 'highBP with BPLevel', 'highBP with UriantionFreq']
    correlation_list = sorted(correlation_list)


    correlation_to_analyse = st.sidebar.selectbox('Correlation', correlation_list)

    st.write("In the sidebar there is a select box in which it's possible to select a correlation to see the analysis among those that were found to be most interesting in the analysis phase. " +
            "After selecting the couple, the correlation analysis is shown below.")

    create_correlation_plot(correlation_to_analyse, df_diabetes)