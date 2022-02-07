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

        # ---------------------- Analysis of alla features and interesting correlation
        st.header('Analysis of all features and interesting correlations')
        st.write('First of all, we wanted to analyse the distribution of each columns and the most interesting correlations between the features of the dataset.')
        st.write("In this first part of data analysis, the target column of the project (the column *Diabetic*) it has not been considered, since every analysis with the target is presented in the next data analysis section.")


        # get the distribution for each feature
        with st.expander('Get distribution of data for each column'):
                st.write('**Notes about distribution plot**: for every features it has been specified a custom order of its values and a specific color for each of them.')
                choosen_column = st.selectbox('Choose a column to see its distribution', df_diabetes.columns.to_list())
                create_distribution_plot(choosen_column, df_diabetes)


        # list that contains the indicator of all interesting correlations
        # that list is used in the sidebar selectbox
        correlation_list = ['highBP with Age', 'PhysicallyActive with Age', 'Age with BMI', 'RegularMedicine with Age', 'BPLevel with Age', 'JunkFood with Age', 'Age with Stress',
                        'Age with UrinationFreq', 'Gender with Smoking', 'Gender with BMI', 'Gender with Alcohol', 'Gender with RegularMedicine', 'Gender with UrinationFreq', 
                        'PhysicallyActive with BMI', 'PhysicallyActive with Gender', 'PhysicallyActive with JunkFood', 'Smoking with Alcohol',
                        'Stress with Smoking', 'Smoking with JunkFood', 'Alcohol with highBP', 'Alcohol with BMI', 'Alcohol with SoundSleep', 'Alcohol with BPLevel', 
                        'Sleep with SoundSleep and BMI', 'JunkFood with BMI', 'RegularMedicine with Family_Diabetes', 'RegularMedicine with JunkFood', 'RegularMedicine with Stress',
                        'RegularMedicine with BPLevel', 'RegularMedicine with UrinationFreq', 'highBP with BMI', 'highBP with Stress', 'highBP with BPLevel', 'highBP with UrinationFreq']
        correlation_list = sorted(correlation_list)
        # add a default element to list 'Select a correlation' so correlation graph is not shown automatically when the app starts
        correlation_list = ['Select a correlation'] + correlation_list


        correlation_to_analyse = st.sidebar.selectbox('Correlation', correlation_list)

        st.write("Select a correlation in the box in the sidebar to see an interesting correlation.")
        if correlation_to_analyse != 'Select a correlation':
                create_correlation_plot(correlation_to_analyse, df_diabetes)
                text_correlation_analysis(correlation_to_analyse)
        if correlation_to_analyse == 'RegularMedicine with JunkFood':
                if st.button('Show correlation between RegularMedicine and blood pressure level (BPLevel)'):
                        create_correlation_plot('RegularMedicine with BPLevel', df_diabetes)
        if correlation_to_analyse == 'RegularMedicine with UrinationFreq':
                if st.button('Show correlation between RegularMedicine and Age'):
                        create_correlation_plot('RegularMedicine with Age', df_diabetes)
                if st.button('Show correlation between UrinationFreq and Age'):
                        create_correlation_plot('Age with UrinationFreq', df_diabetes)
        if correlation_to_analyse == 'Smoking with JunkFood':
                if st.button('Show correlation between Smkong and Alcohol'):
                        create_correlation_plot('Smoking with Alcohol', df_diabetes)
        if correlation_to_analyse == 'highBP with UrinationFreq':
                if st.button('Show correlation between highBP and Age'):
                        create_correlation_plot('highBP with Age', df_diabetes)
                if st.button('Show correlation between Age and UrinationFreq'):
                        create_correlation_plot('Age with UrinationFreq', df_diabetes)

        st.header('Analysis of diabetes causes')
        st.write("The aim of this section of the project is to analyze the main causes of diabetes: to do so, it's important to analyze the correlations between every feature with the target feature *Diabetic*.")

        st.subheader('More details about the target feature')

        st.write('As we can see in the graph below, only 28% of the participants have diabetes.')

        diabetic_values = df_diabetes.Diabetic.value_counts(normalize=True).to_frame().reset_index()
        diabetic_values.columns = ['Diabetic', 'Participants Percentage']
        diabetic_values['Participants Percentage'] = round(diabetic_values['Participants Percentage'] * 100,0)
        diabetic_values = diabetic_values.sort_values(by='Participants Percentage', ascending=False)
        
        fig = plt.figure(figsize=(10,6))
        g = sns.barplot(data=diabetic_values, x='Diabetic', y='Participants Percentage')
        g.set_title('Participants distribution according to Diabetic feature')

        for p in g.patches:
                g.annotate(format(p.get_height(), '.0f'),
                        (p.get_x()+p.get_width() /2., p.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0,9),
                        textcoords = 'offset points',
                        size=12)
        st.pyplot(fig)

        st.write("In the following expander, it's possible to see the correlation between the target feature and all the others.")

        # all features are categorized into four different cagtegories. Each category is a list:
        general_info = ['Age', 'Gender', 'Sleep', 'SoundSleep']
        medical_info = ['Family_Diabetes', 'Pdiabetes', 'Pregnancies', 'RegularMedicine', 'UrinationFreq']
        health_info = ['highBP', 'BMI', 'Stress', 'BPLevel']
        lifestyle_info = ['PhysicallyActive', 'Smoking', 'Alcohol', 'JunkFood']

        # creation of a list contaings the name of the features categories. This list is used for the selectbox in the expander
        categories_list = ['General informations', 'Medical informations', 'Health informations', 'Lifestyle informations']

        with st.expander('Show analysis about the causes of diabetes'):
                st.write('All the features in the dataset have been divided into four different categories:')
                st.markdown('- **general informations**: *Age*, *Gender*, *Sleep* and *SoundSleep*;')
                st.markdown('- **medical informations**: *Famyly_Diabetes*, *Pdiabates*, *Pregnancies*, *RegularMedicine*, and *UrinationFreq*;')
                st.markdown('- **health informations**: *highBP*, *BMI*,  *Stress*, and *BPLevel*; ')
                st.markdown('- **lifestyle informations**: *PhysicallyActive*, *Smoking*, *Alcohol* and *JunkFood*.')

                st.write('Select an option below to see the analysis about diabetes causes (correlation of each feature with the target one).')

                choosen_category = st.selectbox('', ['Select a category'] + categories_list)

                if choosen_category == 'General informations':
                        correlations_with_target(general_info, df_diabetes)
                if choosen_category == 'Medical informations':
                        correlations_with_target(medical_info, df_diabetes)
                if choosen_category == 'Health informations':
                        correlations_with_target(health_info, df_diabetes)
                if choosen_category == 'Lifestyle informations':
                        correlations_with_target(lifestyle_info, df_diabetes)