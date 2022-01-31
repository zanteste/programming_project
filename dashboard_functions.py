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

# function to define text for the best correlation analysis founded
def text_for_correlation_hypothesis(choosen_column, df_diabetes):
    st.write('It can be interesting to analyse the correlation with the following columns:')
    if choosen_column == 'Age':
        st.markdown("- **highBP**: analyse the probability of having high blood pressure based on the age;")
        st.markdown("- **PhysicallyActive**: analyse how time spent in sport activities based on the age;")
        st.markdown("- **BMI**: analyse if the fat percentage change based on the age;")
        st.markdown("- **RegularMedicine**: it's normal to think that old people take medicine regularly;")
        st.markdown("- **BPLevel**: analyse if the level of the blood pressure changes due to the age.")
    if choosen_column == 'Gender':
        st.markdown("- **Smoking**: analyse the probability of smoking based on the gender;")
        st.markdown("- **Alcohol**: analyse the probability of alcohol consumption based on the gender;")
        st.markdown("- **JunkFood**: analyse the probability of junkfood consumption based on the gender;")
        st.markdown("- **PhysicallyActive**: analyse if there is a trend in time spent in sport activities based on gender.")
    if choosen_column == 'PhysicallyActive':
        st.markdown("- **highBP**: analyse if a healthier lifestyle affects the possibility to have a high blood pressure;")
        st.markdown("- **BMI**: it's common to think that a healthier lifestyle may cause a lower fat percentage; ")
        st.markdown("- **Smoking**: it's common to think that a sports person has less probability to smoke;")
        st.markdown("- **Alcohol**: it's common to think that a sports person doesn't like to consume to much alcohol;")
        st.markdown("- **JunkFood**: it's common to think that a sports person prefers to consume healthier food;")
        st.markdown("- **Stress**: usually a sports man is less stressed than a person who doesn't play any sports, since sport is a good cure for stress;")
        st.markdown("- **UrinationFreq**: usaully, a sports man drinks a lot of water so he is expected to have an higher urination frequency;")
        st.markdown("- **Sleep**: analyse if a healthier lifestyle affects the number of sleep hours;")
        st.markdown("- **SoundSleep**: analyse if a healthier lifestyle affects the number of soundsleep hours;")
    if choosen_column == 'Smoking':
        st.markdown("- **Alcohol**: analyse if there is a connection between smoking and alcohol consumption; ")
        st.markdown("- **Stress**: analyse if a person who smokes has an higher level of stress compared to a non smoker;")
        st.markdown("- **Sleep**: analyse if smoking affects the number of sleep hours;")
        st.markdown("- **SoundSleep**: analyse if smoking affects the number of soundsleep hours")
    if choosen_column == 'Alcohol':
        st.markdown("- **Stress**: analyse if alcohol consumption causes a higher level of stress;")
        st.markdown("- **Sleep**: analyse if alcohol consumption affects the number of sleep hours;")
        st.markdown("- **SoundSleep**: analyse if alcohol consumption affects the number of soundsleep hours.")
    if choosen_column == 'Sleep':
        st.markdown("- **SoundSleep**: analyse if the hours of sleep are, in some way, correlated with the hours of soundsleep.")
        st.markdown("- **BMI**: analyse if the stress level can be, in some way, be correlected to a higher fat percentage.")
        st.write('Since *Sleep* and the two suggested columns are numeric, the correlation is analysed with the correlation matrix.')
    if choosen_column == 'JunkFood':
        st.markdown("- **BMI**: it's common to think that if someone consumes more junky food there is a higher probability to have more fat percentage;")
    if choosen_column == 'RegularMedicine':
        st.markdown("- **BMI**: analyse if taking medicine regularly affects the fat percetnage of a person;")
        st.markdown("- **PhysicallyActive**: analyse if taking medicine regularly affects the sports activity of a person.")

        
        



# function to create a plot to analyse the correlation between two columns
def create_correlation_plot(col1, col_corr, df):
    if (df[col_corr].dtype != 'object') & (col_corr != 'Pregnancies'):
        fig = plt.figure(figsize=(10,6))
        g = sns.violinplot(x=col1, y=col_corr, data = df)
        g.set_title(col1 + ': correlation with column ' + col_corr, size=12)
        g.set_xlabel('')
        st.pyplot(fig)
    else:
        # groupby for the column in analysis (col1)
        df_col1_groupby = df.groupby(col1).count()['Diabetic'].to_frame().reset_index()
        name_column = 'Total_Participants_for_' + col1 + '_categories'
        df_col1_groupby.columns = [col1, name_column]
        #groupby for column in analysis (col1) and the columnt choosen for correlation (col_corr)
        df_groupby_corr = df.groupby([col1, col_corr]).count()['Diabetic'].to_frame().reset_index()
        df_groupby_corr.columns = [col1, col_corr, 'Number of Participants']
        
        #add total participants for category to df_groupby_corr
        df_groupby_corr = pd.merge(df_groupby_corr, df_col1_groupby, on=col1)
        df_groupby_corr['Participants percentage'] = df_groupby_corr['Number of Participants'] / df_groupby_corr[name_column] * 100
        df_groupby_corr['Participants percentage'] = round(df_groupby_corr['Participants percentage'],0)

        # df_groupby_corr custom sort values based on 'Age' categories
        if col1 == 'Age':
            df_groupby_corr['Age'] = pd.Categorical(df_groupby_corr['Age'], ['less than 40', '40-49', '50-59', '60 or older'])
            df_groupby_corr = df_groupby_corr.sort_values(by='Age')
        fig = plt.figure(figsize=(10,6))
        g = sns.barplot(data=df_groupby_corr, x=col1, y='Participants percentage', hue=col_corr)
        for p in g.patches:
            g.annotate(format(p.get_height(), '.0f'),
                      (p.get_x()+p.get_width() /2., p.get_height()),
                      ha = 'center', va = 'center',
                      xytext = (0,9),
                      textcoords = 'offset points',
                      size=12)
        st.pyplot(fig)

