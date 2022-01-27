from numpy import number
import pandas as pd


# import the data contained in csv downloaded from Kaggle in a dataframe
df_diabetes = pd.read_csv('data/diabetes_dataset__2019.csv')

print(df_diabetes.info())

print(df_diabetes.head())

columns_list = df_diabetes.columns.to_list()


# check if there are some null values in the columns of the dataframe df_diabetes
list_of_columns_with_null_values = []
for column in columns_list:
    number_of_null_values = df_diabetes[column].isnull().sum()
    if number_of_null_values > 0:
        print('The column ' + column + ' has ' + str(number_of_null_values) + ' null values.')
        list_of_columns_with_null_values.append(column)


for column in list_of_columns_with_null_values:
    print('\nCOLUMN: ' + column)
    print('VALUES: ', df_diabetes[column].value_counts())

