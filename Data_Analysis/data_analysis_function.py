import pandas as pd 
import streamlit as st
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
    if col == 'Pregnancies':
        list_order_values = ['0.0', '1.0', '2.0', '3.0', '4.0']
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

# function to create a dataframe that make a groupby for the category of the column in analysis 
# with the ecategories of the column for which we want to analyse the correlation
def groupby_to_have_percentage_for_categories(col, col_corr, df):
    # groupby for the column in analysis (col1)
    df_col1_groupby = df.groupby(col).size().reset_index()
    name_column = 'Total_Participants_for_' + col + '_categories'
    df_col1_groupby.columns = [col, name_column]
    #groupby for column in analysis (col1) and the columnt choosen for correlation (col_corr)
    df_groupby_corr = df.groupby([col, col_corr]).size().reset_index()
    df_groupby_corr.columns = [col, col_corr, 'Number of Participants']
    
    #add total participants for category to df_groupby_corr
    df_groupby_corr = pd.merge(df_groupby_corr, df_col1_groupby, on=col)
    df_groupby_corr['Participants percentage'] = df_groupby_corr['Number of Participants'] / df_groupby_corr[name_column] * 100
    df_groupby_corr['Participants percentage'] = round(df_groupby_corr['Participants percentage'],0)
    
    return df_groupby_corr


def text_correlation_analysis(choosen_correlation):
    if choosen_correlation == 'Age with BMI':
        st.write("Analyzing the graph above, it's possible to notice a growth in the percentage of participants with a BMI indicator above 30 with increasing of the age, until the participants have an age smaller than 60." +
        " The percentage of paticipants with a BMI greather than 30 decreases among the participants older than 60. Also, in the latter category it's possible to notice that there is a higher percentage of people with a BMI" + 
        " indicator lower than 20. So, it's possible to say that older people (category *60 or older*) tend to have a lower BMI indicator: this can be caused by the fact that, in general," + 
        " older people eat less than they did before.")
    if choosen_correlation == 'Alcohol with BMI':
        st.write("Analyzing the graph above, it's possible to notice that most of the participants who consume alcohol has a BMI indicator grather than 25, while those who declare to not consume alcohol " + 
                " has a BMI indicator lower than 25, with high probability. So, it's possible to say that if a person consumes alcohol, he has a higher percentage of fat mass than a person who does not drink alcohol, " +
                "with high probability."  )
    if choosen_correlation == 'Alcohol with BPLevel':
        st.write("Analyzing the graph above, it's possible to see that the percentage of participants with high blood pressure (red columns) is higher among those who declare to consume alcohol. Furthermore, " + 
                "there are no alcohol-consuming participants that have a low blood pressure (yellow column). So, it's possible to say that who consumes alcohol has a higher risk to have high blood pressure than those "
                + "who do not drink alcohol.")
    if choosen_correlation == 'Alcohol with SoundSleep':
        st.write("Analyzing the graph above, it's possible to notice that the portion where the distribution is greather than 6 (more than 6 hours of soundsleep) is bigger among the participants that consume alcohol." + 
                " So, it's possible to say that alcohol consumption may be a cause of more soundsleep hours." )
    if choosen_correlation == 'Alcohol with highBP':
        st.write("Analyzing the graph above, it's possible to notice that there are more participants with high blood pressure diagnosed among those who declare to consume alcohol. Indeed. the percentage" + 
        " for this category is 33, while it is 22 among thosee who don't drink alcohol.")
    if choosen_correlation == 'BPLevel with Age':
        st.write("Analyzing the graph above, it's possible to notice that the 'level' of blood pressure increases with the age of the participants. Indeed, if we consider the portion of the graph relating to those " + 
                "who have high blood pressure we have higher percentage among the participants with an age greather than 50 (last two columns). so, it's possible to say that older people have higher probebility to have high blood pressure.")
    if choosen_correlation == 'Gender with Alcohol':
        st.write("Analyzing the graph above, it's possible to notice that alcohol consumption higher among males: 30% against 4% among the females.")
    if choosen_correlation == 'Gender with BMI':
        st.write("Analyzing the graph above, it's possible to notice that the portion where the distribution is greather than 30 (BMI is greather than 30) is bigger among females. In particular, the distribution of " + 
                "the BMI indicator relating to males highliths that most males have an indicator less than or equal to 25. So, females participants have higher fat mass percentage with higher probability.")
    if choosen_correlation == 'Gender with RegularMedicine':
        st.write("Analyzing the graph above, it's possible to notice that female participants take medicine more regularly than males (red columns): 43% vs 30%.")
    if choosen_correlation == 'Gender with Smoking':
        st.write("Analyzing the graph above, it's possible to notice that only male participants smoke (only red column for 'male' category).")
    if choosen_correlation == 'Gender with UrinationFreq':
        st.write("Analyzing the graph abobe, it's possible to notice that female participants have an higher urination frequency than male participants (green column): 41% vs 23%.")
    if choosen_correlation == 'JunkFood with Age':
        st.write("Analyzing the graph above, it's possible to notice that the consumption of junk food is lower among older people (60 or older). Furthermore, participants who consume junk food the most " +
                " are those with an age lower than 40.")
    if choosen_correlation == 'JunkFood with BMI':
        st.write("Analyzing the graph above, it's possible to notice that a high consumption of junk food does not correspond to an increase in the BMI indicator. Indeed, the differences between the four distributions" +
                " are not so apparent.")
    if choosen_correlation == 'PhysicallyActive with Age':
        st.write("Analyzing the graph above, it can be seen that the level of physical activity decreases with increasing age, as everyone might expect.")
    if choosen_correlation == 'PhysicallyActive with BMI':
        st.write("Analyzing the graph above, it can be seen that more physical activity does not correspond to a better BMI indicator. Indeed, it seems that most of the people who do not do physical activity (none)"+
                " have a better BMI indicator than everyone else ")
    if choosen_correlation == 'PhysicallyActive with Gender':
        st.write("Analyzing the graph above, it can be seen that male participants do more physical activity than females.")
    if choosen_correlation == 'PhysicallyActive with JunkFood':
        st.write("Everyone can think that more physical activity corresponds to a healthier life style, so it might be thought that participants who exercise more might be expected to consume less junk food." + 
        " However by analyzing the graph above, it can be seen that participants that do one hour or more of physical activity are the ones with higher consume of junk food.")
    if choosen_correlation == 'RegularMedicine with Age':
        st.write("Analyzing the graph above, it can be seen that older participants take medicine more regularly.")
    if choosen_correlation == 'RegularMedicine with BPLevel':
        st.write("Analyzing the graph above, it can be seen that there are more people with high blood pressure level among the participants who declare to take medicine regularly. Considering the data represented, " +
                "we can say that having high blood pressre it's one of the main causes of taking medicine regularly.")
    if choosen_correlation == 'RegularMedicine with Famliy_Diabetes':
        st.write("Analyzing the graph above, it can be seen that the participants who have cases of diabetes in the family take medicine regularly with higher probability.")
    if choosen_correlation == 'RegularMedicine with JunkFood':
        st.write("Analyzing the graph above, it can be seen that the participants who takes medicine regularly eat less junk food. This could be caused by the fact that doctors suggested a more balanced diet when prescribing the medicines: " + 
        " to better understand what has just ben described, it is good to consider the fact that most of those who take medicines regularly have high blood pressure which can get worse if you eat junk food." )
    if choosen_correlation == 'RegularMedicine with Stress':
        st.write("Analyzing the graph above, it can be see that the participants that take medicine regularly have a high level stress (*very often* or *always*) with more probability than those who " +
                "don't take medicine.")
    if choosen_correlation == 'RegularMedicine with UrinationFreq':
        st.write("Analyzing the graph above, it can be seen that the participants that take medicine regularly have a high urination frequency (*quite often*) with  more probability."+
                " This could be caused by the fact that those who tak emedicine regularly are mostly elederly people, who have a high probability of having high urination frequency." )
    if choosen_correlation == 'Smoking with Alcohol':
        st.write("Analyzing the graph above, it can be seen that most of the participants who smoke also consume alcohol.")
    if choosen_correlation == 'Smoking with JunkFood':
        st.write("Analyzing the graph above, it can be seen that who smoke have an higher junk food consumption. This can be seen as who smoke don't have a healthy life-style, above all if we consider also the fact that who smoke also consume alcohol")
    if choosen_correlation == 'Age with Stress':
        st.write("Analyzing the graph above, it can be seen that the level of stress increase with the age of the participants, without considering the participants older than 60. Indeed, the older participants (60 or older) have less stress than" +
        " those with an age between 50 and 59: this could be caused by the fact that most of old people don't work anymore and everyone knows that work is one of the major causes of stress.")
    if choosen_correlation == 'Stess with Smoking':
        st.write("Analyzing the graph above, it can be seen that if a participant has an high level of stress (*very often* or *always*) smoke with more probability.")
    if choosen_correlation == 'Age with UrinationFreq':
        st.write('Analyzing the graph above, it can be seen that urination frequency is high among older people with more probability: indeed, the 47% of participants older than 60 have the highest urination frequency (**quite often**).')
    if choosen_correlation == 'highBP with Age':
        st.write('Analysing the graph above, it can be seen that the percentage of participants with high blood pressure diagnosed increases with the age of the participants. So, we can say that the older a person is, the more likely he is' + 
                ' to be diagnosed with high blood pressure.')
    if choosen_correlation == 'highBP with BMI':
        st.write("Analyzing the graph above, it can be seen that the participants with high blood pressure diagnosed have a worst BMI distribution. Indeed, it can be seen that most of the participants with high blood pressure  " +
                "have a BMI between 25 and 30, while most of the participants with no high blood pressure diagnosed have a BMI between 20 and 25. So, it seems that high blood pressure is one of the causes of a worst BMI. ")
    if choosen_correlation == 'highBP with BPLevel':
        st.write('As it can be seen in the graph above, 25% of the participants diagnosed with high blood pressure at the moment of the questionaire had a normal blood pressure, while the 6% that were diagnosed with no high' + 
                ' blood pressure had high blood pressure when compiling the questionaire. ')
    if choosen_correlation == 'highBP with Stress':
        st.write("From the graph above, it's possible to see that there is an higher percentage of participants that declare the highest level of stress (always) among the ones whos has been diagnosed with high Blood Pressure")
    if choosen_correlation == 'highBP with UrinationFreq':
        st.write("Analyzing the graph above, it can be seen that there are more participants with high urination frequency (*quite often*) among those with high blood pressure diagnosed. This could be caused by the fact that most" + 
                " of older participants have been diagnosed with high blood pressure and they are the category with the highest percentage of people with high urination frequency.")   
    if choosen_correlation == 'Sleep with SoundSleep and BMI':
        st.write("Analyzing the graph above, it can be seen that there are not so strong correlations between the features considered. The strongest correlations is between *Sleep* and *SoundSleep* with a Pearson coeffient of 0.53. " +
                "All the others couples denote a negative correlation but not so strong.")
# function to create the plot used to analyse the correlation than can be choose in the sidebar
def create_correlation_plot(choosen_correlation, df):
    if choosen_correlation != 'Sleep with SoundSleep and BMI':
        col1 = choosen_correlation.split(" with ")[0]
        col2 = choosen_correlation.split(" with ")[1]


        if (df[col2].dtype != 'object'):
            df = custom_order_based_on_values_one_col(col1, df)
            colors_based_on_value = color_sequence_for_graph(col1)
            fig = plt.figure(figsize=(10,6))
            g = sns.violinplot(x=col1, y=col2, data = df, palette=colors_based_on_value)
            g.set_title(col1 + ': correlation with column ' + col2, size=12)
            g.set_xlabel('')
            st.pyplot(fig)
        else:
            # creation of a dataset to groupby the data according to the two columns for which the correlation is selected
            df_groupby = groupby_to_have_percentage_for_categories(col1, col2, df)
            # df_groupby sorted according to the custom order defined for the two columns
            df_groupby = custom_order_based_on_values_two_col(col1, col2, df_groupby)
            # need to use the colors defined in the function 'color_sequence_for_graph' for the second column (col2)
            colors_based_on_value = color_sequence_for_graph(col2)

            fig = plt.figure(figsize=(10,6))
            g = sns.barplot(data=df_groupby, x=col1, y='Participants percentage', hue=col2, palette=colors_based_on_value)
            g.set_title(col1 + ': correlation with column ' + col2, size=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            for p in g.patches:
                g.annotate(format(p.get_height(), '.0f'),
                        (p.get_x()+p.get_width() /2., p.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0,9),
                        textcoords = 'offset points',
                        size=12)
            st.pyplot(fig)
    else:
        cols_list_for_matrix = ['Sleep', 'SoundSleep', 'BMI']
        # dataset that contains obly the columns for the heatmap
        df_for_heatmap = df[cols_list_for_matrix]

        # heatmap creation
        fig = plt.figure(figsize=(10,6))
        sns.heatmap(df_for_heatmap.corr(), annot=True, cmap='OrRd')

        st.pyplot(fig)

# function to create correlation plot for between every feature and the target feature
def correlations_with_target(category_list, df):
    # only the columns in the category_list + the feature target
    df_category = df[category_list + ['Diabetic']]
    colors_based_on_value = color_sequence_for_graph('Diabetic')
    if len(category_list) == 4:
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,8))
        for j, i in enumerate(category_list):
            if  (df_category[i].dtype == 'object') | (df_category[i].dtype == 'category'):
                df_perc = groupby_to_have_percentage_for_categories(i, 'Diabetic', df_category)
                df_perc = custom_order_based_on_values_two_col(i, 'Diabetic', df_perc)
                g = sns.barplot(data=df_perc, x=i, y='Participants percentage', hue='Diabetic', palette = colors_based_on_value, ax=axes.flat[j])
                g.set_xlabel('')
                g.set_title('Diabetic correlation with ' + i, fontsize=8)
                if i == 'PhysicallyActive':
                    for item in g.get_xticklabels():
                        item.set_rotation(20)
            else:
                g = sns.violinplot(df_category['Diabetic'],df_category[i], palette = colors_based_on_value, ax=axes.flat[j])
                g.set_xlabel('')
                g.set_title('Diabetic correlation with ' + i, fontsize=8)
        st.pyplot(fig)
    if len(category_list) == 5:
        fig1, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
        for j, i in enumerate(category_list[:3]):
            if df_category[i].dtype == 'object':
                df_perc = groupby_to_have_percentage_for_categories(i, 'Diabetic', df_category)
                df_perc = custom_order_based_on_values_two_col(i, 'Diabetic', df_perc)
                g = sns.barplot(data=df_perc, x=i, y='Participants percentage', hue='Diabetic', palette = colors_based_on_value, ax=axes.flat[j])
                g.set_xlabel('')
                g.set_title('Diabetic correlation with ' + i, fontsize=8)
            else:
                g = sns.violinplot(x='Diabetic',y=i, data=df_category, palette = colors_based_on_value, ax=axes.flat[j])
                g.set_xlabel('')
                g.set_title('Diabetic correlation with ' + i, fontsize=8)
        fig2, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
        for j, i in enumerate(category_list[3:]):
            if df_category[i].dtype == 'object':
                df_perc = groupby_to_have_percentage_for_categories(i, 'Diabetic', df_category)
                df_perc = custom_order_based_on_values_two_col(i, 'Diabetic', df_perc)
                g = sns.barplot(data=df_perc, x=i, y='Participants percentage', hue='Diabetic', palette = colors_based_on_value, ax=axes.flat[j])
                g.set_title('Diabetic correlation with ' + i, fontsize=8)
            else:
                g = sns.violinplot(x='Diabetic',y=i, data=df_category, palette = colors_based_on_value, ax=axes.flat[j])
        st.pyplot(fig1)
        st.pyplot(fig2)

# function to define text analysis for correlations with the target feature
def text_correlations_with_target(choosen_category):
    if choosen_category == 'General informations':
        st.markdown('- From the first graph it can be noticed that the number of participants with diabetes is higher among older people;')
        st.markdown("- From the second graph it's not possible to noticed a particular behavior that allows us to define whether a male is more likely to have diabetes or vice versa;")
        st.markdown("- From the third and the fourth graph it's possible to see that the distributions regarding the amount of sleep hour and sound sleep hours have no such differences" +
                    " as to say that a certain number of hours of sleep or soundsleep is more likely to cause diabetes among the participants.")
        
        st.write("Concluding, it's possible to say that the age can be a determining factor for diabetes: so, the feature *Age* can be an important feature in predicting whether or not a person may have diabetes.")
    if choosen_category == 'Medical informations':
        st.markdown("- *Family_Diabetes*: it can be noticed that among the participants with diabetes cases in their family there is a high percentage of people with diabetes.")
        st.markdown("- *Pdiabetes*: it can be noticed that most of the participants with gestation diabetes are diabetic. ")
        st.markdown("- *Pregnancies*: it can be noticed that the percentage of diabetic participants increases with the number of pregnancies. In particular, it can be seen " +
                    "all the participants with 4 participants are diabetic.")
        st.markdown("- *RegularMedicine*: it can be noticed that about 60% of the participants that takes medicine regularly are diabetic, while there is a little number of cases" + 
                    " among the participants that don't take medicine regularly.")
        st.markdown("- *UrinationFreq*: it can be seen that there are more diabetic people among the participants with the highest urination frequency. However, the difference is " +
                    " not such that it can be said that a high urination frequency indicates diabetes.")
        
        st.write('Concluding, *having diabetes cases in the family*, *having gestation diabetes*, *the number of pregnancies* and *taking medicine regularly* can be' +
                'considered as causes of diabetes. So, the features *Family_Diabetes*, *Pdiabetes*, *Pregnancies* and *RegularMedicine* could have an important role' +
                ' in predicting whether or not a person is diabetic.')

    if choosen_category == 'Health informations':
        st.markdown("- *highBP*: it can be noticed that about 60% of the participants with high blood pressure diagnosed are diabetic, while only the 20% among the others.")
        st.markdown("- *BMI*: it can be noticed that there is a little difference between the two distributions. Most of the diabetic participants have an higher BMI.")
        st.markdown("- *Stress*: it can be noticed that the percentage of diabetic people increases with the stress level pf the participants.")
        st.markdown("- *BPLevel*: it can be noticed that there are no diabetic people among the participants with low blood pressure, while most of " +
                    "the participants with high blood pressure are diabetic.")

        st.write("Concluding, *having high blood pressure* and *high level of stress* could be considered as causes of diabates, while the difference noted about the BMI indicator " +
                "is not sufficient to define BMI as a cause of diabetes. So, the features *highBP*, *Stress* and *BPLevel* could have an important role " +
                "in predicting whether or not a person is diabetic.")
    
    if choosen_category == 'Lifestyle informations':
        st.markdown("- *PhysicallyActive*: it can be noticed that there is a high percentage of diabetic people among the participants not physically active.")
        st.markdown("- *Smoking*: it can be noticed that there is any behavior that allows us to define whether a smoker is more likely to be diabetic.")
        st.markdown("- *Alcohol*: it can be noticed that there is a higher percentage of diabetic people among the participants that consume alcohol.")
        st.markdown("- *JunkFood*: it can be noticed that there is any behavior that allows us to define if a particular level of junk food consumption can cause diabetes.")

        st.write("Concluding, *not being physically active* can be a cause of being diabetic and *alcohol consumption* can be a minor cause. So, the features " + 
                "*PhysicallyActive* and *Alcohol* could have an important role in predicting whether or not a person is diabetic.")
