import streamlit as st


def app():
    st.title('Diabetes Prediction')

    st.header('Introduction')
    
    st.write("The work presented in this application has the goal to analyze the main causes of diabetes and to predict if a person has " +
            "diabetes or not. The data used here can be found on Kaggle at the following link: https://www.kaggle.com/tigganeha4/diabetes-dataset-2019.")

    st.write("The application is composed by three different pages thant can be reach from the menu in the sidebar: ")
    st.markdown('- **Data Exploration**, in which the dataset is explored and data cleaning operations are described.')
    st.markdown("- **Data Analysis**, in which the features of the dataset are analyzed to find some correlations between them and to find " +
                " the main causes of diabetes.")
    st.markdown("- **Machine Learning Algorithms**, in which the original dataset is modified in order to be ready for machine learning algorithms and " +
                "the result of three different algorithms trained on the available data are shown.")

    