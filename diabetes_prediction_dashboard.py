# this is the file that must to be run in order to see the dashboard

from platform import machine
from re import S
import streamlit as st

#import data_exploration
import introduction
import Data_Exploration.data_exploration as exploration
import Data_Analysis.data_analysis as analysis
import Machine_Learning.machinelearning_algorithm as machinelearning
pages = {
    "Introduction": introduction, 
    "Data Exploration": exploration,
    "Data Analysis": analysis,
    "Machine Learning Algorithms": machinelearning
}

st.sidebar.title('Diabetes Prediction App')
selection = st.sidebar.radio("Go to", list(pages.keys()))
page = pages[selection]
page.app()