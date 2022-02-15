# this is the file that must to be run in order to see the dashboard

from platform import machine
from re import S
import streamlit as st

import data_exploration
import data_analysis
import machinelearning_algorithm
import introduction

pages = {
    "Introduction": introduction, 
    "Data Exploration": data_exploration,
    "Data Analysis": data_analysis,
    "Machine Learning Algorithms": machinelearning_algorithm
}

st.sidebar.title('Diabetes Prediction App')
selection = st.sidebar.radio("Go to", list(pages.keys()))
page = pages[selection]
page.app()