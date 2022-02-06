# this is the file that must to be run in order to see the dashboard

from re import S
import streamlit as st

import data_exploration
import data_analysis

pages = {
    "Data Exploration": data_exploration,
    "Data Analysis": data_analysis
}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(pages.keys()))
page = pages[selection]
page.app()