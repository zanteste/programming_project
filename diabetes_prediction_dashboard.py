# this is the file that must to be run in order to see the dashboard

from re import S
import streamlit as st

from multipage import MultiPage
import data_exploration
import data_analysis

app = MultiPage()

## adding pages to the streamlit app
app.add_page('Data Analysis', data_analysis.app)
app.add_page('Data Exploration and Data Cleaning', data_exploration.app)

app.run()