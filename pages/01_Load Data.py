import numpy as np
import pandas as pd
import datetime as dt
import streamlit as st

st.header(':blue[Load the dataset] :ok_hand:')
st.subheader('1. Returns .csv file')
st.subheader('2. CGL production')

uploaded_files = st.file_uploader("Upload CSV", type="csv", accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        file.seek(0)
    uploaded_data_read = [pd.read_csv(file, sep=';') for file in uploaded_files]
    #https: // towardsdatascience.com / creating - true - multi - page - streamlit - apps - the - new - way - 2022 - b859b3ea2a15
    st.session_state['df5KM'] = uploaded_data_read[0]
    df5KM = st.session_state['df5KM']

    if len(uploaded_data_read) == 2:
        dfCGLprod = uploaded_data_read[1]

        st.session_state['dfCGLprod'] = dfCGLprod

