import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

df5KMfilt = st.session_state['df5KMfilt']
dfCGLprodfilt = st.session_state['dfCGLprodfilt']

st.title('5 KM - Tev sales analysis')

## PROVISOIRE
# df5KMfilt = df5KMfilt[df5KMfilt["MATERIAL_C"] == 'ML25930']
# dfCGLprodfilt = dfCGLprodfilt[dfCGLprodfilt["MATERIAL_C"] == 'ML25930']
# df5KMfilt = df5KMfilt[df5KMfilt["BIRTH_YEAR"] == '2018']
# dfCGLprodfilt = dfCGLprodfilt[dfCGLprodfilt["BIRTH_YEAR"] == '2018']

dfret = df5KMfilt.groupby(['MATERIAL_C', 'CAI', 'LPC', 'MACRO_CGL_C', 'RETURN_QUARTER'])['ITEM_NB'].sum().reset_index()
dfret.rename({'RETURN_QUARTER': 'QUARTER', 'ITEM_NB':'Claim by return date'}, axis=1, inplace=True)

dfmanuf = df5KMfilt.groupby(['MATERIAL_C', 'CAI', 'LPC', 'MACRO_CGL_C', 'BIRTH_QUARTER'])['ITEM_NB'].sum().reset_index()
dfmanuf.rename({'BIRTH_QUARTER': 'QUARTER', 'ITEM_NB':'Claim by manufacturing date'}, axis=1, inplace=True)

dfprod = dfCGLprodfilt.groupby(['MATERIAL_C', 'CAI', 'LPC', 'CGL_C', 'PLANT_C', 'BIRTH_QUARTER'])['PROD_QT'].sum().reset_index()
dfprod.rename({'BIRTH_QUARTER': 'QUARTER', 'PROD_QT':'Production', 'CGL_C':'MACRO_CGL_C'}, axis=1, inplace=True)

result = pd.merge(dfret, dfmanuf, on=["MATERIAL_C", "CAI", "LPC", "MACRO_CGL_C", "QUARTER"], how='outer')
dfall = pd.merge(result, dfprod, on=["MATERIAL_C", "CAI", "LPC", "MACRO_CGL_C", "QUARTER"], how='outer')

dfall.sort_values(by=['QUARTER'], inplace=True)

#st.write('dfall', dfall)

# Tev - dfcumul by MATERIAL
result = df5KMfilt.groupby(['MATERIAL_C', 'tq'])['ITEM_NB'].sum() \
    .groupby(level=[0]).cumsum().reset_index()

dfprod = dfCGLprodfilt.groupby(['MATERIAL_C'])['PROD_QT'].sum().reset_index()

dfcumul = pd.merge(result, dfprod, on=["MATERIAL_C"], how='left')

dfcumul['event'] = ((dfcumul['ITEM_NB'] / dfcumul['PROD_QT']) * 1_000_000).round(0)

st.subheader('Survival analysis per Material')

fig, ax = plt.subplots(figsize = (14,8))
sns.set_theme(style="darkgrid")
# Plot the responses for different events and regions
g = sns.lineplot(x="tq", y="event",
                 hue="MATERIAL_C", style="MATERIAL_C",
                 data=dfcumul, ax=ax)
# add label to the axis and label to the plot
g.set(xlabel ="QUARTER", ylabel ='Claims (ppm)')
plt.title('Tev sales by Material', fontsize=25)
ax.legend(ncol = 2, loc = 'upper left')
plt.legend()
plt.xticks(rotation=90)
st.pyplot(fig)

st.write('Table of cumulative claims (ITEM_NB) per Material per Quarter (tq)', dfcumul)

st.subheader('Detail per Material')

# Tev - dfcumul by MATERIAL by YEAR
result = df5KMfilt.groupby(['BIRTH_YEAR', 'MATERIAL_C', 'tq'])['ITEM_NB'].sum() \
    .groupby(level=[0,1]).cumsum().reset_index()

dfprod = dfCGLprodfilt.groupby(['BIRTH_YEAR', 'MATERIAL_C'])['PROD_QT'].sum().reset_index()

dfcumul = pd.merge(result, dfprod, on=["BIRTH_YEAR", "MATERIAL_C"], how='left')

dfcumul['event'] = ((dfcumul['ITEM_NB'] / dfcumul['PROD_QT']) * 1_000_000).round(0)

fig = sns.relplot(data=dfcumul, x="tq", y="event", col="MATERIAL_C", col_wrap=2, kind="line",
                  hue='BIRTH_YEAR').tick_params(axis='x', rotation=90, labelsize=8)
st.pyplot(fig)