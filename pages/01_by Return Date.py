import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def funcRET(x):
    rdt = x['RETURN_DT']
    return dt.datetime(int(rdt[6:]), int(rdt[3:5]), int(rdt[0:2]), 0, 0)


def calcul_birth(df, week=True, tod='MANUFACT_YEAR_WEEK'):
    def funcAGE(x):
        try:
            return dt.date.fromisocalendar(int(x[tod][:4]), int(x[tod][4:6]), 3)
        except:
            return dt.date.fromisocalendar(int(x[tod][:4]), int(x[tod][4:6])-1, 3)

    def funcAGE_MONTH(x):
        return dt.datetime(int(x[tod][:4]), int(x[tod][4:6]), 15)

    # Using Series.astype() to convert column to string
    df[tod] = df[tod].astype(str)
    if week:
        df['BIRTH'] = df.apply(funcAGE, axis=1)
    else:
        df['BIRTH'] = df.apply(funcAGE_MONTH, axis=1)

    df['BIRTH'] = pd.to_datetime(df['BIRTH'])
    df['BIRTH_YEAR'] = df['BIRTH'].dt.year
    df['BIRTH_QUARTER'] = df['BIRTH'].dt.quarter
    df['BIRTH_YEAR'] = df['BIRTH_YEAR'].astype(str)
    df['BIRTH_QUARTER'] = df['BIRTH_QUARTER'].astype(str)
    df['BIRTH_QUARTER'] = df['BIRTH_YEAR'] + df['BIRTH_QUARTER']

    if week:
        df['td'] = (df['RETURN_DT'] - df['BIRTH']) / np.timedelta64(1, 'D')
        df['tw'] = (df['RETURN_DT'] - df['BIRTH']) / np.timedelta64(1, 'W')
        df['tm'] = (df['RETURN_DT'] - df['BIRTH']) / np.timedelta64(1, 'M')
        df['tq'] = (df['RETURN_DT'].dt.year - df['BIRTH'].dt.year) * 4 + (
                    df['RETURN_DT'].dt.quarter - df['BIRTH'].dt.quarter)
        df['tq'] = df['tq'].astype('int')
        df['ty'] = (df['RETURN_DT'] - df['BIRTH']) / np.timedelta64(1, 'Y')

    return df

# load dataset "5KM" using relative path
df5KM = pd.read_csv("./Data/5KM_returns.csv", sep=";")
df5KMfilt = df5KM
df5KMfilt['ARD'] = df5KMfilt['ARD'].astype(str)
df5KMfilt = df5KMfilt[df5KMfilt["MANUFACT_YEAR_WEEK"] != 195052]

# load dataset CGL production
dfCGLprod = pd.read_csv("./Data/5KM_production.csv", sep=";")
dfCGLprodfilt = dfCGLprod
dfCGLprodfilt = dfCGLprodfilt[dfCGLprodfilt['PROD_QT'].notnull()]

df5KMfilt['RETURN_DT'] = df5KMfilt['RETURN_DT'].apply(lambda x: x[:10])

df5KMfilt['RETURN_DT'] = df5KMfilt.apply(funcRET, axis=1)
# Using dt.to_period() to convert column to Quarter
df5KMfilt['RETURN_QUARTER'] = df5KMfilt['RETURN_DT'].dt.to_period('Q').dt.strftime('%Y%q')
df5KMfilt['RETURN_YEAR'] = df5KMfilt['RETURN_DT'].dt.to_period('Y').dt.strftime('%Y')

calcul_birth(df5KMfilt, week=True)
calcul_birth(dfCGLprodfilt, week=False, tod='YEAR_MONTH')

st.title('5 KM - Material claims analysis')
st.subheader('Selection box')
c1, c2, c3 = st.columns(3)
# GEOGRAPHIC_AREA: Find unique values of a column and convert to a list
lstsel_reg = df5KMfilt.GEOGRAPHIC_AREA_C.unique().tolist()
geo = c1.multiselect(
    'What are your favorite Geographic area',
    lstsel_reg,
    lstsel_reg)
df5KMfilt = df5KMfilt[df5KMfilt['GEOGRAPHIC_AREA_C'].isin(geo)]

# ARD: Find unique values of a column and convert to a list
lstsel_reg = df5KMfilt.ARD.unique().tolist()
geo = c2.multiselect(
    'What are your favorite ARD damage code',
    lstsel_reg,
    lstsel_reg)
df5KMfilt = df5KMfilt[df5KMfilt['ARD'].isin(geo)]

# PLANT: Find unique values of a column and convert to a list
lstsel_reg = df5KMfilt.MANUFACT_PLANT_C.unique().tolist()
geo = c3.multiselect(
    'What are your favorite SCULPTURE',
    lstsel_reg,
    lstsel_reg)
df5KMfilt = df5KMfilt[df5KMfilt['MANUFACT_PLANT_C'].isin(geo)]

# SCULPTURE: Find unique values of a column and convert to a list
lstsel_reg = df5KMfilt.SCULPTURE.unique().tolist()
geo = st.sidebar.multiselect(
    'What are your favorite SCULPTURE',
    lstsel_reg,
    lstsel_reg)
df5KMfilt = df5KMfilt[df5KMfilt['SCULPTURE'].isin(geo)]

st.write('Number of claims returns selected: ', df5KMfilt.shape[0])

st.subheader('Claims by return date analysis')
dfretcumul = df5KMfilt.groupby(['RETURN_QUARTER', 'MATERIAL_C'])['ITEM_NB'].sum() \
    .groupby(level=0).cumsum().reset_index()
dfretcumul.rename({'RETURN_QUARTER': 'QUARTER', 'ITEM_NB':'Claim number'}, axis=1, inplace=True)

fig, ax = plt.subplots(figsize = (14,12))
sns.set_theme(style="darkgrid")
# Plot the responses for different events and regions
g = sns.lineplot(x="QUARTER", y="Claim number",
                 hue="MATERIAL_C", style="MATERIAL_C",
                 data=dfretcumul, ax=ax)
# add label to the axis and label to the plot
g.set(xlabel ="QUARTER", ylabel ='Claims')
plt.title('Number of claims per material per quarter (Return date)', fontsize=25)
ax.legend(ncol = 2, loc = 'upper left')
plt.legend()
plt.xticks(rotation=90)
st.pyplot(fig)

lstutil = ["MATERIAL_C"]
dfpivot = pd.pivot_table(dfretcumul, index=lstutil, columns=['QUARTER'],
               values=["Claim number"], aggfunc=np.sum)

dfpivot.columns = dfpivot.columns.droplevel(0) #remove ITEM_NB
dfpivot = dfpivot.reset_index().rename_axis(None, axis=1) #index to columns

st.write(dfpivot)

st.subheader('Claims by Material by ARD')
dfretcumul = df5KMfilt.groupby(['RETURN_QUARTER', 'MATERIAL_C', 'ARD'])['ITEM_NB'].sum() \
    .groupby(level=0).cumsum().reset_index()
dfretcumul.rename({'RETURN_QUARTER': 'QUARTER', 'ITEM_NB':'Claim number'}, axis=1, inplace=True)

#fig, ax = plt.subplots()
fig = sns.relplot(data=dfretcumul, x="QUARTER", y="Claim number", col="MATERIAL_C", col_wrap=2, kind="line",
                  hue='ARD').tick_params(axis='x', rotation=90, labelsize=8)
#plt.xticks(rotation=90)
st.pyplot(fig)

st.subheader('Claims by Material by PLANT')
dfretcumul = df5KMfilt.groupby(['RETURN_QUARTER', 'MATERIAL_C', 'MANUFACT_PLANT_C'])['ITEM_NB'].sum() \
    .groupby(level=0).cumsum().reset_index()
dfretcumul.rename({'RETURN_QUARTER': 'QUARTER', 'ITEM_NB':'Claim number'}, axis=1, inplace=True)

#fig, ax = plt.subplots()
fig = sns.relplot(data=dfretcumul, x="QUARTER", y="Claim number", col="MATERIAL_C", col_wrap=2, kind="line",
                  hue='MANUFACT_PLANT_C').tick_params(axis='x', rotation=90, labelsize=8)
st.pyplot(fig)


st.subheader('Pairplot ARD (5 KM)')
lstutil = ["MATERIAL_C", "RETURN_QUARTER"]
dfpivot = pd.pivot_table(df5KMfilt, index=lstutil, columns=['ARD'],
               values=["ITEM_NB"], aggfunc=np.sum)

dfpivot.columns = dfpivot.columns.droplevel(0) #remove ITEM_NB
dfpivot = dfpivot.reset_index().rename_axis(None, axis=1) #index to columns
#dfpivot['BIRTH_QUARTER'] = dfpivot['BIRTH_QUARTER'].astype(str)
dfpivot.fillna(0, inplace=True)

g = sns.PairGrid(dfpivot, hue="MATERIAL_C")
g.map_diag(plt.hist)
#g.map_offdiag(sns.scatterplot)
g.map_offdiag(sns.scatterplot)
#g.map_lower(corrfunc)
g.add_legend()

st.pyplot(g)

