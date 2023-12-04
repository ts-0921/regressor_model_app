import pandas as pd 
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import codecs
import datetime

with codecs.open("商談課題_進捗.csv", "r", "Shift-JIS", "ignore") as file:
    df= pd.read_table(file, delimiter=",")
df=pd.DataFrame(df)

with codecs.open("営業所情報.csv", "r", "Shift-JIS", "ignore") as file:
    df2= pd.read_table(file, delimiter=",")
df2=pd.DataFrame(df2)

df['課題/商談年月']=pd.to_datetime(df['課題/商談年月'])
df['課題/商談: 作成日']=pd.to_datetime(df['課題/商談: 作成日'])
df['課題/商談: 最終更新日']=pd.to_datetime(df['課題/商談: 最終更新日'])
df['課題/商談: 作成年']=df['課題/商談: 作成日'].dt.strftime('%Y')

df_M=pd.merge(df,df2,left_on='課題/商談: 所有者 ロール',right_on='営業所名',how='left')
df_HP=df_M.loc[df_M['本部名']=='第一営業本部']
df_GP=df_M.loc[(df_M['本部名']=='GP東日本営業本部')|(df_M['本部名']=='GP西日本営業本部')]

resorce=df2[['営業所コード','営業所名','営業所人数']]

df_tmp_HP=df_HP.groupby(['課題/商談: 作成年','営業所コード','営業所名']).agg({'課題/商談: 課題/商談No':'count'}).reset_index()
df_tmp_HP.rename(columns={'課題/商談: 課題/商談No':'課題/商談件数'},inplace=True)     
df_tmp_HP.rename(columns={'課題/商談: 作成年':'作成年'},inplace=True)     
df_tmp_HP=df_tmp_HP.sort_values(['作成年','営業所コード'])
df_tmp_HP=pd.merge(df_tmp_HP,resorce,on=['営業所コード','営業所名'],how='left')

df_tmp_GP=df_GP.groupby(['課題/商談: 作成年','営業所コード','営業所名']).agg({'課題/商談: 作成日':'count'}).reset_index()
df_tmp_GP.rename(columns={'課題/商談: 作成日':'課題/商談件数'},inplace=True)
df_tmp_GP.rename(columns={'課題/商談: 作成年':'作成年'},inplace=True)  
df_tmp_GP=df_tmp_GP.sort_values(['作成年','営業所コード'])
df_tmp_GP=pd.merge(df_tmp_GP,resorce,on=['営業所コード','営業所名'],how='left')

df_tmp2=df_HP.pivot_table(index=['課題/商談: 作成年','営業所名'], columns='Win/Lost', 
                            values='課題/商談: 課題/商談No',aggfunc='count',fill_value=0).reset_index()
df_tmp2.rename(columns={'営業所名':'HP 営業所名'},inplace=True)
df_tmp2.rename(columns={'課題/商談: 作成年':'作成年'},inplace=True)

df_tmp3=df_GP.pivot_table(index=['課題/商談: 作成年','営業所名'], columns='Win/Lost', 
                            values='課題/商談: 課題/商談No',aggfunc='count',fill_value=0).reset_index()
df_tmp3.rename(columns={'営業所名':'GP 営業所名'},inplace=True)
df_tmp3.rename(columns={'課題/商談: 作成年':'作成年'},inplace=True)

df_tmp_HP.rename(columns={'営業所名':'HP 営業所名'},inplace=True)
df_M_HP=pd.merge(df_tmp_HP,df_tmp2,on=['作成年','HP 営業所名'],how='left')

df_tmp_GP.rename(columns={'営業所名':'GP 営業所名'},inplace=True)
df_M_GP=pd.merge(df_tmp_GP,df_tmp3,on=['作成年','GP 営業所名'],how='left')
df_M_GP=df_M_GP.fillna(0)

X=df_M_GP[['課題/商談件数','営業所人数']]
y=df_M_GP['Win']

model_GP = LinearRegression()
model_GP.fit(X, y)

X=df_M_HP[['課題/商談件数','営業所人数']]
y=df_M_HP['Win']

model_HP = LinearRegression()
model_HP.fit(X, y)

st.title('成約数予測アプリ')

#st.header('HP目標成約数')
syoudan_HP = st.sidebar.number_input('HP商談件数を入力', value=0.0)
ninzu_HP = st.sidebar.number_input('HP営業所人数を入力', value=0.0)

pred_HP=model_HP.predict([[syoudan_HP,ninzu_HP]])
st.header(f'HP 目標成約数: {round(pred_HP[0])}')



#st.header('GP目標成約数')
syoudan_GP = st.sidebar.number_input('GP商談件数を入力', value=0.0)
ninzu_GP = st.sidebar.number_input('GP営業所人数を入力', value=0.0)

pred_GP=model_GP.predict([[syoudan_GP,ninzu_GP]])
st.header(f'GP 目標成約数: {round(pred_GP[0])}')

