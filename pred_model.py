#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:02:30 2018

@author: Sanjeet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import gc
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from sklearn.linear_model import LinearRegression as lr

def p_q(s):
    lag_acf = acf(s, nlags=50)
    lag_pacf = pacf(s, nlags=50, method='ols')
    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(s)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(s)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(s)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(s)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    return
## Extract debit card related data in data_debit
data_rbi1 = pd.read_excel("RBIB1.xlsx", header=1)
data_rbi1 = data_rbi1.reset_index()
data_rbi1 = data_rbi1.drop(columns= "index")
data_rbi1 = data_rbi1.drop(0)

data_debit = data_rbi1[['Month/Year','     5.2  Debit Cards','Unnamed: 52','            5.2.1  Usage at ATMs', 'Unnamed: 54',  '            5.2.2  Usage at POS', 'Unnamed: 56']]
data_debit.columns=['Month/Year','Debit_Cards_Volume','Debit_Cards_Value','ATM_Volume','ATM_Value','POS_Volume','POS_Value']
data_debit=data_debit.drop([167,168])
# fill missing values 
data_debit.ATM_Volume=data_debit.ATM_Value.fillna(0)
data_debit.ATM_Volume=data_debit.ATM_Value.apply(lambda x:x if x!='–' else 0)

data_debit.ATM_Value=data_debit.ATM_Value.fillna(0)
data_debit.ATM_Value=data_debit.ATM_Value.apply(lambda x:x if x!='–' else 0)

# Make date features
data_debit['Month']=data_debit['Month/Year'].apply(lambda x:x.split('-')[0])
data_debit['Year']=data_debit['Month/Year'].apply(lambda x:x.split('-')[1])
month_dict={
        'Jan' : '01',
        'Feb' : '02',
        'Mar' : '03',
        'Apr' : '04',
        'May' : '05',
        'Jun' : '06',
        'Jul' : '07',
        'Aug' : '08',
        'Sep' : '09', 
        'Oct' : '10',
        'Nov' : '11',
        'Dec' : '12'
}
data_debit['Month']=data_debit['Month'].apply(lambda x:month_dict[x])
data_debit['Date']=data_debit.apply(lambda x: str(x.Month)+'/'+str(x.Year),axis=1)
data_debit['Date']=pd.to_datetime(data_debit.Date,format='%m/%Y')

#arima model

#
#model = ARIMA(train, order=(2, 0, 50))  
#results_ARIMA = model.fit(disp=-1)  
#plt.plot(s)
#plt.plot(results_ARIMA.fittedvalues, color='red')


# linear regression
def lr_model(col_name):
    min_date=data_debit[(data_debit.ATM_Value>0)]['Date'].min()
    max_date=data_debit.Date.iloc[25]
    train=pd.DataFrame()
    train['counter']=(data_debit.Date-min_date).dt.days
    train['month']=data_debit.Month
    train['year']=data_debit.Year
    train['target']=data_debit[col_name].values
    train.index=data_debit.Date
    train_dates=(train.index>min_date) &(train.index<max_date)
    test_dates=(train.index > max_date)
    model=lr()
    model.fit(X=train[train_dates][['counter','month','year']].values,y=train[train_dates]['target'].values)
    train['predict']=model.predict(train[['counter','month','year']])
    train[['target','predict']].plot(figsize=(6, 4))
    print('RMSE: '+str(np.sqrt(np.mean((train['target'][test_dates]-train['predict'][test_dates])**2))))
    return  

def log_lr_model(col_name):
    min_date=data_debit.Date.min()
    max_date=data_debit.Date.iloc[25]
    train=pd.DataFrame()
    train['counter']=(data_debit.Date-min_date).dt.days
    train['month']=data_debit.Month
    train['year']=data_debit.Year
    train['target']=data_debit[col_name].values
    train.index=data_debit.Date
    train_dates=(train.index>min_date) &(train.index<max_date)
    test_dates=(train.index > max_date)
    model=lr()
    model.fit(X=train[train_dates][['counter','month','year']].values,y=np.log(train[train_dates]['target'].values))
    train['predict']=np.exp(model.predict(train[['counter','month','year']]))
    train[['target','predict']].plot()
    print('RMSE: '+str(np.sqrt(np.mean((train['target'][test_dates]-train['predict'][test_dates])**2))))
    return

lr_model('ATM_Volume')
lr_model('ATM_Value')
log_lr_model('POS_Value')
log_lr_model('POS_Volume')

