# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:07:27 2019

@author: pparepal
"""

import logging
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import os
from bottle import Bottle, run

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
port = int(os.getenv("PORT"))
app = Bottle()

def traderDist(x, y):
    m1=(x[0]-y[0])**2
    m2=0 if x[1]==y[1] else 10000000
    m3=0 if x[2]==y[2] else 10000
    m4=(x[3]-y[3])**2
    m5=(x[4]-y[4])**2
    return(m1+m2+m3+m4+m5)
    
    @app.route('/predict', method='POST')


def predict():
    try:
        
        
        logging.info('Python HTTP trigger function processed a request.')
        
        #Get the JSON data
        delivery = request.json['delivery_from'] ##Date in m/d/yyyy
        qty = request.json['qty']
        material = request.json['material']
        counter_party = request.json['counter_party']
        
        
        input_data = pd.DataFrame({'QTY': [qty], 'Material':[material], 'Counter Party': [counter_party],'Delivery':[delivery] },columns=['QTY', 'Material', 'Counter Party','Delivery'])
        
        df=pd.read_csv(r'D:\Users\Default User\Desktop\ContractDummyData.csv',converters={'QTY':float, 'Material':str,'Counter Party':str,'Date From':str})

        df.dropna(how='all',inplace=True)

        X=df[['Date From','Date To','QTY','Material','Counter Party']]
        dict1=dict(zip(X['Material'].unique(),np.arange(X['Material'].unique().shape[0])))
        if input_data.loc[:,'Material'][0] in dict1:
            pass
        else:
            dict1[input_data.loc[:,'Material'][0]]=X['Material'].unique().shape[0]


        X['Material'].replace(dict1,inplace=True)

        dict2=dict(zip(X['Counter Party'].unique(),np.arange(X['Counter Party'].unique().shape[0])))
        if input_data.loc[:,'Counter Party'][0] in dict2:
            pass
        else:
            dict2[input_data.loc[:,'Counter Party'][0]]=X['Counter Party'].unique().shape[0]
        X['Counter Party'].replace(dict2,inplace=True)

        X['Delivery_Period_Month']=[float(x.split('/')[0]) for x in X['Date From'].values]
        X['Delivery_Period_Year']=[float(x.split('/')[2]) for x in X['Date From'].values]

        X.drop(['Date From','Date To'],axis=1,inplace=True)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=traderDist)
    
        nbrs.fit(X.values)
        logging.info('Model trained')
        input_data['QTY']=[float(x) for x in input_data['QTY'].values]


        input_data['Material'].replace(dict1,inplace=True)

        input_data['Counter Party'].replace(dict2,inplace=True)
        input_data['Delivery_Period_Month']=[float(x.split('/')[0]) for x in input_data['Delivery'].values]
        input_data['Delivery_Period_Year']=[float(x.split('/')[2]) for x in input_data['Delivery'].values]
        input_data.drop(['Delivery'],axis=1,inplace=True)

        logging.info(f'{input_data.values[0]}')
    
        logging.info(f'{X.values[0]}')

        distances, indices = nbrs.kneighbors(input_data.values[0].reshape(1, -1))
        if all(v is not None for v in [delivery, qty, material, counter_party]):
            return{df.loc[indices[0][0]].to_json()}
        else:
            return{"Please pass delivery_from, qty, material, counter_party on the query string or in the request body"}
    except:
        tb=traceback.format_exc()
        return{"traceback":tb}
    
    
@app.route('/test', method='POST')

def test():
	return{"testing...."}