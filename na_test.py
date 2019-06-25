# -*- coding: utf-8 -*-
"""
Created on Fri May  3 18:08:52 2019

@author: mebandar
"""

import logging
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from bottle import Bottle,route, run, post, request,hook, response, HTTPResponse
import os
import traceback
import numpy as np

#Inorder to connect our python script from our UI we should have some medeium, That medieum is noting but out bottle. 

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
    

@route('/predict', method='POST')
def predict():
    try:
        logging.info('Python HTTP trigger function processed a request.')
        
        #Get the JSON data
        delivery = request.json['delivery_from'] ##Date in m/d/yyyy
        qty = request.json['qty']
        material = request.json['material']
        counter_party = request.json['counter_party']
        
        df=pd.read_csv("data.csv",converters={'Quantity':float, 'Material':str,'New Counterparty':str,'Del From':str})
        df.dropna(how='all',inplace=True)
        X=df[['Del From','Del To','Quantity','Material','New Counterparty']]
        le_cp=LabelEncoder()
        le_cp.fit(df['New Counterparty'].values)

        le_m=LabelEncoder()
        le_m.fit(df['Material'].values)

        X['Material']=le_m.transform(df['Material'].values)
        X['New Counterparty']=le_cp.transform(df['New Counterparty'].values)

    
        X['Delivery_Period_Month']=[float(x.split('/')[0]) for x in X['Del From'].values]
        X['Delivery_Period_Year']=[float(x.split('/')[2]) for x in X['Del From'].values]
    
        X.drop(['Del From','Del To'],axis=1,inplace=True)
        
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=traderDist)
    
        nbrs.fit(X.values)
    
        logging.info('Model trained')
    
        input_data = pd.DataFrame({'Quantity': [6000.0], 'Material':['10010'], 'New Counterparty': ['100000475'],'Delivery':['2019-05-01 00:00:00'] },columns=['Quantity', 'Material', 'New Counterparty','Delivery'])
        input_data['Quantity']=[float(x) for x in input_data['Quantity'].values]


        input_data['New Counterparty']=le_cp.transform(input_data['New Counterparty'].values)
        input_data['Material']=le_m.transform(input_data['Material'].values)
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
        

@route('/test', method='POST')
def test():
	return{"testing...."}

    
    

if __name__ == '__main__':
   run(host='0.0.0.0', port=port)

