# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 21:41:05 2019

@author: pparepal
"""

import logging
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from bottle import Bottle,route, run, post, request,hook, response, HTTPResponse
import os
import traceback
import numpy as np
import bottle

#Here We are using Bottle framework


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#port = int(os.getenv("PORT"))
app = bottle.default_app()


def traderDist(x, y):
    m1=(x[0]-y[0])**2
    m2=0 if x[1]==y[1] else 10000000
    m3=0 if x[2]==y[2] else 10000
    m4=(x[3]-y[3])**2
    m5=(x[4]-y[4])**2
    return(m1+m2+m3+m4+m5)
    
'''
def main(req: func.HttpRequest) -> func.HttpResponse:

It is used to Inference at runtime

'''


@route('/predict', method='POST')
def predict():
    try:
        logging.info('Python HTTP trigger function processed a request.')

            #Get the JSON data
        delivery = request.json['delivery_from'] ##Date in m/d/yyyy
        qty = request.json['qty']
        material = request.json['material']
        counter_party = request.json['counter_party']

        df=pd.read_csv(r'D:\Users\Default User\Desktop\ContractDummyData.csv',converters={'QTY':float, 'Material':str,'Counter Party':str,'Date From':str})
        df.dropna(how='all',inplace=True)
        le_cp=LabelEncoder()
        le_cp.fit(df['Counter Party'].values)

        le_m=LabelEncoder()
        le_m.fit(df['Material'].values)


        X=df[['Date From','Date To','QTY','Material','Counter Party']]
        X['Material']=le_m.transform(df['Material'].values)
        X['Counter Party']=le_cp.transform(df['Counter Party'].values)
        X['Delivery_Period_Month']=[float(x.split('/')[0]) for x in X['Date From'].values]
        X['Delivery_Period_Year']=[float(x.split('/')[2]) for x in X['Date From'].values]
        X.drop(['Date From','Date To'],axis=1,inplace=True)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto', metric=traderDist)

        nbrs.fit(X.values)

        logging.info('Model trained')

        input_data = pd.DataFrame({'QTY': [qty], 'Material':[material], 'Counter Party': [counter_party],'Delivery':[delivery] },columns=['QTY', 'Material', 'Counter Party','Delivery'])
        input_data['QTY']=[float(x) for x in input_data['QTY'].values]

        input_data['Counter Party']=le_cp.transform(input_data['Counter Party'].values)
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
            print("wrong")
    except:
        return HTTPResponse(
                status=400,
                body= 'Please pass delivery_from, qty, material, counter_party on the query string or in the request body') 
        
        

@route('/test', method='POST')
def test():
	return{"testing...."}

    
    

if __name__ == '__main__':
  #run(host='0.0.0.0', port=port)
  app.run(debug=True)