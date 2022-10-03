import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#cancer breast datasets


data= sklearn.datasets.load_breast_cancer()     #load dataset
#print(data)                                    #to check features
#print(data['feature_names']) 
x=data['data']
#print(x.shape)
y=data['target']                                #to get clasificated values
#print(y.shape)


xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.45,random_state=0)  #split data into test and train probes

regr=RandomForestRegressor(n_estimators = 1000,random_state=0)                  #set up regresion options
regr.fit(xtrain,ytrain)                                                         #to train our model
pred = regr.predict(xtest)
#print(pred)                                      #print predictions
err=abs(pred-ytest)                               #chceck absolute error for each features
print(err)
#print(ytest)

print(round(np.mean(err),2))                    # to check summ error




