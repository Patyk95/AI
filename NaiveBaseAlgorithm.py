import numpy as np
from pandas import array
import sklearn.datasets
from sklearn.metrics import accuracy_score, mean_absolute_error,median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#NaiveBase alg

data= sklearn.datasets.load_diabetes()
x=(data['data'])
y=(data['target'])
#print(x.shape)
#print(y.shape)

xtrain,xtest,ytrain,ytest= train_test_split(x,y,train_size=0.4)
#print(xtest.shape)
#print(ytest.shape)
model=GaussianNB()
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
#print(pred)

print(accuracy_score(ytest,pred))
print(mean_absolute_error(ytest,pred))
print(median_absolute_error(ytest,pred))
#print(ytest)