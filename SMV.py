import numpy as np
from pandas import array
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm

data = sklearn.datasets.load_iris()
x=data['data']
y=data['target']
#print(y)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.35)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

clasf=svm.SVC()
clasf.fit(xtrain,ytrain)
print(ytest)
w=clasf.predict(xtest)
print(w)

accuracy= accuracy_score(ytest,w)
print(accuracy)