import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.ensemble import BaggingClassifier

data=datasets.load_wine()
x=data['data']
y=data['target']
print(x)
print(y)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.75)

model=BaggingClassifier()
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
print(accuracy_score(ytest,pred))

