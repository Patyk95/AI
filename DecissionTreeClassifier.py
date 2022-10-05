import numpy as np
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data= sklearn.datasets.load_iris()

x=data['data']
y=data['target']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.3)

model= DecisionTreeClassifier()

clas=model.fit(xtrain,ytrain)
pred=model.predict(xtest)
print(pred)
print(accuracy_score(ytest,pred))

