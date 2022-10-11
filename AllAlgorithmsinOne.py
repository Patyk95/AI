from cgi import print_arguments
from dataclasses import dataclass
import sklearn.datasets
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier 

d=sklearn.datasets.load_iris()

def bagging(data,s_probe):
    x=data['data']
    y=data['target']
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.75)
    model=BaggingClassifier()
    model.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    f1=accuracy_score(ytest,pred)
    f2=mean_absolute_error(ytest,pred)
    f3=mean_absolute_percentage_error(ytest,pred)
    f3.round(2)
    print('PROCENT DOPASOWANIA MODELU BaggingClassAlg TO: ',f1)
    print('BEZWGLĘDNY ABSOLUTNY BŁĄD MODELU BaggingClassAlg TO: ',f2)
    print('BEZWZGLĘDNY PROCENOWY BŁĄD MODELU BaggingClassAlg TO:',f3)


def decission_tree(data,s_probe):
    x=data['data']
    y=data['target']
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.3)
    model= DecisionTreeClassifier()
    clas=model.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    f1=accuracy_score(ytest,pred)
    f2=mean_absolute_error(ytest,pred)
    f3=mean_absolute_percentage_error(ytest,pred)
    f3.round(2)
    print('PROCENT DOPASOWANIA MODELU DRZEWA DECYZYJNEGO TO: ',f1)
    print('BEZWGLĘDNY ABSOLUTNY BŁĄD MODELU DRZEWA DECYZYJNEGO TO: ',f2)
    print('BEZWZGLĘDNY PROCENOWY BŁĄD MODELU DRZEWA DECYZYJNEGO TO:',f3)


def knn(data,n,s_probe):
    x=data['data']
    y=data['target']
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=s_probe)
    model=KNeighborsClassifier(n_neighbors=n)
    model.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    f1=accuracy_score(ytest,pred)
    f2=mean_absolute_error(ytest,pred)
    f3=mean_absolute_percentage_error(ytest,pred)
    f3.round(2)
    print('PROCENT DOPASOWANIA MODELU KNN TO: ',f1)
    print('BEZWGLĘDNY ABSOLUTNY BŁĄD MODELU KNN TO: ',f2)
    print('BEZWZGLĘDNY PROCENOWY BŁĄD MODELU KNN TO:',f3)


def linearregr(data,s_probe):
    x=data['data']
    y=data['target']
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=s_probe)
    model=LinearRegression()
    model.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    #f1=accuracy_score(ytest,pred)
    f2=mean_absolute_error(ytest,pred)
    f3=mean_absolute_percentage_error(ytest,pred)
    # f3.round(2)
    #print('PROCENT DOPASOWANIA MODELU REGRESJI LINIOWEJ TO: ',f1)
    print('BEZWGLĘDNY ABSOLUTNY BŁĄD MODELU REGRESJI LINIOWEJ TO: ',f2)
    print('BEZWZGLĘDNY PROCENOWY BŁĄD MODELU REGRESJI LINIOWEJ TO:',f3)


def logisticregression(data,s_probe):
    x=data['data']
    y=data['target']
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=s_probe)
    model = LogisticRegression()
    model.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    f1=accuracy_score(ytest,pred)
    f2=mean_absolute_error(ytest,pred)
    f3=mean_absolute_percentage_error(ytest,pred)
    f3.round(2)
    print('PROCENT DOPASOWANIA MODELU REGRESJI LOGISTYCZNEJ TO: ',f1)
    print('BEZWGLĘDNY ABSOLUTNY BŁĄD MODELU REGRESJI LOGISTYCZNEJ TO: ',f2)
    print('BEZWZGLĘDNY PROCENOWY BŁĄD MODELU REGRESJI LOGISTYCZNEJ TO:',f3)


def naivebase(data,s_probe):
    x=data['data']
    y=data['target']
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=s_probe)
    model = GaussianNB()
    model.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    f1=accuracy_score(ytest,pred)
    f2=mean_absolute_error(ytest,pred)
    f3=mean_absolute_percentage_error(ytest,pred)
    f3.round(2)
    print('PROCENT DOPASOWANIA MODELU NAIVE BASE TO: ',f1)
    print('BEZWGLĘDNY ABSOLUTNY BŁĄD MODELU NAIVE BASE TO: ',f2)
    print('BEZWZGLĘDNY PROCENOWY BŁĄD MODELU NAIVE BASE TO:',f3)


def randomforest(data,s_probe):
    x=data['data']
    y=data['target']
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=s_probe)
    model = RandomForestRegressor()
    model.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    #f1=accuracy_score(ytest,pred)
    f2=mean_absolute_error(ytest,pred)
    f3=mean_absolute_percentage_error(ytest,pred)
    f3.round(2)
    #print('PROCENT DOPASOWANIA MODELU RANDOM FOREST TO: ',f1)
    print('BEZWGLĘDNY ABSOLUTNY BŁĄD MODELU RANDOM FOREST TO: ',f2)
    print('BEZWZGLĘDNY PROCENOWY BŁĄD MODELU RANDOM FOREST TO:',f3)

def SMV(data, s_probe):
    x=data['data']
    y=data['target']
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=s_probe)
    clasf=svm.SVC()
    clasf.fit(xtrain,ytrain)
    w=clasf.predict(xtest)
    f1=accuracy_score(ytest,w)
    f2=mean_absolute_error(ytest,w)
    f3=mean_absolute_percentage_error(ytest,w)
    print('PROCENT DOPASOWANIA MODELU SVM TO: ',f1)
    print('BEZWGLĘDNY ABSOLUTNY BŁĄD MODELU SVM TO: ',f2)
    print('BEZWZGLĘDNY PROCENOWY BŁĄD MODELU SVM TO:',f3)

def Stohastic_alg(data,s_probe):
    x=data['data']
    y=data['target']
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=s_probe)
    model=SGDClassifier()
    model.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    f1=accuracy_score(ytest,pred)
    f2=mean_absolute_error(ytest,pred)
    f3=mean_absolute_percentage_error(ytest,pred)
    print('PROCENT DOPASOWANIA MODELU Stochastycznego TO: ',f1)
    print('BEZWGLĘDNY ABSOLUTNY BŁĄD MODELU Stochastycznego TO: ',f2)
    print('BEZWZGLĘDNY PROCENOWY BŁĄD MODELU Stochastycznego TO:',f3)


d=sklearn.datasets.load_iris()

#knnr(d,5,0.45)
#bagging(d,0.5)
#decission_tree(d,0.5)
#linearregr(d,0.45)
#logisticregression(d,0.48)
#naivebase(d,0.38)
#randomforest(d,0.38)
#SMV(d,0.4)
Stohastic_alg(d,0.30)