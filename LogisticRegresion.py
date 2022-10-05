import sklearn.datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

model=sklearn.datasets.load_iris()


x=model['data']
y=model['target']

#print(x)
#print(y)


xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.35)

clasif=LogisticRegression()
clasif.fit(xtrain,ytrain)
pred =clasif.predict(xtest)

print(pred)
print('efectywne:' ,ytest)

print(accuracy_score(ytest,pred))


plt.subplot(1,2,1)
plt.scatter(xtest[:,0],xtest[:,1],c=ytest)
plt.subplot(1,2,2)
plt.scatter(xtest[:,0],xtest[:,1],c=pred)
plt.show()
