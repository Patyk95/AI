import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


b= load_diabetes()
df = pd.DataFrame(data=b.data, columns=b.feature_names)


c=df[['age','bmi','bp']]
X=np.array(c)
#print(X)
Y=np.array(b['target'])
#print(Y)


plt.subplot(131)
X1=df[['age']]
y=Y
plt.scatter(X1,y,color='r')
plt.ylabel('ZALEŻNE OD Y')

plt.subplot(132)
X2=df[['bmi']]
y=Y
plt.xlabel('ZALEŻE OD X')
plt.scatter(X2,y)

plt.subplot(133)
X3=df[['bp']]
y=Y
plt.scatter(X3,y,color='y')
# plt.show()



X_train, X_test,y_train,y_test=train_test_split(X,Y,train_size=0.8)



X=np.array(c)
Y=np.array(b['target'])
reg = LinearRegression().fit(X_train, y_train)
#print(reg.score(X_train, y_train))



y_true = y_test
y_pred = z=(reg.predict(X_test))
#print(y_pred)
#print(y_true)
# print(mean_squared_error(y_true, y_pred))
# print(mean_squared_error(y_true, y_pred, squared=False))




c=df[['age','bmi']]

# print(df[['bp','age','sex']])

c=(list(b.feature_names))
# print(c)
c1=c.copy()         #kopiowanie list
d=c.pop(1)          #sciągnięcie 1
c.append(d)         #dodawanie na koncu
d=c[0:2]
e=c[2:]
print(d)
print(e)

for i in e:
    k=1
    d.append(i)
    print(d)
    if len(i) <len(e):
        print(d)
        c = df[d]
        print(c)
        X = np.array(d)
        Y = np.array(b['target'])
        plt.subplot(131)
        X1=df[['age']]
        y=Y
        plt.scatter(X1,y,color='r')
        plt.ylabel('ZALEŻNE OD Y')
        plt.subplot(132)
        X2=df[['bmi']]
        y=Y
        plt.xlabel('ZALEŻE OD X')
        plt.scatter(X2,y)
        plt.subplot(133)
        X3=df[['bp']]
        y=Y
        plt.scatter(X3,y,color='y')
        plt.show()
        X = np.array(c)
        Y = np.array(b['target'])
        X_train, X_test,y_train,y_test=train_test_split(X,Y,train_size=0.8)
        reg = LinearRegression().fit(X_train, y_train)
        print(mean_squared_error(y_true, y_pred))
        print(mean_squared_error(y_true, y_pred, squared=False))
        f=str(k)
        k+=1
        d.remove(i)
    if len(i) == len(e):
        print('zakończono działanie')
        sys.exit()



