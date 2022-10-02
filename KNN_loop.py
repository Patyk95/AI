import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from xlswriter.workbook import Workbook
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#ładownanie  zbioru danych wino datasets skilearn
dane=sklearn.datasets.load_wine()

#def to_excel():
    #dane = sklearn.datasets.load_wine()
    #x = dane['data']
    #workbook = Workbook('dane.xlsx')
    #sheet=workbook._add_sheet('sheet1')
    #for i,row in enumerate(x):
    #    for j,value  in enumerate(row):
    #        sheet.write(i,j,row[j])
    #workbook.close()

#wyodrębnienie próbek


def KNN (n):
    x = dane['data']
    y = dane['target']
    x = x[:, 0:2]
    xtest, xtrain, ytest, ytrain = train_test_split(x, y, test_size=35, random_state=0)
    klas= KNeighborsClassifier(n_neighbors=n)
    klas.fit(xtrain,ytrain)
    k_predict= klas.predict(xtest)
    #print(k_predict)
    #print(ytest)
    v=accuracy_score(ytest,k_predict)
    #print(v)

KNN(6)

def acc_knn(n):
    dane = sklearn.datasets.load_wine()
    c = []
    cont=0
    for i in range(13):
        for j in range(13):
            if i>=j:
                pass
            if i<j:
                x = dane['data']
                y = dane['target']
                x1 =x[:,i]
                x2= x[:,j]
                x =np.stack((x1,x2),axis=1)
                xtest, xtrain, ytest, ytrain = train_test_split(x, y, test_size=35, random_state=0)
                klas = KNeighborsClassifier(n_neighbors=n)
                klas.fit(xtrain, ytrain)
                k_predict= klas.predict(xtest)
                v = accuracy_score(ytest, k_predict)
                c.append(v)
                d=max(c)
                if v>=d:
                    cont+=1
                    plt.plot()
                    plt.scatter(x1, x2, marker='o', c=y)
                    plt.title('WYkres dla danych z couln ' + str(i) +' oraz ' + str(j))
                    plt.show()
                    print('\nNajwiększe współczynniki dopasowania wystęują dla  danych z column '+ str(i) +' oraz ' + str(j)+ ' i wynosi '+ str(v))
    print('\nPośród wymienionych współczynników największy wynosi ' + str(d))

acc_knn(7)