import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_moons
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print('4a- generowanie  150 wartosci')


X,y=make_moons(150,noise=0.48,random_state=1)
print(X,y)


sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, marker="o", s=25, edgecolor="k", legend=False
).set_title("wizualizacja wygenereowanych zbiorów")
plt.show()


iris=datasets.load_iris()
D=iris.data[:,:2]
t=iris.target



plt.scatter (D[:, 0], D[:, 1], c=t, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title('Wizualizacja zbioru IRIS')
plt.show()

print('b dzielenie danych na probki uczące i testujące')


X_train, X_test,y_train,y_test=train_test_split(X,y,train_size=0.7)
D_train, D_test,t_train,t_test=train_test_split(D,t,train_size=0.7)



clf=KNeighborsClassifier(n_neighbors=4)
clf.fit(X_train,y_train)
z=(clf.predict(X))
print(z)
print(y_test)

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=z, marker="o", s=25, edgecolor="k", legend=False
).set_title("wizualizacja wygenereowanych zbiorów z przewidywanym wektorem")
plt.show()

clf1=KNeighborsClassifier(n_neighbors=4)
clf1.fit(D_train,t_train)
z1=clf1.predict(D)

print(D_test)
print(clf1.predict(D_test))
print('testowy2')
print(t_test)

plt.scatter (D[:, 0], D[:, 1], c=z1, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title('Wizualizacja zbioru IRIS z przewidywanym wektorem')
plt.show()




print(accuracy_score(y,clf.predict(X)))
print(accuracy_score(t,clf1.predict(D)))

print(y)
print(clf.predict(X))

print(t)
print(clf1.predict(D))

print('g')
D=iris.data[:,:2]
t=iris.target
print(iris.data)