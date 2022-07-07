import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_moons
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D



X,y=make_moons(150,noise=0.18,random_state=1)
print(X.shape)
print(X)


sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, marker="o", s=25, edgecolor="k", legend=False
).set_title("wizualizacja wygenereowanych zbiorów")
plt.show()


iris=datasets.load_iris()
print(iris.values())

D=iris.data[:,:2]
t=iris.target




plt.scatter (D[:, 0], D[:, 1], c=t, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title('Wizualizacja zbioru IRIS')
plt.show()




X_train, X_test,y_train,y_test=train_test_split(X,y,train_size=0.7)
print(len(train_test_split(X,y,train_size=0.7)[0]))
print(len(train_test_split(X,y,train_size=0.7)[1]))
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test)

print(train_test_split(X,y,train_size=0.7))

D_train, D_test,t_train,t_test=train_test_split(D,t,train_size=0.7)

print('******iris*******')
print(D_train.shape)
print(D_test.shape)
print(D_test)
print(t_test)



clf=KNeighborsClassifier(n_neighbors=5) #zbiór losowy
clf.fit(X_train,y_train)
z=(clf.predict(X_test))
print('dopasowanie zbioru losowego',accuracy_score(y_test,z))


clf1=KNeighborsClassifier(n_neighbors=5)    #zbiór IRIS
clf1.fit(D_train,t_train)
z1=clf1.predict(D_test)
print('dopasowanie zboiru Iris',accuracy_score(t_test,z1))

print('preddykcja',z1)
print('rzeczywiste',y)
print('testowy2')
print(t_test)

plt.scatter (D_test[:, 0], D_test[:, 1], c=t_test, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title('Wizualizacja zbioru IRIS z wekotrami rzeczywistymi')
plt.show()

print(y)
print(clf.predict(X))

print(t)
print(clf1.predict(D))


sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=z, marker="o", s=25, edgecolor="k", legend=False
).set_title("wizualizacja przewidywanych wektorów zbioru losowego")
plt.show()



D=iris.data[:,:3]


fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection ="3d")

ax.scatter3D(
(D[:,0]),
(D[:,1]),
(D[:,2]),
c=t,
cmap=plt.cm.Set1,
edgecolor="g",
s=50)
plt.title("Zbiór IRIS 3D")
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
ax.set_zlabel('petal length')

plt.show()

D=iris.data[:,:3]
t=iris.target
D_train, D_test,t_train,t_test=train_test_split(D,t,train_size=0.7)
clf1=KNeighborsClassifier(n_neighbors=2)    #zbiór IRIS
clf1.fit(D_train,t_train)
z2=clf1.predict(D_test)
# print(z2)
# print(t)



fig = plt.figure(figsize=(9,6))
ax = plt.axes(projection ="3d")

ax.scatter3D(
(D_test[:,0]),
(D_test[:,1]),
(D_test[:,2]),
c=z2,
cmap=plt.cm.Set1,
edgecolor="g",
s=50)
plt.title("Zbiór IRIS 3D dla wektorów predykcji")
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
ax.set_zlabel('petal length')

plt.show()
print(D_test)