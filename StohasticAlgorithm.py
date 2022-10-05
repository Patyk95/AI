import sklearn.datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

datta=sklearn.datasets.load_iris()
#print(datta)
x=datta['data']
y=datta['target']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.4)
model=SGDClassifier()
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
print(accuracy_score(ytest,pred))