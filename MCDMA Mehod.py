from pymcdm import weights as mcdm_weigts
import numpy as np
from pymcdm import normalizations as norm
from pymcdm import methods as mcdm_methods
from pymcdm.helpers import rankdata
from pymcdm import correlations as corr

#PRBONLEM WYBORU PRACY
#ZAD NA OC 3)

a='wysokość pensji zasadniczej brutto', #profit
b='odległość od domu',#koszt
c='wymiar urlopu',#profit
d='premia',#profit
e='negatywnych opinii',#koszt
f='pozytywnych opinii',#profit

matrix=np.array([
    [10500,35,26,250,50,40],
    [5400,25,26,380,48,32],
    [7500,50,26,500,100,20],
    [7500,68,30,450,25,40],
    [4500,1,36,1000,2,3]
    ])

types=np.array([1,-1,1,1,-1,1])
weights=mcdm_weigts.entropy_weights(matrix)
#print(norm.normalize_matrix(matrix, norm.minmax_normalization, types))
topsis = mcdm_methods.TOPSIS()
#print(topsis(matrix, weights, types))
ranking = rankdata(topsis(matrix, weights, types), reverse=True)
#print(ranking)

#Zad na oc 4 Wybór auta

g='rok produkcji', #profit
h='przebieg',#koszt
i='liczba udziałów w kolizji',#koszt
j='prestiż marki w skali 1-6',#profit
k='wartość potrzebnych napraw',#koszt

matrix1=np.array([
    [2000,250000,3,5,250],
    [2004,84000,0,1,0],
    [2020,10,0,5,0],
    [2015,54000,0,3,0],
    [2008,140000,0,1,0]])

types1=np.array([1,-1,-1,1,-1])

weights1_sub =np.array([0.25,0.2,0.2,0.25,0.1])

weights1_ob=mcdm_weigts.entropy_weights(matrix1)
# print(weights1_sub)
# print(weights1_ob)
topsis = mcdm_methods.TOPSIS()
pref1 = topsis(matrix1, weights1_ob, types1)
copras = mcdm_methods.COPRAS()
pref2 = copras(matrix1, weights1_sub, types1)
similarity = corr.weighted_spearman(pref1, pref2)
ranking = rankdata(topsis(matrix1, weights1_ob, types1), reverse=True)
ranking2 = rankdata(topsis(matrix1, weights1_sub, types1), reverse=True)
print(ranking)
print(ranking2)
print(pref1)
print('******')
print(pref2)
print(similarity)