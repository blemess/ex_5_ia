from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.io.arff import loadarff
import pandas as pd

X, y = load_iris(return_X_y=True)

#test_size= 0.5 | random_state = 0 
# Precision Score :94.66666666666667%
#F1 Score :94.66666666666667%
#----------------------------
#test_size= 0.8 | random_state = 42 
# Precision Score :95%
#F1 Score :95%
#----------------------------
#test_size= 0.2 | random_state = 42 
# Precision Score :100%
#F1 Score :100%
#----------------------------
#test_size= 0.1 | random_state = 42 
# Precision Score :100%
#F1 Score :100%
#----------------------------
#test_size= 0.3 | random_state = 42 
# Precision Score :97.77777777777777%
#F1 Score :97.77777777777777%

print('#----------------------------')
print('GaussianNB')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
precision =precision_score(y_test,y_pred,average='micro')
f1Score= f1_score(y_test,y_pred,average='micro')
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print(confusion_matrix(y_test,y_pred))
print(f'Precision Score :{precision * 100}%')
print(f'F1 Score :{f1Score * 100}%')

print('#----------------------------')
print('KNeighbors')

# default values:
# n_neighbors =  4; algorithm = auto; leaf_size = 30; p = 2; metric = minkowski

def knn(X_train, y_train, y_test, n, alg, p, metric):
    print('\n\nn_neighbors, algorithm, p and metric = ', n, alg, p, metric)
    neigh = KNeighborsClassifier(n_neighbors= n, algorithm=alg, p = p, metric=metric)
    neigh.fit(X_train,y_train)
    y_predK = neigh.predict(X_test)
    precision =precision_score(y_test,y_predK,average='micro')
    f1Score= f1_score(y_test,y_predK,average='micro')
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_predK).sum()))
    print(confusion_matrix(y_test,y_predK))
    print(f'Precision Score :{precision * 100}%')
    print(f'F1 Score :{f1Score * 100}%')

knn(X_train,y_train, y_test, 5, 'auto', 2, 'minkowski')
knn(X_train,y_train, y_test, 4, 'auto', 2, 'minkowski')
knn(X_train,y_train, y_test, 3, 'auto', 2, 'minkowski')
knn(X_train,y_train, y_test, 2, 'auto', 2, 'minkowski')

print('\nBEST n_neighbors = 4')

knn(X_train,y_train, y_test, 4, 'auto', 2, 'minkowski')
knn(X_train,y_train, y_test, 4, 'ball_tree', 2, 'minkowski')
knn(X_train,y_train, y_test, 4, 'kd_tree', 2, 'minkowski')
knn(X_train,y_train, y_test, 4, 'brute', 2, 'minkowski')

print('no diffs :(')

knn(X_train,y_train, y_test, 4, 'auto', 1, 'minkowski')
knn(X_train,y_train, y_test, 4, 'auto', 2, 'minkowski')

print('\nBEST p = 2')


knn(X_train,y_train, y_test, 4, 'auto', 2, 'minkowski')   #97.34
knn(X_train,y_train, y_test, 4, 'auto', 2, 'euclidean')   #97.34
knn(X_train,y_train, y_test, 4, 'auto', 2, 'cityblock')   #96.46
knn(X_train,y_train, y_test, 4, 'auto', 2, 'cosine')      #96.46
knn(X_train,y_train, y_test, 4, 'auto', 2, 'chebyshev')   #98.23 
#|-> measures distance between two points as the maximum difference over any of their axis value
knn(X_train,y_train, y_test, 4, 'auto', 2, 'canberra')    #94.69
knn(X_train,y_train, y_test, 4, 'auto', 2, 'correlation') #96.46
knn(X_train,y_train, y_test, 4, 'auto', 2, 'manhattan')   #96.46
knn(X_train,y_train, y_test, 4, 'auto', 2, 'kulsinski')   #30.97
knn(X_train,y_train, y_test, 4, 'auto', 2, 'l2')          #97.34
knn(X_train,y_train, y_test, 4, 'auto', 2, 'hamming')     #82.30
knn(X_train,y_train, y_test, 4, 'auto', 2, 'matching')    #30.97

print('\nBEST metric = chebyshev')