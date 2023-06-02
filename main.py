from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

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

neigh = KNeighborsClassifier(n_neighbors= 4 )
neigh.fit(X_train,y_train)
y_predK = neigh.predict(X_test)
precision =precision_score(y_test,y_predK,average='micro')
f1Score= f1_score(y_test,y_predK,average='micro')
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_predK).sum()))
print(confusion_matrix(y_test,y_predK))
print(f'Precision Score :{precision * 100}%')
print(f'F1 Score :{f1Score * 100}%')

