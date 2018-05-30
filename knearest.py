from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import file as f
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression



file = f.read("wdbc")  # lectura del archivo
X = file["data"]
y = file["target"]

knn = KNeighborsClassifier(n_neighbors=5)  # se instancia el estimador  k = 5
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print(scores)
print(scores.mean())

logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=5, scoring='accuracy').mean())



# knn.fit(X, y)  # aprende relación de x e y
# y_pred = knn.predict(X)  # predicción
# print(metrics.accuracy_score(y, y_pred))
