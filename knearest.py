from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)  # se instancia el estimador  k = 1

print(knn)

knn.fit(X, y)  # aprende relación de x e y
