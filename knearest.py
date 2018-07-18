from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import file as f
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind

file = f.read("wdbc")  # lectura del archivo
X = file["data"]
y = file["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

knn = KNeighborsClassifier(n_neighbors=10)  # se instancia el estimador  k = 5
scores_knn = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print("|| Accuracy K-Neighbors Classifier ||\n")
print(scores_knn)
print("\n")

# print("Promedio: " + str(scores_knn.mean()))

f1_knn = cross_val_score(knn, X, y, cv=5, scoring='f1')
print("|| F1 K-Neighbors Classifier ||\n")
print(f1_knn)

print("\n")

logreg = LogisticRegression()
scores_log = cross_val_score(logreg, X, y, cv=5, scoring='accuracy')
print("|| Accuracy Logistic Regression ||\n")
print(scores_log)
print("Promedio: " + str(scores_log.mean()))
print("\n")

f1_log = cross_val_score(knn, X, y, cv=5, scoring='f1')
print("|| F1 Logistic Regression ||\n")
print(f1_log)

plt.plot(scores_knn, '-o', label='K-Neighbors')
plt.plot(scores_log, '-o', label='Logistic Regression')
plt.xlabel('k-fold')
plt.ylabel('Accuracy Score')
plt.legend()
plt.grid()
plt.show()

result = ttest_ind(scores_knn, scores_log)
mean_knn = scores_knn.mean()
mean_log = scores_log.mean()
print(result)

# knn.fit(X, y)  # aprende relación de x e y
# y_pred = knn.predict(X)  # predicción
# print(metrics.accuracy_score(y, y_pred))
