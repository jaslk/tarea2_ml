
# Problema
 
Estimados alumnos/as:

La tarea de clasificación consiste en lo siguiente:

- Deben comparar 2 técnicas de clasificación (K neareast Neighbor, Naive Bayes, SVM, Random Forest, etc) en los datos de Boston Cancer (wdbc.data que se encuentra en http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/)
- Se debe realizar un Repeated K-Fold Cross Validation con K=5, reportando las métricas Accuracy y F1. Recuerde calcular las metricas para ambos modelos con los mismos datos. 
- Debe concluir los experimentos indicando cual de los 2 modelos genera el mejor desempeño. Para ello puede utilizar un test de hipotesis de medias. Se puede apoyar con gráficos (Bar plot, Boxplot, etc.) Revise: https://towardsdatascience.com/inferential-statistics-series-t-test-using-numpy-2718f8f9bf2f (1. An Independent Samples t-test compares the means for two groups.)

Debe entregar el Jupyter Notebook, con código comentado y conclusiones finales.

Saludos

Héctor

# Conclusiones

De lo anteriormente descrito, se realizó un test de hipótesis media para estimar que método de clasificación obtenía el mejor desempeño según la métrica Accuracy. El test arrojó los siguientes resultados: 

t test = -1.3663361600481534
p-value = 0.2090000044038142

Con el valor obtenido de T-test se puede concluir que los modelos difieren en cuanto al desempeño obtenido, siendo el método Logistic Regression el que genera un mejor resultado promediando una precisión de 0.9526 frente a 0.9316 del método K-neighbors Classifier.
