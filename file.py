import csv
from sklearn.model_selection import train_test_split
import numpy as np


def read(name):
    data = []
    target = []
    with open(str(name) + '.data') as f:
        reader = csv.reader(f)
        for row in reader:
            f = []
            for r in row:
                try:
                    f.append(float(r))  # se guardan los datos
                except ValueError as e:
                    b = 0 if r == 'M' else 1
                    target.append(b)  # se guarda el resultado en una lista distinta

            data.append(f[1:])  # Se elimina el id
    return {'data': data, 'target': target}