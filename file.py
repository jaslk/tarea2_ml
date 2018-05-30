import csv


def read(name):
    matrix = []
    with open(str(name) + '.data') as f:
        reader = csv.reader(f)
        for row in reader:
            f = []
            for r in row:
                try:
                    f.append(float(r))
                except ValueError as e:
                    f.append(r)
            matrix.append(f)
    return matrix


m = read("wdbc")

for a in m:
    print(a[2])
