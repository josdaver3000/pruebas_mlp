import csv
import math
from MLP_sencillo import MLP

mapeo = {"O(log n)":0, "O(n)":1, "O(n log n)":2, "O(n^2)":3, "O(n) avg":1}

def one_hot(idx, size):
    v = [0]*size
    v[idx] = 1
    return v

def load_data(path="RA1_MLP_basico/recursos.csv"):
    X = []; Y = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            n = float(r["tamano"])
            t = float(r["tiempo"]) if r["tiempo"] != "inf" else 1e6
            cls = r["complejidad"]
            if cls not in mapeo:
                continue
            X.append([n, t])
            Y.append(mapeo[cls])
    return X, Y

# MODIFICACIÓN: se aplica transformación logarítmica para reducir
# el solapamiento entre clases O(n), O(n log n), O(n^2),
# ya que en escala lineal el MLP no distinguía bien las tendencias.

#=======================================codigo anterior======================================

#def normalize(X):
#    max_n = max(x[0] for x in X)
#    max_t = max(x[1] for x in X)
#    return [[x[0]/max_n, x[1]/max_t] for x in X]

#===========================================================================================

def normalize(X):
    eps = 1e-9
    X2 = []
    max_n = max(math.log(x[0] + 1) for x in X)
    max_t = max(math.log(x[1] + 1) for x in X)
    for n, t in X:
        X2.append([
            math.log(n + 1) / max_n,
            math.log(t + eps) / max_t
        ])
    return X2
if __name__ == "__main__":
    X, Y_idx = load_data()
    X = normalize(X)
    n_outputs = len(set(Y_idx))
    Y = [one_hot(i, n_outputs) for i in Y_idx]

    mlp = MLP(n_inputs=2, n_hidden=8, n_outputs = n_outputs, lr=0.1)
    epochs = 10000
    for e in range(epochs):
        loss = mlp.train_epoch(X, Y)

        # # MODIFICACIÓN: reducción progresiva del learning rate
        # # Se ejecuta cada 500 épocas para mejorar la estabilidad de convergencia.

        # if e % 500 == 0 and e > 0:
        #     mlp.lr *= 0.5
        #     print(f"Learning rate reducido a {mlp.lr:.5f}")

        if e % 100 == 0:
            print(f"Epoch {e} - Loss {loss:.6f}")


    correct = 0
    for x, yi in zip(X, Y_idx):
        pred = mlp.predict(x)
        if pred == yi:
            correct += 1
    acc = correct / len(X)
    print("Accuracy on training set:", acc)
