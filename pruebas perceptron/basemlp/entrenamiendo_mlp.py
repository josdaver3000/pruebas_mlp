import csv
import math
from MLP_sencillo import MLP

mapeo = {"O(log n)":0, "O(n)":1, "O(n log n)":2, "O(n^2)":3, "O(n) avg":1, "O(2^n)":4}

def one_hot(idx, size):
    v = [0]*size
    v[idx] = 1
    return v

def load_data(path="recursos.csv"):
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

def normalize(X):
    max_n = max(x[0] for x in X)
    max_t = max(x[1] for x in X)
    return [[x[0]/max_n, x[1]/max_t] for x in X]

if __name__ == "__main__":
    X, Y_idx = load_data()
    X = normalize(X)
    n_outputs = len(set(Y_idx))
    Y = [one_hot(i, n_outputs) for i in Y_idx]

    mlp = MLP(n_inputs=2, n_hidden=8, n_outputs = n_outputs, lr=0.1)
    epochs = 2000
    for e in range(epochs):
        loss = mlp.train_epoch(X, Y)
        if e % 100 == 0:
            print(f"Epoch {e} - Loss {loss:.6f}")

    correct = 0
    for x, yi in zip(X, Y_idx):
        pred = mlp.predict(x)
        if pred == yi:
            correct += 1
    acc = correct / len(X)
    print("Accuracy on training set:", acc)
