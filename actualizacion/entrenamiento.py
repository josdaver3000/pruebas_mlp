import csv
import math
from mlp import MLP


mapeo = {"O(log n)": 0, "O(n)": 1, "O(n log n)": 2, "O(n^2)": 3}


def one_hot(idx, size):
    v = [0] * size
    v[idx] = 1
    return v


def load_data(path="recursos.csv"):
    X = []
    Y = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            tamano = float(r["tamano"])
            tiempo = float(r["tiempo"])
            operaciones_reales = float(r["operaciones_reales"])
            ops_norm = float(r["operaciones_normalizadas"])
            ratio_ops = float(r["ratio_ops_tamano"])
            log_ops = float(r["log_operaciones"])
            log_tam = float(r["log_tamano"])
            tiempo_norm = float(r["tiempo_normalizado"])
            
            cls = r["complejidad"]
            if cls not in mapeo:
                continue
            
            X.append([
                tamano,
                operaciones_reales,
                ops_norm,
                ratio_ops,
                log_ops,
                log_tam,
                tiempo,
                tiempo_norm
            ])
            Y.append(mapeo[cls])
    
    return X, Y


def normalize(X):
    """Normaliza cada característica al rango [0, 1]"""
    if not X or len(X[0]) == 0:
        return X
    
    num_features = len(X[0])
    X_norm = []
    
    for feat_idx in range(num_features):
        values = [x[feat_idx] for x in X]
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        if feat_idx == 0:
            X_norm = [[0] * num_features for _ in range(len(X))]
        
        for i, x in enumerate(X):
            X_norm[i][feat_idx] = (x[feat_idx] - min_val) / range_val
    
    return X_norm


if __name__ == "__main__":
    X, Y_idx = load_data()
    print(f"Datos cargados: {len(X)} muestras")
    print(f"Numero de caracteristicas: {len(X[0])}")
    print(f"Clases: {len(set(Y_idx))}")
    
    print("\nPrimeras 5 muestras:")
    clases_map = {0: "O(log n)", 1: "O(n)", 2: "O(n log n)", 3: "O(n^2)"}
    for i in range(min(5, len(X))):
        print(f"  Sample {i}: clase={clases_map[Y_idx[i]]}, ops={X[i][1]:.0f}, tamano={X[i][0]:.0f}")
    
    X = normalize(X)
    n_outputs = len(set(Y_idx))
    Y = [one_hot(i, n_outputs) for i in Y_idx]

    mlp = MLP(n_inputs=8, n_hidden=8, n_outputs=n_outputs, lr=0.1)
    epochs = 10000
    
    print(f"\nEntrenando MLP: 8 entradas -> 64 ocultas -> {n_outputs} salidas")
    print(f"Learning rate: 0.03")
    print(f"Epocas: {epochs}\n")
    
    for e in range(epochs):
        loss = mlp.train_epoch(X, Y)
        if e % 100 == 0:
            print(f"Epoch {e:5d} - Loss {loss:.8f}")

    print("\n" + "="*70)
    correct = 0
    predicciones = []
    for x, yi in zip(X, Y_idx):
        pred = mlp.predict(x)
        predicciones.append((yi, pred))
        if pred == yi:
            correct += 1
    acc = correct / len(X)
    
    print(f"Accuracy on training set: {acc:.4f} ({correct}/{len(X)})")
    
    print("\nDetalles por clase:")
    for clase_idx in range(4):
        total = Y_idx.count(clase_idx)
        correctos = sum(1 for yi, pred in predicciones if yi == clase_idx and pred == clase_idx)
        if total > 0:
            print(f"  {clases_map[clase_idx]:12s} -> {correctos}/{total} correctas ({100*correctos/total:.1f}%)")
    
    print("="*70)