
#* ===== ENTRENAMIENTO COMBINADO DE LA RED NEURONAL =====
#* Este módulo carga datos de algoritmos conocidos, extrae características
#* del código fuente y entrena la MLP para reconocer complejidad algorítmica

import csv
import math
from mlp import MLP
from analizador import extraer_caracteristicas_para_mlp


#*===== MAPEOS DE COMPLEJIDAD =====

#! Mapeo numérico de complejidades para clasificación
mapeo = {"O(log n)": 0, "O(n)": 1, "O(n log n)": 2, "O(n^2)": 3}
#! Mapeo inverso: números a complejidades
mapeo_inv = {0: "O(log n)", 1: "O(n)", 2: "O(n log n)", 3: "O(n^2)"}


#*===== UTILIDADES =====

def one_hot(idx, size):
    #! Convierte un índice a representación one-hot
    #! Ej: one_hot(0, 4) = [1, 0, 0, 0]
    #! Usado para entrenamiento supervisado
    v = [0] * size
    v[idx] = 1
    return v


def get_codigo_algoritmo(nombre_algoritmo):
    #! Retorna el código fuente de algoritmos conocidos
    #! Estos algoritmos tienen complejidad bien definida
    
    codigos = {
        'busqueda_lineal': """def busqueda_lineal(arr, x):
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1""",
        
        'busqueda_binaria': """def busqueda_binaria(arr, x):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1""",
        
        'bubble_sort': """def bubble_sort(arr):
    a = arr[:]
    n = len(a)
    for i in range(n):
        for j in range(n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a""",
        
        'selection_sort': """def selection_sort(arr):
    a = arr[:]
    n = len(a)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]
    return a""",
        
        'insertion_sort': """def insertion_sort(arr):
    a = arr[:]
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a""",
        
        'merge_sort': """def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged""",
        
        'quick_sort': """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)""",
    }
    return codigos.get(nombre_algoritmo, "")


#*===== CARGA DE DATOS =====

def load_data_combinado(path="recursos.csv"):
    #! Carga datos de recursos.csv
    #! En lugar de usar características de ejecución, extrae características DEL CÓDIGO
    #! Esto permite que la MLP aprenda patrones de código, no solo resultados de ejecución
    
    X = []
    Y_idx = []
    
    #! Cache para evitar recalcular características de algoritmos repetidos
    cache_features = {}
    
    print("\n Cargando recursos.csv...")
    
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            algoritmo = r["algoritmo"]
            complejidad = r["complejidad"]
            
            if complejidad not in mapeo:
                continue
            
            #! Si no tenemos las características de este algoritmo, extraerlas
            if algoritmo not in cache_features:
                codigo = get_codigo_algoritmo(algoritmo)
                if codigo:
                    #! Extrae 8 características REALES del código
                    features = extraer_caracteristicas_para_mlp(codigo)
                    cache_features[algoritmo] = features
                else:
                    #! Si no encontramos el código, usar características neutras
                    cache_features[algoritmo] = [0.0] * 8
            
            X.append(cache_features[algoritmo])
            Y_idx.append(mapeo[complejidad])
    
    print(f" {len(X)} muestras cargadas")
    print(f"  {len(cache_features)} algoritmos únicos")
    print(f"  Características: {len(X[0])} features extraídas del código")
    
    return X, Y_idx


#*===== NORMALIZACIÓN =====

def normalize(X):
    #! Normaliza cada característica al rango [0, 1]
    #! Esto mejora el entrenamiento de la red neuronal
    
    if not X or len(X[0]) == 0:
        return X
    
    num_features = len(X[0])
    X_norm = [[0] * num_features for _ in range(len(X))]
    
    #! Para cada característica
    for feat_idx in range(num_features):
        #! Extrae todos los valores de esa característica
        values = [x[feat_idx] for x in X]
        #! Calcula mínimo y máximo
        min_val = min(values)
        max_val = max(values)
        #! Evita división por cero
        range_val = max_val - min_val if max_val != min_val else 1
        
        #! Normaliza usando: (valor - min) / (max - min)
        for i, x in enumerate(X):
            X_norm[i][feat_idx] = (x[feat_idx] - min_val) / range_val
    
    return X_norm


#*===== ENTRENAMIENTO =====

def entrenar_mlp_combinado():
    #! Función principal que coordina todo el entrenamiento
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO CON DATASET COMBINADO")
    print("="*70)
    
    #! Carga datos desde CSV
    X, Y_idx = load_data_combinado()
    
    #! Normaliza características
    print("\n Normalizando características...")
    X = normalize(X)
    
    #! Muestra distribución del dataset
    print("\n Distribución de clases:")
    conteo = {}
    for idx in Y_idx:
        comp = mapeo_inv[idx]
        conteo[comp] = conteo.get(comp, 0) + 1
    
    for comp in sorted(conteo.keys()):
        print(f"   {comp:12s}: {conteo[comp]:3d} muestras")
    
    #! Convierte etiquetas a one-hot
    n_outputs = len(set(Y_idx))
    Y = [one_hot(i, n_outputs) for i in Y_idx]
    
    #! Crea la red neuronal
    print("\n Creando red neuronal...")
    mlp = MLP(n_inputs=8, n_hidden=16, n_outputs=n_outputs, lr=0.1)
    
    print(f"   Arquitectura: 8 entradas → 16 ocultas → {n_outputs} salidas")
    print(f"   Learning rate: 0.1")
    
    #! Entrena la red
    epochs = 5000
    print(f"\n Entrenando por {epochs} épocas...\n")
    
    for e in range(epochs):
        loss = mlp.train_epoch(X, Y)
        if e % 500 == 0:
            #! Calcula accuracy en el dataset de entrenamiento
            correct = sum(1 for x, yi in zip(X, Y_idx) if mlp.predict(x) == yi)
            acc = correct / len(X)
            print(f"  Época {e:5d} | Loss: {loss:.8f} | Acc: {acc:.2%} ({correct}/{len(X)})")
    
    #! Evaluación final del modelo
    print("\n" + "="*70)
    print("EVALUACIÓN FINAL")
    print("="*70)
    
    predicciones = []
    correct = 0
    for x, yi in zip(X, Y_idx):
        pred = mlp.predict(x)
        predicciones.append((yi, pred))
        if pred == yi:
            correct += 1
    
    acc = correct / len(X)
    print(f"\n Accuracy global: {acc:.2%} ({correct}/{len(X)})")
    
    #! Muestra accuracy por cada clase de complejidad
    print("\n Accuracy por clase:")
    for clase_idx in range(4):
        total = Y_idx.count(clase_idx)
        correctos = sum(1 for yi, pred in predicciones if yi == clase_idx and pred == clase_idx)
        if total > 0:
            acc_clase = correctos / total
            print(f"   {mapeo_inv[clase_idx]:12s}: {acc_clase:.2%} ({correctos}/{total})")
    
    #! Muestra errores si los hay
    errores = [(yi, pred) for yi, pred in predicciones if yi != pred]
    if errores:
        print(f"\n Errores encontrados: {len(errores)}/{len(X)}")
        print("  Primeros 5 errores:")
        for i, (real, pred) in enumerate(errores[:5]):
            print(f"     Real: {mapeo_inv[real]:12s} | Predicho: {mapeo_inv[pred]:12s}")
    else:
        print("\n Perfecto: Sin errores en el dataset")
    
    print("="*70)
    
    return mlp, acc


if __name__ == "__main__":
    mlp_entrenado, accuracy = entrenar_mlp_combinado()
    
    print("\n" + "="*70)
    if accuracy >= 0.90:
        print(" MLP listo para usar en producción")
    elif accuracy >= 0.80:
        print(" MLP con buena precisión. Resultados confiables")
    else:
        print("  Accuracy moderado. Considera revisar los datos")
    print("="*70)
