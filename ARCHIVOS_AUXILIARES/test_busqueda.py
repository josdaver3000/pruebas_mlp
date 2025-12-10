import math
from mlp import MLP
from entrenamiento import load_data, normalize, one_hot, mapeo
from analizador import analizar_codigo

# Código de prueba
codigo = """def busqueda_lineal_simple(arr, x):
    \"\"\"O(n) - busqueda lineal simple\"\"\"
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1"""

# Entrenar MLP
print("Entrenando MLP...")
X, Y_idx = load_data()
X = normalize(X)
n_outputs = len(set(Y_idx))
Y = [one_hot(i, n_outputs) for i in Y_idx]

mlp = MLP(n_inputs=8, n_hidden=8, n_outputs=n_outputs, lr=0.1)
for e in range(3000):
    loss = mlp.train_epoch(X, Y)
    if e % 1000 == 0:
        print(f"Epoch {e:5d} - Loss {loss:.8f}")

# Analizar código
resultado = analizar_codigo(codigo)
print("\n" + "="*70)
print("ANÁLISIS ESTÁTICO")
print("="*70)
print(f"Nombre de función:        {resultado['nombre']}")
print(f"Loops detectados:         {resultado['loops']}")
print(f"Recursión detectada:      {'Sí' if resultado['recursion'] else 'No'}")
print(f"Operaciones básicas:      {resultado['operaciones']}")
print(f"Complejidad estática:     {resultado['complejidad']}")

# Características
loops = resultado['loops']
recursion = 1 if resultado['recursion'] else 0
operaciones = resultado['operaciones']

loops_weight = loops * 2.0
recursion_weight = recursion * 3.0
ops_norm = operaciones / 20.0
ratio_ops_loops = operaciones / (loops + 0.1)
tiene_loop_anidado = 1.0 if loops >= 2 else 0.0
tiene_recursion = recursion * 1.0
ops_estimadas = operaciones * (loops + 1) * (2 if recursion else 1)
log_ops = math.log(ops_estimadas + 1)
densidad_ops = operaciones / max(1, loops)

caracteristicas = [
    loops_weight,
    tiene_loop_anidado,
    tiene_recursion,
    ratio_ops_loops,
    ops_norm,
    log_ops,
    recursion_weight,
    densidad_ops
]

# Normalizar
rangos = [
    (0, 4),
    (0, 1),
    (0, 3),
    (0, 50),
    (0, 10),
    (0, 8),
    (0, 3),
    (0, 30)
]

caracteristicas_norm = []
for i, carac in enumerate(caracteristicas):
    min_val, max_val = rangos[i]
    range_val = max_val - min_val if max_val != min_val else 1
    norm = (carac - min_val) / range_val
    norm = max(0, min(1, norm))
    caracteristicas_norm.append(norm)

# Predecir
prediccion_idx = mlp.predict(caracteristicas_norm)
mapeo_inverso = {0: "O(log n)", 1: "O(n)", 2: "O(n log n)", 3: "O(n^2)"}
prediccion = mapeo_inverso[prediccion_idx]

print("\n" + "="*70)
print("PREDICCIÓN MLP")
print("="*70)
print(f"Complejidad predicha:     {prediccion}")
print(f"Índice de clase:          {prediccion_idx}")

print("\n" + "="*70)
print("COMPARACIÓN")
print("="*70)
print(f"Análisis estático:        {resultado['complejidad']}")
print(f"Predicción MLP:           {prediccion}")
coinciden = resultado['complejidad'] == prediccion
print(f"¿Coinciden?               {'✓ SÍ' if coinciden else '✗ NO'}")
print("="*70)
