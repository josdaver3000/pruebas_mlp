import time
import random
import csv
import math


operaciones_totales = 0


def incrementar_operaciones(cantidad=1):
    """Incrementa el contador global de operaciones"""
    global operaciones_totales
    operaciones_totales += cantidad


def busqueda_lineal_contada(arr, x):
    """Búsqueda lineal con conteo de operaciones"""
    global operaciones_totales
    operaciones_totales = 0
    
    for i in range(len(arr)):
        incrementar_operaciones(1)
        if arr[i] == x:
            return i, operaciones_totales
    return -1, operaciones_totales


def busqueda_binaria_contada(arr, x):
    """Búsqueda binaria con conteo de operaciones"""
    global operaciones_totales
    operaciones_totales = 0
    
    low = 0
    high = len(arr) - 1
    while low <= high:
        incrementar_operaciones(1)
        mid = (low + high) // 2
        incrementar_operaciones(3)
        
        incrementar_operaciones(1)
        if arr[mid] == x:
            return mid, operaciones_totales
        
        incrementar_operaciones(1)
        if arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    
    return -1, operaciones_totales


def bubble_sort_contado(arr):
    """Bubble sort con conteo de operaciones"""
    global operaciones_totales
    operaciones_totales = 0
    
    a = arr[:]
    n = len(a)
    for i in range(n):
        incrementar_operaciones(1)
        for j in range(0, n - i - 1):
            incrementar_operaciones(1)
            incrementar_operaciones(1)
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                incrementar_operaciones(2)
    return a, operaciones_totales


def selection_sort_contado(arr):
    """Selection sort con conteo de operaciones"""
    global operaciones_totales
    operaciones_totales = 0
    
    a = arr[:]
    n = len(a)
    for i in range(n):
        incrementar_operaciones(1)
        min_idx = i
        for j in range(i + 1, n):
            incrementar_operaciones(1)
            incrementar_operaciones(1)
            if a[j] < a[min_idx]:
                min_idx = j
        incrementar_operaciones(2)
        a[i], a[min_idx] = a[min_idx], a[i]
    return a, operaciones_totales


def insertion_sort_contado(arr):
    """Insertion sort con conteo de operaciones"""
    global operaciones_totales
    operaciones_totales = 0
    
    a = arr[:]
    for i in range(1, len(a)):
        incrementar_operaciones(1)
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            incrementar_operaciones(2)
            a[j + 1] = a[j]
            j -= 1
        incrementar_operaciones(1)
        a[j + 1] = key
    return a, operaciones_totales


def merge_sort_contado(arr):
    """Merge sort con conteo de operaciones"""
    global operaciones_totales
    
    def merge_helper(left, right):
        merged = []
        i = j = 0
        while i < len(left) and j < len(right):
            incrementar_operaciones(2)
            if left[i] < right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
            incrementar_operaciones(1)
        
        merged.extend(left[i:])
        merged.extend(right[j:])
        incrementar_operaciones(len(left[i:]) + len(right[j:]))
        return merged
    
    def merge_sort_rec(arr):
        if len(arr) <= 1:
            return arr
        
        incrementar_operaciones(2)
        mid = len(arr) // 2
        left = merge_sort_rec(arr[:mid])
        right = merge_sort_rec(arr[mid:])
        incrementar_operaciones(2)
        
        return merge_helper(left, right)
    
    operaciones_totales = 0
    resultado = merge_sort_rec(arr)
    return resultado, operaciones_totales


def quick_sort_contado(arr):
    """Quick sort con conteo de operaciones"""
    global operaciones_totales
    
    def particionar(a, low, high):
        pivot = a[high]
        i = low - 1
        for j in range(low, high):
            incrementar_operaciones(1)
            incrementar_operaciones(1)
            if a[j] <= pivot:
                i += 1
                a[i], a[j] = a[j], a[i]
                incrementar_operaciones(2)
        
        a[i + 1], a[high] = a[high], a[i + 1]
        incrementar_operaciones(2)
        return i + 1
    
    def quick_sort_rec(a, low, high):
        if low < high:
            incrementar_operaciones(1)
            pivot_index = particionar(a, low, high)
            quick_sort_rec(a, low, pivot_index - 1)
            quick_sort_rec(a, pivot_index + 1, high)
    
    operaciones_totales = 0
    a = arr[:]
    quick_sort_rec(a, 0, len(a) - 1)
    return a, operaciones_totales


# def dijkstra_contado(graph, start):
#     """Dijkstra con conteo de operaciones - O(n^2)"""
#     global operaciones_totales
#     operaciones_totales = 0
    
#     n = len(graph)
#     dist = [float('inf')] * n
#     visited = [False] * n
#     dist[start] = 0
    
#     for _ in range(n):
#         incrementar_operaciones(1)
#         u = -1
#         for i in range(n):
#             incrementar_operaciones(1)
#             incrementar_operaciones(1)
#             if not visited[i] and (u == -1 or dist[i] < dist[u]):
#                 u = i
        
#         if u == -1 or dist[u] == float('inf'):
#             break
        
#         visited[u] = True
#         incrementar_operaciones(1)
        
#         for v in range(n):
#             incrementar_operaciones(1)
#             incrementar_operaciones(1)
#             if graph[u][v] > 0 and not visited[v]:
#                 new_dist = dist[u] + graph[u][v]
#                 incrementar_operaciones(1)
#                 if new_dist < dist[v]:
#                     dist[v] = new_dist
#                     incrementar_operaciones(1)
    
#     return dist, operaciones_totales


algoritmos = [
    ("busqueda_lineal", busqueda_lineal_contada, "O(n)"),
    ("busqueda_binaria", busqueda_binaria_contada, "O(log n)"),
    ("bubble_sort", bubble_sort_contado, "O(n^2)"),
    ("selection_sort", selection_sort_contado, "O(n^2)"),
    ("insertion_sort", insertion_sort_contado, "O(n^2)"),
    ("merge_sort", merge_sort_contado, "O(n log n)"),
    ("quick_sort", quick_sort_contado, "O(n log n)"),
    # ("dijkstra", dijkstra_contado, "O(n^2)"),
]

entrada = [10, 15, 20, 25, 30, 50, 75, 100, 150, 200, 210, 350, 400, 450, 550, 660]


with open("recursos.csv", mode="w", newline="") as archivo:
    writer = csv.writer(archivo)
    writer.writerow([
        "algoritmo", 
        "tamano", 
        "tiempo", 
        "complejidad",
        "operaciones_reales",
        "operaciones_normalizadas",
        "ratio_ops_tamano",
        "log_operaciones",
        "log_tamano",
        "tiempo_normalizado"
    ])

    for nombre, funcion, clase in algoritmos:
        print(f"\nmonitoreando algoritmo {nombre}")
        
        for n in entrada:

            arr = [random.randint(0, 10000) for i in range(n)]

            if nombre == "busqueda_binaria":
                arr.sort()
                x = random.choice(arr)
                args = (arr, x)
            elif nombre == "busqueda_lineal":
                x = random.choice(arr)
                args = (arr, x)
            # elif nombre == "dijkstra":
            #     graph = [[random.randint(0, 20) if i != j else 0 for j in range(n)] for i in range(n)]
            #     args = (graph, 0)
            else:
                args = (arr,)

            if n <= 100:
                repeticiones = 10
            elif n <= 300:
                repeticiones = 7
            else:
                repeticiones = 5
            
            tiempos = []
            operaciones_lista = []
            
            for _ in range(repeticiones):
                inicio = time.time()
                resultado = funcion(*args)
                fin = time.time()
                
                if isinstance(resultado, tuple):
                    _, ops = resultado
                else:
                    ops = 0
                
                tiempos.append(fin - inicio)
                operaciones_lista.append(ops)

            promedio_tiempo = sum(tiempos) / len(tiempos)
            promedio_ops = sum(operaciones_lista) / len(operaciones_lista)
            
            ops_normalizadas = promedio_ops / (n ** 2 + 1)
            ratio_ops_tamano = promedio_ops / n if n > 0 else 0
            log_ops = math.log(promedio_ops + 1)
            log_tamano = math.log(n + 1)
            tiempo_norm = promedio_tiempo / (n ** 2 + 1)
            
            writer.writerow([
                nombre, 
                n, 
                promedio_tiempo, 
                clase,
                int(promedio_ops),
                ops_normalizadas,
                ratio_ops_tamano,
                log_ops,
                log_tamano,
                tiempo_norm
            ])
            print(f"  n={n:4d} -> ops={int(promedio_ops):8d} | tiempo={promedio_tiempo:.9f} s | ratio={ratio_ops_tamano:.6f}")

print("\n" + "="*80)
print("Datos guardados en recursos.csv")
print("Dataset de 128 muestras (8 algoritmos x 16 tamaños)")
print("="*80)