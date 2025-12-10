#* ===== BÚSQUEDA LINEAL =====
#* Complejidad: O(n) - peor caso cuando el elemento está al final o no existe
#* Funciona recorriendo cada elemento del array hasta encontrar el objetivo
def busqueda_lineal(arr, x):
    #! Recorre linealmente desde el inicio hasta el final
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1

#* ===== BÚSQUEDA BINARIA =====
#* Complejidad: O(log n) - divide el espacio de búsqueda a la mitad en cada iteración
#* Requiere que el array esté ordenado. Más eficiente que búsqueda lineal para arrays grandes
def busqueda_binaria(arr, x):
    #! Inicializa punteros en los extremos del array
    low, high = 0, len(arr) - 1
    #! Continúa mientras haya espacio para buscar
    while low <= high:
        #! Calcula el punto medio
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        #! Si el elemento buscado es mayor, descarta la mitad izquierda
        elif arr[mid] < x:
            low = mid + 1
        #! Si es menor, descarta la mitad derecha
        else:
            high = mid - 1
    return -1


#* ===== BUBBLE SORT =====
#* Complejidad: O(n^2) - compara cada par de elementos consecutivos
#* Simple de implementar pero muy ineficiente para datos grandes
def bubble_sort(arr):
    a = arr[:]
    n = len(a)
    #! Loop externo: controla cuántas pasadas hace por el array
    for i in range(n):
        #! Loop interno: compara pares consecutivos y los intercambia si están desordenados
        for j in range(n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a


#* ===== MERGE SORT =====
#* Complejidad: O(n log n) - utiliza divide y conquista
#* Estable y predecible, aunque usa memoria adicional para las sublistas
def merge_sort(arr):
    #! Caso base: arrays de 0 o 1 elemento ya están ordenados
    if len(arr) <= 1:
        return arr
    #! Divide el array en dos mitades
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    #! Combina las dos mitades ordenadas
    return _merge(left, right)


#* Función auxiliar que combina dos arrays ordenados en uno
#* Usado por merge_sort para fusionar las sublistas
def _merge(left, right):
    #! Array que almacenará el resultado
    merged = []
    #! Punteros para recorrer left y right
    i = j = 0
    #! Compara elementos de ambas listas y agrega el menor
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    #! Agrega los elementos restantes (solo uno de los dos loops añadirá elementos)
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

#"""O(n log n) promedio, O(n^2) peor caso - particiona alrededor de pivote"""
# def quick_sort(arr):
#     a = arr[:]
#     def _quick_sort_rec(a, low, high):
#         if low < high:
#             pi = _particionar(a, low, high)
#             _quick_sort_rec(a, low, pi - 1)
#             _quick_sort_rec(a, pi + 1, high)
    
#     _quick_sort_rec(a, 0, len(a) - 1)
#     return a

# def _particionar(a, low, high):
#     """Función auxiliar para quick_sort"""
#     pivot = a[high]
#     i = low - 1
#     for j in range(low, high):
#         if a[j] <= pivot:
#             i += 1
#             a[i], a[j] = a[j], a[i]
#     a[i + 1], a[high] = a[high], a[i + 1]
#     return i + 1



#     """Dijkstra con conteo de operaciones - O(n^2)"""
# def dijkstra_contado(graph, start):
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