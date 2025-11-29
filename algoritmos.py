def busqueda_lineal(arr, x):
    """O(n) - peor caso: elemento al final o no existe"""
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1

def busqueda_binaria(arr, x):
    """O(log n) - divide el espacio de búsqueda a la mitad"""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def bubble_sort(arr):
    """O(n^2) - compara cada par de elementos consecutivos"""
    a = arr[:]
    n = len(a)
    for i in range(n):
        for j in range(n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a

def merge_sort(arr):
    """O(n log n) - divide y conquista"""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left, right):
    """Función auxiliar para merge_sort"""
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
    return merged

def quick_sort(arr):
    """O(n log n) promedio, O(n^2) peor caso - particiona alrededor de pivote"""
    a = arr[:]
    def _quick_sort_rec(a, low, high):
        if low < high:
            pi = _particionar(a, low, high)
            _quick_sort_rec(a, low, pi - 1)
            _quick_sort_rec(a, pi + 1, high)
    
    _quick_sort_rec(a, 0, len(a) - 1)
    return a

def _particionar(a, low, high):
    """Función auxiliar para quick_sort"""
    pivot = a[high]
    i = low - 1
    for j in range(low, high):
        if a[j] <= pivot:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[high] = a[high], a[i + 1]
    return i + 1

def dijkstra(graph, start):
    """O(n^2) - encuentra caminos más cortos en grafo"""
    n = len(graph)
    dist = [float('inf')] * n
    visited = [False] * n
    dist[start] = 0
    
    for _ in range(n):
        u = -1
        for i in range(n):
            if not visited[i] and (u == -1 or dist[i] < dist[u]):
                u = i
        if u == -1 or dist[u] == float('inf'):
            break
        visited[u] = True
        for v in range(n):
            if graph[u][v] > 0 and not visited[v]:
                new_dist = dist[u] + graph[u][v]
                if new_dist < dist[v]:
                    dist[v] = new_dist
    
    return dist
