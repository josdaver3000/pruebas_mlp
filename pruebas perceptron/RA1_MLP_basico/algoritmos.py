
#=======busqueda lineal=======

def busqueda_lineal(arr, x):
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1
    



#=======busqueda binaria=======

def busqueda_binaria(arr, x):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1




#=======bubble_sort=======

def bubble_sort(arr):
    a = arr[:]
    n = len(a)
    for i in range(n):
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a




#=======algoritmo merge_sort=======

def merge_sort(arr,):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
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




#=======algoritmo quick_sort=======   
 
def quick_sort(arr):
    a = arr[:]
    quick_sort_rec(a, 0, len(a) - 1)
    return a

def quick_sort_rec(a, low, high):
    if low < high:
        pivot_index = particionar(a, low, high)
        quick_sort_rec(a, low, pivot_index - 1)
        quick_sort_rec(a, pivot_index + 1, high)

def particionar(a, low, high):
    pivot = a[high]
    i = low - 1
    for j in range(low, high):
        if a[j] <= pivot:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[high] = a[high], a[i + 1]
    return i + 1



#======algoritmo de dijkstra=======
#!!!!!usado en RA1 solo para generar el dataset!!!!!!!!

# def dijkstra(graph, start):
#     n = len(graph)
#     dist = [float('inf')] * n
#     visited = [False] * n
#     dist[start] = 0

#     for _ in range(n):
#         u = min_distance(dist, visited)
#         if u == -1:
#             break
#         visited[u] = True

#         for v in range(n):
#             if graph[u][v] > 0 and not visited[v]:
#                 new_dist = dist[u] + graph[u][v]
#                 if new_dist < dist[v]:
#                     dist[v] = new_dist
#     return dist

# def min_distance(dist, visited):
#     min_val = float('inf')
#     min_index = -1
#     for i in range(len(dist)):
#         if not visited[i] and dist[i] < min_val:
#             min_val = dist[i]
#             min_index = i
#     return min_index


