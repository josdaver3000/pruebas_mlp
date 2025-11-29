import time #para hacer uso de la medicion de tiempos
import random #creacion de datos aleatorios
import csv #para esribir el archivo .csv
from algoritmos import * #archivo externo donde se almacenan las funciones de los algoritmos

algoritmos = [
    ("busqueda_lineal", busqueda_lineal, "O(n)"),
    ("busqueda_binaria", busqueda_binaria, "O(log n)"),
    ("bubble_sort", bubble_sort, "O(n^2)"),
    ("merge_sort", merge_sort, "O(n log n)"),
    ("quick_sort", quick_sort, "O(n log n)"),
    # ("dijkstra", dijkstra, "O(n^2)")
]

#tamaños de la entrada para cada algoritmo
entrada = [15, 60, 120, 240, 480, 960] 


#escritura y creacion del archivo .csv,  
with open("RA1_MLP_basico/recursos.csv", mode="w", newline="") as archivo: #("nombre de archio", modo = write {escritura}, nuevalinea)
    writer = csv.writer(archivo) #crea el archivo
    writer.writerow(["algoritmo","tamano","tiempo","complejidad"]) #modifica como se distribuyen los datos 

    for nombre, funcion, clase in algoritmos:
        print(f"monitoreando algoritmo {nombre}")
        for n in entrada:

            # Generar entrada aleatoria
            arr = [random.randint(0, 10000) for i in range(n)]

            # Caso especial: busqueda_binaria requiere lista ordenada
            if nombre == "busqueda_binaria":
                arr.sort()
                x = random.choice(arr)
                args = (arr, x)

            # # Caso especial: dijkstra usa matriz de adyacencia
            # elif nombre == "dijkstra":
            #     graph = [[random.randint(0, 20) if i != j else 0 for j in range(n)] for i in range(n)]
            #     args = (graph, 0)

            # Casos de búsqueda lineal
            elif nombre == "busqueda_lineal":
                x = random.choice(arr)
                args = (arr, x)

            # Resto de algoritmos de ordenamiento
            else:
                args = (arr,)


            # Medir tiempo promedio de ejecución
            repeticiones = 3
            tiempos = []
            for _ in range(repeticiones):
                inicio = time.time()
                funcion(*args)
                fin = time.time()
                tiempos.append(fin - inicio)

            promedio = sum(tiempos) / len(tiempos)
            writer.writerow([nombre, n, promedio, clase])
            print(f"  n={n:4} → {promedio:.6f} s")

print("Datos guardados en recursos.csv")