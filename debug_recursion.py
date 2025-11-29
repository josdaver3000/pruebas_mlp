import sys
sys.path.insert(0, r'c:\Users\david\Desktop\UNIVERSIDAD\cuarto\AN Y DIS A\github\ADA-projects-exercises-main\pruebas perceptron\actualizacion')

from analizador import detectar_recursion, contar_loops, contar_operaciones, analizar_complejidad_manual

codigo = """def merge_sort(arr):
    \"\"\"O(n log n) - divide y conquista\"\"\"
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
    return merged"""

loops = contar_loops(codigo)
recursion = detectar_recursion(codigo)
operaciones = contar_operaciones(codigo)
complejidad = analizar_complejidad_manual(loops, recursion, operaciones, codigo)

print(f"Loops: {loops}")
print(f"Recursión: {recursion}")
print(f"Operaciones: {operaciones}")
print(f"Complejidad: {complejidad}")
