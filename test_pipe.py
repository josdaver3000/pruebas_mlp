import sys
sys.path.insert(0, r'c:\Users\david\Desktop\UNIVERSIDAD\cuarto\AN Y DIS A\github\ADA-projects-exercises-main\pruebas perceptron\actualizacion')

from analizador import analizar_codigo

# Simular entrada desde pipe
lineas = []
while True:
    try:
        linea = input()
        if linea.strip() == "FIN":
            break
        lineas.append(linea)
    except EOFError:
        break

codigo = "\n".join(lineas)
print("=== CÓDIGO RECIBIDO ===")
print(repr(codigo))
print("=== FIN ===\n")

resultado = analizar_codigo(codigo)
print(f"Loops: {resultado['loops']}")
print(f"Recursión: {resultado['recursion']}")
print(f"Complejidad: {resultado['complejidad']}")
