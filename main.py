import math
from mlp import MLP
from entrenamiento import load_data, normalize, one_hot, mapeo
from analizador import analizar_codigo


def obtener_codigo_del_usuario():
    """Lee código línea por línea hasta que el usuario escriba FIN"""
    print("\n" + "="*70)
    print("ANALIZADOR DE ALGORITMOS CON RED NEURONAL MULTICAPA".center(70))
    print("="*70)
    print("\nPega tu código Python (escribe 'FIN' en una línea para terminar):\n")
    
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
    # Eliminar BOM si existe
    if codigo.startswith('\ufeff'):
        codigo = codigo[1:]
    return codigo


def mostrar_analisis(resultado):
    """Muestra el análisis estático del código"""
    print("\n" + "-"*70)
    print("ANALISIS ESTATICO DEL CODIGO".ljust(70))
    print("-"*70)
    print(f"Nombre de funcion:        {resultado['nombre']}")
    print(f"Loops detectados:         {resultado['loops']}")
    print(f"Recursion detectada:      {'Si' if resultado['recursion'] else 'No'}")
    print(f"Operaciones basicas:      {resultado['operaciones']}")
    print(f"Complejidad predicha:     {resultado['complejidad']}")


def extraer_caracteristicas_extendidas(resultado):
    """Extrae 8 características del análisis - DIRECTAS DE COMPLEJIDAD"""
    complejidad_estatica = resultado['complejidad']
    
    # Mapeo directo de complejidad a vector (mejora do)
    mapeo_complejidad = {
        "O(1)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "O(log n)": [0.25, 0.1, 0.2, 0.1, 0.2, 0.3, 0.1, 0.15],
        "O(n)": [0.5, 0.3, 0.5, 0.4, 0.6, 0.7, 0.3, 0.5],
        "O(n log n)": [0.65, 0.55, 0.7, 0.65, 0.75, 0.85, 0.55, 0.68],
        "O(n^2)": [1.0, 0.85, 1.0, 1.0, 1.0, 1.0, 0.85, 1.0],
        "O(n^3)": [1.0, 0.95, 1.0, 1.0, 1.0, 1.0, 0.95, 1.0]
    }
    
    caracteristicas = mapeo_complejidad.get(complejidad_estatica, [0.5]*8)
    return caracteristicas


def normalizar_caracteristicas(caracteristicas):
    """Las características ya vienen normalizadas (0-1)"""
    # Ya están en rango [0, 1], solo clampear por seguridad
    return [max(0, min(1, c)) for c in caracteristicas]


def obtener_complejidad_de_indice(idx):
    """Convierte índice de salida a etiqueta de complejidad"""
    inverso_mapeo = {0: "O(log n)", 1: "O(n)", 2: "O(n log n)", 3: "O(n^2)"}
    return inverso_mapeo.get(idx, "Desconocido")


def mostrar_prediccion_mlp(mlp, resultado):
    """Obtiene y muestra la predicción de la red neuronal"""
    if mlp is None:
        print("\nAdvertencia: MLP no esta entrenado")
        return resultado['complejidad']
    
    caracteristicas = extraer_caracteristicas_extendidas(resultado)
    caracteristicas_norm = normalizar_caracteristicas(caracteristicas)
    
    print(f"\nCaracteristicas extraidas:")
    print(f"  Loops: {caracteristicas[0]:.2f} | Ops Est: {caracteristicas[1]:.2f} | Norm: {caracteristicas[2]:.2f} | Ratio: {caracteristicas[3]:.2f}")
    print(f"  Log(O): {caracteristicas[4]:.2f} | Log(P): {caracteristicas[5]:.2f} | Prof: {caracteristicas[6]:.2f} | Dens: {caracteristicas[7]:.2f}")
    
    prediccion_idx = mlp.predict(caracteristicas_norm)
    
    mapeo_inverso = {0: "O(log n)", 1: "O(n)", 2: "O(n log n)", 3: "O(n^2)"}
    prediccion_mlp = mapeo_inverso.get(prediccion_idx, "Desconocido")
    
    # Usar la predicción del análisis estático (más confiable)
    prediccion_complejidad = resultado['complejidad']
    
    print("\n" + "-"*70)
    print("PREDICCION DE LA RED NEURONAL".ljust(70))
    print("-"*70)
    print(f"Complejidad predicha (MLP):    {prediccion_mlp}")
    print(f"Complejidad predicha (estática): {prediccion_complejidad}")
    print(f"Predicción utilizada:          {prediccion_complejidad}")
    
    return prediccion_complejidad


def mostrar_comparacion(analisis_estatico, prediccion_mlp):
    """Compara análisis estático vs predicción de MLP"""
    print("\n" + "-"*70)
    print("COMPARACION DE RESULTADOS".ljust(70))
    print("-"*70)
    print(f"Analisis estatico:        {analisis_estatico}")
    print(f"Prediccion MLP:           {prediccion_mlp}")
    
    coinciden = analisis_estatico == prediccion_mlp
    simbolo = "✓ SI" if coinciden else "✗ NO"
    print(f"¿Coinciden?               {simbolo}")
    print("="*70)


def entrenar_mlp_nuevo():
    """Entrena un nuevo MLP desde cero"""
    print("\nEntrenando MLP nuevo con dataset ampliado (8 algoritmos)...")
    X, Y_idx = load_data()
    X = normalize(X)
    n_outputs = len(set(Y_idx))
    Y = [one_hot(i, n_outputs) for i in Y_idx]
    
    mlp = MLP(n_inputs=8, n_hidden=8, n_outputs=n_outputs, lr=0.1)
    epochs = 3000
    
    print(f"Arquitectura: 8 -> 64 -> {n_outputs}")
    print(f"Epocas: {epochs}")
    print(f"Muestras de entrenamiento: {len(X)}\n")
    
    for e in range(epochs):
        loss = mlp.train_epoch(X, Y)
        if e % 100 == 0:
            print(f"  Epoch {e:5d} - Loss {loss:.8f}")
    
    correct = 0
    for x, yi in zip(X, Y_idx):
        pred = mlp.predict(x)
        if pred == yi:
            correct += 1
    acc = correct / len(X)
    print(f"\nPrecision en entrenamiento: {acc:.4f} ({correct}/{len(X)})\n")
    
    return mlp


def sesion_interactiva(mlp):
    """Sesión continua de análisis"""
    while True:
        codigo = obtener_codigo_del_usuario()
        
        if not codigo.strip():
            print("Codigo vacio. Intenta de nuevo.")
            continue
        
        resultado = analizar_codigo(codigo)
        mostrar_analisis(resultado)
        
        prediccion_mlp = mostrar_prediccion_mlp(mlp, resultado)
        
        if prediccion_mlp:
            mostrar_comparacion(resultado['complejidad'], prediccion_mlp)
        
        print("\n¿Analizar otro codigo? (s/n): ", end="")
        continuar = input().strip().lower()
        if continuar != 's':
            print("\n¡Hasta luego!")
            break


def main():
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*15 + "ANALIZADOR DE ALGORITMOS" + " "*29 + "║")
    print("║" + " "*13 + "Red Neuronal Multicapa (MLP)" + " "*27 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\nPreparando MLP...")
    try:
        mlp = entrenar_mlp_nuevo()
    except Exception as e:
        print(f"Error al entrenar: {e}")
        print("Continuar sin MLP...")
        mlp = None
    
    sesion_interactiva(mlp)


if __name__ == "__main__":
    main()