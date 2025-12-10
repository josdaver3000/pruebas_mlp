import os #* para manejo de archivos
import json #* para guardar y cargar modelo
from mlp import MLP #* clase MLP "red neuronal multicapa"
from entrenamiento_combinado import load_data_combinado, normalize, one_hot, mapeo_inv #* funciones de carga y preprocesamiento
from analizador import analizar_codigo #* función de análisis estático


#*==================== PERSISTENCIA DEL MODELO ===================
#* Funciones para guardar y cargar modelos entrenados
#* Esto permite reutilizar modelos sin necesidad de re-entrenar


def guardar_modelo(mlp, archivo="modelo_mlp.json"):
    #! Serializa la red neuronal en formato JSON
    #! Guarda arquitectura y todos los pesos aprendidos
    
    modelo = {
        'arquitectura': {
            'n_inputs': mlp.n_inputs,
            'n_hidden': mlp.n_hidden,
            'n_outputs': mlp.n_outputs,
            'lr': mlp.lr
        },
        'pesos': {
            'w1': mlp.w1,
            'b1': mlp.b1,
            'w2': mlp.w2,
            'b2': mlp.b2
        }
    }
    
    with open(archivo, 'w') as f:
        json.dump(modelo, f)

def cargar_modelo(archivo="modelo_mlp.json"):
    #! Deserializa un modelo guardado previamente
    #! Retorna instancia de MLP lista para usar
    
    try:
        with open(archivo, 'r') as f:
            modelo = json.load(f)
        
        arq = modelo['arquitectura']
        mlp = MLP(
            n_inputs=arq['n_inputs'],
            n_hidden=arq['n_hidden'],
            n_outputs=arq['n_outputs'],
            lr=arq['lr']
        )
        
        pesos = modelo['pesos']
        mlp.w1 = pesos['w1']
        mlp.b1 = pesos['b1']
        mlp.w2 = pesos['w2']
        mlp.b2 = pesos['b2']
        
        return mlp
    except:
        return None
#*===== ENTRENAMIENTO =====

def entrenar_mlp_inicial():
    #! Entrena la MLP desde cero usando el dataset combinado
    #! Se ejecuta solo si no existe modelo guardado
    
    X, Y_idx = load_data_combinado()
    X = normalize(X)
    Y = [one_hot(i, 4) for i in Y_idx]
    
    mlp = MLP(n_inputs=8, n_hidden=16, n_outputs=4, lr=0.1)
    
    for e in range(5000):
        mlp.train_epoch(X, Y)
    
    return mlp


#*===== CORRECCIÓN DE ERRORES EN TIEMPO REAL =====

def corregir_en_tiempo_real(mlp, caracteristicas, complejidad_correcta):
    #! Re-entrena la MLP si el usuario indica que la predicción fue incorrecta
    #! Esto permite que el modelo aprenda de sus errores
    
    print("\n" + "="*70)
    print("AUTO-CORRECCIÓN EN TIEMPO REAL".center(70))
    print("="*70)
    print("\nSe detecto una discrepancia en la predicción")
    print("Corrigiendo el error...")
    print("Por favor espera unos segundos...\n")
    
    #! Convertir complejidad correcta a índice numérico
    mapeo = {"O(log n)": 0, "O(n)": 1, "O(n log n)": 2, "O(n^2)": 3, "O(n^3)": 4}
    idx_correcto = mapeo.get(complejidad_correcta, 1)
    
    #! Crear representación one-hot del valor correcto
    y_correcto = [0] * 4
    if idx_correcto < 4:
        y_correcto[idx_correcto] = 1
    else:
        y_correcto = [0, 0, 0, 1]  #! Si es O(n^3), usar O(n^2) como más cercano
    
    #! Re-entrenar con este nuevo dato
    print("Re-entrenando con el dato correcto...")
    for _ in range(500):
        mlp.train_epoch([caracteristicas], [y_correcto])
    
    print("Corrección completada")
    print("Guardando conocimiento actualizado...")
    
    guardar_modelo(mlp)
    
    print("Modelo actualizado y guardado\n")
    print("="*70)
    
    return mlp


#*===== INTERFAZ DE USUARIO =====

def obtener_codigo_del_usuario():
    #! Lee código Python del usuario línea por línea
    #! Termina cuando el usuario escribe "FIN"
    
    print("\nPega tu código Python.")
    print("Escribe 'FIN' cuando termines.\n")
    
    lineas = []
    while True:
        try:
            linea = input()
            if linea.strip() == "FIN":
                break
            lineas.append(linea)
        except EOFError:
            break
    
    return '\n'.join(lineas)

def mostrar_resultado(pred_mlp, complejidad_estatica, confianza, es_segunda_vez=False):
    #! Muestra los resultados del análisis de forma clara
    #! Compara predicción MLP vs análisis estático
    
    print("\n" + "="*70)
    if es_segunda_vez:
        print("RESULTADO CORREGIDO".center(70))
    else:
        print("RESULTADO DEL ANÁLISIS".center(70))
    print("="*70)
    
    #! Mostrar AMBAS predicciones para comparar
    print(f"\nRed Neuronal (MLP):       {pred_mlp}")
    print(f"Análisis Estático:        {complejidad_estatica}")
    
    #! Estado de coincidencia
    if pred_mlp == complejidad_estatica:
        print(f"Estado: COINCIDEN")
        if es_segunda_vez:
            print(f" Ahora la prediccion es correcta!")
    else:
        print(f"Estado: DIFIEREN")
    
    #! Confianza
    if confianza >= 0.85:
        nivel = "MUY ALTA"
    elif confianza >= 0.70:
        nivel = "ALTA"
    elif confianza >= 0.50:
        nivel = "MEDIA"
    else:
        nivel = "BAJA"
    
    print(f"Confianza MLP: {nivel} ({confianza:.1%})")
    
    #! Explicación de la complejidad
    explicaciones = {
        "O(log n)": "Muy eficiente - crece logaritmicamente",
        "O(n)": "Eficiente - crece linealmente ",
        "O(n log n)": "Bastante eficiente - buen equilibrio ",
        "O(n^2)": "Menos eficiente - crece cuadraticamente ",
        "O(n^3)": "Ineficiente - crece cubicamente "
    }
    
    print(f"\nSignificado de {pred_mlp}:")
    print(f"  {explicaciones.get(pred_mlp, 'Sin descripcion')}")
    
    print("\n" + "="*70)


#! Analiza código y se autocorrige SOLO si hay diferencia real.
def analizar_con_autocorreccion(mlp, codigo):

    #! 1. Analizar código
    resultado = analizar_codigo(codigo)
    caracteristicas = resultado['caracteristicas_mlp']
    complejidad_estatica = resultado['complejidad']
    
    #! 2. Predecir con MLP
    prediccion_idx = mlp.predict(caracteristicas)
    prediccion_mlp = mapeo_inv[prediccion_idx]
    output = mlp.forward(caracteristicas)
    confianza = output[prediccion_idx]
    
    #! 3. Mostrar resultado (SIEMPRE muestra comparación)
    mostrar_resultado(prediccion_mlp, complejidad_estatica, confianza)
    
    #! 4. VALIDAR: ¿Hay diferencia REAL?
    hay_diferencia = (prediccion_mlp != complejidad_estatica)
    
    if hay_diferencia:
        #! AUTO-CORRECCIÓN EN TIEMPO REAL
        mlp = corregir_en_tiempo_real(mlp, caracteristicas, complejidad_estatica)
        
        #! PEDIR QUE PEGUE EL CÓDIGO OTRA VEZ
        print("\nAhora que aprendi, vuelve a pegar el mismo codigo")
        print("para verificar que ahora doy la respuesta correcta.\n")
        
        codigo_nuevo = obtener_codigo_del_usuario()
        
        if codigo_nuevo.strip():
            print("\nAnalizando nuevamente...")
            
            #! Analizar de nuevo
            resultado_nuevo = analizar_codigo(codigo_nuevo)
            caracteristicas_nuevo = resultado_nuevo['caracteristicas_mlp']
            complejidad_estatica_nuevo = resultado_nuevo['complejidad']
            
            #! Predecir con MLP CORREGIDA
            prediccion_idx_nuevo = mlp.predict(caracteristicas_nuevo)
            prediccion_mlp_nuevo = mapeo_inv[prediccion_idx_nuevo]
            output_nuevo = mlp.forward(caracteristicas_nuevo)
            confianza_nuevo = output_nuevo[prediccion_idx_nuevo]
            
            #! Mostrar resultado corregido
            mostrar_resultado(
                prediccion_mlp_nuevo,
                complejidad_estatica_nuevo,
                confianza_nuevo,
                es_segunda_vez=True
            )
    else:
        #! NO hay diferencia - No hace falta corregir
        print("\nLa prediccion de la red neuronal es correcta.")
        print("No es necesario re-entrenar.\n")
    
    return mlp


#* ==================== MAIN ====================

def main():
    print("\n" + "="*70)
    print("ANALIZADOR DE COMPLEJIDAD ALGORITMICA".center(70))
    print("Con Auto-Correccion en Tiempo Real".center(70))
    print("="*70)
    
    print("\nFuncionamiento:")
    print("  - Analiza tu codigo con Red Neuronal + Analisis Estatico")
    print("  - Si difieren, me corrijo INMEDIATAMENTE")
    print("  - Te pido el codigo otra vez para verificar")
    print("  - Ahora doy la respuesta correcta\n")
    
    #! 1. Cargar o entrenar modelo
    MODELO_FILE = "modelo_mlp.json"
    
    print("Cargando inteligencia artificial...")
    mlp = cargar_modelo(MODELO_FILE)
    
    if mlp:
        print("Modelo cargado correctamente\n")
    else:
        print("Primera ejecucion - Entrenando modelo inicial...")
        print("Esto puede tomar unos segundos...\n")
        mlp = entrenar_mlp_inicial()
        guardar_modelo(mlp)
        print("Modelo entrenado y guardado\n")
    
    #! 2. Sesión continua
    while True:
        print("="*70)
        print("\nDeseas analizar la complejidad de un algoritmo? (s/n): ", end="")
        respuesta = input().strip().lower()
        
        if respuesta not in ['s', 'si', 'sí', 'yes', 'y']:
            print("\nHasta luego!")
            break
        
        #! Obtener código
        codigo = obtener_codigo_del_usuario()
        
        if not codigo.strip():
            print("\nCodigo vacio. Intenta nuevamente.")
            continue
        
        print("\nAnalizando...")
        
        #! Analizar con auto-corrección
        mlp = analizar_con_autocorreccion(mlp, codigo)
    
    print("\nGracias por usar el analizador.")
    print("Todo el conocimiento adquirido ha sido guardado.\n")


if __name__ == "__main__":
    main()