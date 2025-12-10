
#*==================== ANÁLISIS ESTÁTICO DE COMPLEJIDAD ===================
#* Este módulo implementa heurísticas para detectar la complejidad Big-O
#* analizando características del código sin ejecutarlo


#*===== DETECCIÓN DE PATRONES =====

def detectar_busqueda_binaria(codigo):
    #! Identifica si el código implementa búsqueda binaria
    #! Busca los patrones típicos: low, high, mid (o variantes)
    return ('low' in codigo or 'start' in codigo) and ('high' in codigo or 'end' in codigo) and ('mid' in codigo or 'middle' in codigo)


def detectar_division_binaria(codigo):
    #! Detecta si hay división por 2 (característica de O(log n))
    #! Busca: // 2, / 2, >> 1
    return ('// 2' in codigo or '/ 2' in codigo or '>> 1' in codigo)


#*===== CONTEO DE LOOPS =====

def contar_loops(codigo):
    #! Cuenta cuántos bucles for o while hay en el código
    #! No cuenta anidamiento, solo la cantidad total
    count = 0
    lineas = codigo.split('\n')
    for linea in lineas:
        linea_limpia = linea.strip()
        if linea_limpia.startswith('for ') or linea_limpia.startswith('while '):
            count += 1
    return count


def contar_loops_anidados_reales(codigo):
    #! Detecta la profundidad máxima de loops anidados
    #! Esto es crucial: O(n) = 1 loop, O(n^2) = 2 loops anidados, etc.
    
    lineas = codigo.split('\n')
    stack_indentacion = []
    profundidad_maxima = 0

    for linea in lineas:
        if not linea.strip():
            continue
        
        #! Calcula el nivel de indentación (espacios en blanco)
        espacios = len(linea) - len(linea.lstrip())
        nivel_indentacion = espacios // 4
        linea_stripped = linea.strip()

        if linea_stripped.startswith('for ') or linea_stripped.startswith('while '):
            #! Filtra loops en niveles anteriores (ya no están anidados)
            stack_indentacion = [ind for ind in stack_indentacion if ind < nivel_indentacion]
            stack_indentacion.append(nivel_indentacion)
            profundidad_actual = len(stack_indentacion)
            if profundidad_actual > profundidad_maxima:
                profundidad_maxima = profundidad_actual
        else:
            #! Ajusta el stack si salimos de un nivel de indentación
            stack_indentacion = [ind for ind in stack_indentacion if ind < nivel_indentacion]

    return profundidad_maxima


#*===== DETECCIÓN DE RECURSIÓN =====

def detectar_recursion(codigo):
    #! Detecta si una función se llama a sí misma (recursión)
    
    lineas = codigo.split('\n')
    nombre_funcion = ""
    
    #! Extrae el nombre de la función
    for linea in lineas:
        linea_limpia = linea.strip()
        if linea_limpia.startswith('def '):
            partes = linea_limpia.split('(')
            if len(partes) > 0:
                nombre_funcion = partes[0].replace('def ', '').strip()
            break
    
    if not nombre_funcion:
        return False
    
    #! Busca la llamada recursiva en el cuerpo de la función
    encontrada_def = False
    for linea in lineas:
        if linea.strip().startswith('def '):
            encontrada_def = True
            continue
        
        #! Después de encontrar la definición, busca la llamada
        if encontrada_def:
            sin_comentario = linea.split('#')[0]
            if nombre_funcion + '(' in sin_comentario:
                return True
    
    return False


#*===== CONTEO DE OPERACIONES =====

def contar_operaciones(codigo):
    #! Cuenta operaciones básicas en el código
    #! Incluye: asignaciones, comparaciones, aritméticas
    
    count = 0
    operadores = ['=', '<', '>', '==', '+', '-', '*', '/']
    
    lineas = codigo.split('\n')
    for linea in lineas:
        for op in operadores:
            count += linea.count(op)
    
    return count


#*===== UTILIDADES =====

def extraer_nombre_funcion(codigo):
    #! Extrae el nombre de la función definida en el código
    lineas = codigo.split('\n')
    for linea in lineas:
        linea_limpia = linea.strip()
        if linea_limpia.startswith('def '):
            partes = linea_limpia.split('(')
            if len(partes) > 0:
                return partes[0].replace('def ', '').strip()
    return "desconocido"


#*===== CLASIFICACIÓN DE COMPLEJIDAD =====

def analizar_complejidad_manual(recursion, codigo=""):
    #! Clasifica la complejidad basándose en heurísticas
    #! Usa: anidamiento de loops, recursión, patrones especiales
    
    #! Caso especial: búsqueda binaria siempre es O(log n)
    if detectar_busqueda_binaria(codigo):
        return "O(log n)"
    
    #! Calcula la profundidad de loops anidados (muy importante)
    profundidad_anidamiento = contar_loops_anidados_reales(codigo)
    
    #! Decisiones basadas en recursión + loops
    if recursion and profundidad_anidamiento == 0:
        if detectar_division_binaria(codigo):
            return "O(log n)"
        return "O(n)"
    elif recursion and profundidad_anidamiento == 1:
        return "O(n log n)"

    #! Decisiones basadas en loops puros
    if profundidad_anidamiento == 3:
        return "O(n^3)"
    elif profundidad_anidamiento == 2:
        return "O(n^2)"
    elif profundidad_anidamiento == 1:
        return "O(n)"
    else:
        return "O(1)"


#*===== EXTRACCIÓN DE CARACTERÍSTICAS PARA MLP =====

def extraer_caracteristicas_para_mlp(codigo):
    #! Extrae 8 características REALES del código
    #! Estas características se usan como entrada para la red neuronal
    #! No son complejidades directas, son métricas del código
    
    lineas = codigo.split('\n')
    lineas_limpias = [l.strip() for l in lineas if l.strip() and not l.strip().startswith('#')]
    
    if len(lineas_limpias) == 0:
        return [0.0] * 8
    
    #! Feature 1: Profundidad máxima de loops anidados (normalizado 0-3)
    profundidad_anidamiento = contar_loops_anidados_reales(codigo)
    feat_1 = min(profundidad_anidamiento / 3.0, 1.0)
    
    #! Feature 2: Número total de loops (normalizado 0-4)
    total_loops = contar_loops(codigo)
    feat_2 = min(total_loops / 4.0, 1.0)
    
    #! Feature 3: Tiene recursión (binario: 0 o 1)
    feat_3 = 1.0 if detectar_recursion(codigo) else 0.0
    
    #! Feature 4: Número de llamadas recursivas (normalizado 0-3)
    nombre_funcion = extraer_nombre_funcion(codigo)
    llamadas = 0
    if nombre_funcion and nombre_funcion != "desconocido":
        for linea in lineas_limpias:
            if not linea.startswith('def '):
                llamadas += linea.count(nombre_funcion + '(')
    feat_4 = min(llamadas / 3.0, 1.0)
    
    #! Feature 5: Patrón de búsqueda binaria (binario: 0 o 1)
    feat_5 = 1.0 if detectar_busqueda_binaria(codigo) else 0.0
    
    #! Feature 6: División del espacio (binario: 0 o 1)
    feat_6 = 1.0 if detectar_division_binaria(codigo) else 0.0
    
    #! Feature 7: Operaciones por línea de código (normalizado 0-5)
    operaciones = contar_operaciones(codigo)
    ops_por_linea = operaciones / len(lineas_limpias)
    feat_7 = min(ops_por_linea / 5.0, 1.0)
    
    #! Feature 8: Patrón de merge (binario: 0 o 1)
    #! Detecta palabras clave de algoritmos tipo merge sort
    palabras_merge = ['merge', 'left', 'right', 'extend', 'append']
    codigo_lower = codigo.lower()
    coincidencias = sum(1 for p in palabras_merge if p in codigo_lower)
    feat_8 = 1.0 if coincidencias >= 3 else 0.0
    
    return [feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, feat_7, feat_8]


#*===== FUNCIÓN PRINCIPAL =====

def analizar_codigo(codigo):
    #! Función principal: analiza código y retorna todas las métricas
    
    loops = contar_loops(codigo)
    recursion = detectar_recursion(codigo)
    operaciones = contar_operaciones(codigo)
    nombre = extraer_nombre_funcion(codigo)
    complejidad = analizar_complejidad_manual(recursion, codigo)
    
    #! Extrae características para la MLP
    caracteristicas_mlp = extraer_caracteristicas_para_mlp(codigo)
    
    resultado = {
        "nombre": nombre,
        "loops": loops,
        "recursion": recursion,
        "operaciones": operaciones,
        "complejidad": complejidad,
        "caracteristicas_mlp": caracteristicas_mlp
    }
    
    return resultado