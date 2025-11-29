def detectar_busqueda_binaria(codigo):
    """Detecta si es búsqueda binaria (mantiene low, high, mid)"""
    return ('low' in codigo or 'start' in codigo) and ('high' in codigo or 'end' in codigo) and ('mid' in codigo or 'middle' in codigo)


def detectar_division_binaria(codigo):
    """Detecta división binaria del rango (// 2, / 2, >> 1)"""
    return ('// 2' in codigo or '/ 2' in codigo or '>> 1' in codigo)


def contar_loops(codigo):
    """Cuenta loops for y while en el código"""
    count = 0
    lineas = codigo.split('\n')
    for linea in lineas:
        linea_limpia = linea.strip()
        if linea_limpia.startswith('for ') or linea_limpia.startswith('while '):
            count += 1
    return count


def detectar_recursion(codigo):
    """Detecta si hay llamadas recursivas"""
    lineas = codigo.split('\n')
    nombre_funcion = ""
    
    # Extraer nombre de la función
    for linea in lineas:
        linea_limpia = linea.strip()
        if linea_limpia.startswith('def '):
            partes = linea_limpia.split('(')
            if len(partes) > 0:
                nombre_funcion = partes[0].replace('def ', '').strip()
            break
    
    if not nombre_funcion:
        return False
    
    # Buscar la llamada a la función en el cuerpo (después de la definición)
    encontrada_def = False
    for linea in lineas:
        if linea.strip().startswith('def '):
            encontrada_def = True
            continue
        
        # Después de encontrar la definición, buscar la llamada
        if encontrada_def:
            sin_comentario = linea.split('#')[0]
            if nombre_funcion + '(' in sin_comentario:
                return True
    
    return False


def contar_operaciones(codigo):
    """Cuenta operaciones basicas (=, <, >, ==, +, -, *, /)"""
    count = 0
    operadores = ['=', '<', '>', '==', '+', '-', '*', '/']
    
    lineas = codigo.split('\n')
    for linea in lineas:
        for op in operadores:
            count += linea.count(op)
    
    return count


def extraer_nombre_funcion(codigo):
    """Extrae el nombre de la función"""
    lineas = codigo.split('\n')
    for linea in lineas:
        linea_limpia = linea.strip()
        if linea_limpia.startswith('def '):
            partes = linea_limpia.split('(')
            if len(partes) > 0:
                return partes[0].replace('def ', '').strip()
    return "desconocido"


def analizar_complejidad_manual(loops, recursion, operaciones, codigo=""):
    """Clasifica complejidad según heurísticas mejoradas"""
    
    # Caso especial: detectar búsqueda binaria
    if detectar_busqueda_binaria(codigo):
        return "O(log n)"
    
    # Heurística mejorada: el número de loops es el factor más importante
    if loops >= 3:
        return "O(n^3)"  # Triple loop
    elif loops == 2:
        return "O(n^2)"  # Double loop
    elif loops == 1 and recursion:
        # Un loop + recursión generalmente es O(n log n)
        return "O(n log n)"
    elif loops == 1:
        return "O(n)"  # Un solo loop
    elif recursion and loops == 0:
        # Recursión sin loops: generalmente O(log n) o O(n)
        # Si hay muchas operaciones, es más probable O(n)
        if operaciones > 15:
            return "O(n)"
        return "O(log n)"
    else:
        # Sin loops ni recursión
        return "O(1)"


def analizar_codigo(codigo):
    """Función principal que analiza el código"""
    
    loops = contar_loops(codigo)
    recursion = detectar_recursion(codigo)
    operaciones = contar_operaciones(codigo)
    nombre = extraer_nombre_funcion(codigo)
    complejidad = analizar_complejidad_manual(loops, recursion, operaciones, codigo)
    
    resultado = {
        "nombre": nombre,
        "loops": loops,
        "recursion": recursion,
        "operaciones": operaciones,
        "complejidad": complejidad
    }
    
    return resultado