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
    
    for linea in lineas:
        linea_limpia = linea.strip()
        if linea_limpia.startswith('def '):
            partes = linea_limpia.split('(')
            if len(partes) > 0:
                nombre_funcion = partes[0].replace('def ', '').strip()
            break
    
    if not nombre_funcion:
        return False
    
    for linea in lineas:
        if nombre_funcion + '(' in linea and 'def ' not in linea:
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


def analizar_complejidad_manual(loops, recursion, operaciones):
    """Clasifica complejidad según heurísticas simples"""
    
    if loops >= 2:
        return "O(n^2)"
    elif loops == 1 and recursion:
        return "O(n log n)"
    elif loops == 1:
        return "O(n)"
    elif recursion:
        return "O(log n)"
    else:
        return "O(1)"


def analizar_codigo(codigo):
    """Función principal que analiza el código"""
    
    loops = contar_loops(codigo)
    recursion = detectar_recursion(codigo)
    operaciones = contar_operaciones(codigo)
    nombre = extraer_nombre_funcion(codigo)
    complejidad = analizar_complejidad_manual(loops, recursion, operaciones)
    
    resultado = {
        "nombre": nombre,
        "loops": loops,
        "recursion": recursion,
        "operaciones": operaciones,
        "complejidad": complejidad
    }
    
    return resultado