"""
API REST Flask para el Analizador de Complejidad Algorítmica
Endpoint: POST /analizar
"""

from flask import Flask, request, jsonify
from analizador import analizar_codigo
import sys

app = Flask(__name__)

@app.route('/analizar', methods=['POST'])
def analizar():
    """
    Analiza la complejidad de un algoritmo Python
    
    Request JSON:
    {
        "codigo": "def busqueda_lineal(arr, x):\n    for i in range(len(arr)):\n        if arr[i] == x:\n            return i\n    return -1"
    }
    
    Response JSON:
    {
        "nombre": "busqueda_lineal",
        "loops": 1,
        "recursion": "No",
        "operaciones": 5,
        "complejidad": "O(n)",
        "exito": true
    }
    """
    try:
        data = request.json
        codigo = data.get('codigo', '').strip()
        
        if not codigo:
            return jsonify({
                'error': 'Código vacío',
                'exito': False
            }), 400
        
        # Eliminar BOM si existe (por problemas de encoding)
        if codigo.startswith('\ufeff'):
            codigo = codigo[1:]
        
        # Analizar el código
        resultado = analizar_codigo(codigo)
        
        return jsonify({
            'nombre': resultado['nombre'],
            'loops': resultado['loops'],
            'recursion': 'Sí' if resultado['recursion'] else 'No',
            'operaciones': resultado['operaciones'],
            'complejidad': resultado['complejidad'],
            'exito': True
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Error al analizar: {str(e)}',
            'exito': False
        }), 500


@app.route('/salud', methods=['GET'])
def salud():
    """Health check - verifica que la API esté funcionando"""
    return jsonify({
        'estado': 'OK',
        'servicio': 'Analizador de Complejidad',
        'version': '1.0'
    }), 200


@app.route('/', methods=['GET'])
def inicio():
    """Información del API"""
    return jsonify({
        'nombre': 'Analizador de Complejidad Algorítmica',
        'version': '1.0',
        'descripcion': 'Analiza la complejidad O(n) de algoritmos Python',
        'endpoints': {
            'GET /': 'Esta información',
            'GET /salud': 'Health check',
            'POST /analizar': 'Analizar complejidad de código'
        },
        'ejemplo': {
            'method': 'POST',
            'url': '/analizar',
            'body': {
                'codigo': 'def ejemplo(arr):\n    for i in range(len(arr)):\n        print(arr[i])'
            }
        }
    }), 200


if __name__ == '__main__':
    # Ejecutar en 0.0.0.0:5000 para acceso externo
    print("🚀 Analizador de Complejidad API iniciado")
    print("📍 Escuchando en http://0.0.0.0:5000")
    print("📊 Endpoints disponibles:")
    print("   - GET  /           (información)")
    print("   - GET  /salud      (health check)")
    print("   - POST /analizar   (analizar código)")
    print("\n💡 Ejemplo:")
    print('   curl -X POST http://localhost:5000/analizar \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"codigo": "def test(x):\\n    for i in range(x):\\n        print(i)"}\'')
    print("\n✅ Presiona Ctrl+C para detener\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
