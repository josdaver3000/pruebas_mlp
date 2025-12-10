"""
API REST Flask para conectar con N8N
Ejecutar: python api_n8n.py
Acceder en: http://localhost:5000

Este módulo expone endpoints HTTP para que sistemas externos
como N8N puedan analizar algoritmos sin necesidad de Python
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from analizador import analizar_codigo
from entrenamiento import load_data, normalize, one_hot, mapeo
from mlp import MLP

app = Flask(__name__)
CORS(app)

#* Variable global para almacenar la red neuronal entrenada
#* Se carga una sola vez al iniciar la aplicación
mlp_global = None


def entrenar_mlp():
    #! Función de inicialización: entrena el MLP una sola vez
    #! Esto evita entrenar en cada request (muy lento)
    global mlp_global
    print("Cargando y normalizando datos...")
    X, Y_idx = load_data()
    X = normalize(X)
    n_outputs = len(set(Y_idx))
    Y = [one_hot(i, n_outputs) for i in Y_idx]
    
    print("Entrenando MLP...")
    mlp_global = MLP(n_inputs=8, n_hidden=16, n_outputs=n_outputs, lr=0.1)
    epochs = 5000
    
    for e in range(epochs):
        loss = mlp_global.train_epoch(X, Y)
        if e % 500 == 0:
            print(f"  Epoch {e:5d} - Loss {loss:.8f}")
    
    print(" MLP entrenado correctamente\n")


@app.route('/', methods=['GET'])
def inicio():
    #! Endpoint raíz: retorna información sobre el API
    #! Útil para verificar que el servicio está corriendo
    return jsonify({
        'nombre': 'Analizador de Complejidad Algorítmica API',
        'version': '1.0',
        'descripcion': 'Analiza la complejidad O(n) de algoritmos Python',
        'endpoints': {
            'GET /': 'Esta información',
            'GET /salud': 'Health check',
            'POST /analizar': 'Analizar complejidad de código'
        }
    }), 200


@app.route('/salud', methods=['GET'])
def salud():
    #! Endpoint de health check: verifica estado del servicio
    #! Retorna OK si la MLP está cargada correctamente
    return jsonify({
        'estado': 'OK',
        'servicio': 'Analizador de Complejidad',
        'mlp_cargado': mlp_global is not None
    }), 200


@app.route('/analizar', methods=['POST'])
def analizar():
    #! Endpoint principal: recibe código Python y retorna análisis de complejidad
    #! Combina análisis estático + predicción MLP
    #* Request esperado: {"codigo": "<código Python>"}
    
    print("RAW JSON ->", data)
    print("CODIGO ->", data.get("codigo"))

    try:
        print("FULL JSON BODY ->", request.data)

        data = request.json
        codigo = data.get('codigo', '').strip()
        if not codigo:
            return jsonify({
                'exito': False,
                'error': 'Código vacío',
                'mensaje': 'Por favor proporciona código Python'
            }), 400
        
        #! Eliminar BOM (Byte Order Mark) si existe en el código
        if codigo.startswith('\ufeff'):
            codigo = codigo[1:]
        
        #! Ejecutar análisis estático del código
        resultado = analizar_codigo(codigo)
        
        #! Retornar respuesta exitosa con los resultados
        return jsonify({
            'exito': True,
            'nombre': resultado['nombre'],
            'loops': resultado['loops'],
            'recursion': resultado['recursion'],
            'operaciones': resultado['operaciones'],
            'complejidad': resultado['complejidad'],
            'mensaje': f"La complejidad es {resultado['complejidad']}"
        }), 200
    
    except Exception as e:
        #! Si hay error, retornar respuesta de error 500
        return jsonify({
            'exito': False,
            'error': str(e),
            'mensaje': 'Error al analizar el código'
        }), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("ANALIZADOR DE COMPLEJIDAD - API REST")
    print("="*70)
    
    entrenar_mlp()
    
    print(" API iniciado")
    print("="*50)
    print(" Endpoints disponibles:")
    print("   - GET  /           (información)")
    print("   - GET  /salud      (health check)")
    print("   - POST /analizar   (analizar código)")
    print("\n Presiona Ctrl+C para detener\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
