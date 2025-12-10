# Analizador de Complejidad Algorítmica con Red Neuronal

Analiza automáticamente la complejidad O(n) de algoritmos Python usando análisis estático y una Red Neuronal Multicapa (MLP).

## Características Principales

- Detección automática de complejidad: O(log n), O(n), O(n log n), O(n^2)
- Análisis estático inteligente: Detección de loops, recursión y búsqueda binaria
- Red Neuronal MLP: Validación y predicción con precisión verificada
- API REST Flask: Integrable con N8N y otros sistemas
- Bot de Telegram: Interfaz conversacional para análisis en tiempo real

## Instalación

### Requisitos
- Python 3.8+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/ADA-projects-exercises-main.git
cd "pruebas perceptron/actualizacion"

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la aplicación interactiva
python main.py

# 4. O ejecutar como API
python api_n8n.py
# El API escucha en http://localhost:5000
```

## Uso

### Opción 1: Interfaz Interactiva

```bash
python main.py
```

Pega tu código y escribe `FIN`:

```python
def busqueda_binaria(arr, x):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1
FIN
```

Resultado:
```
Análisis estático:        O(log n)
Predicción MLP:           O(log n)
¿Coinciden?               SI
```

### Opción 2: API REST

```bash
python api_n8n.py
```

Request:
```bash
curl -X POST http://localhost:5000/analizar \
  -H "Content-Type: application/json" \
  -d '{"codigo": "def busqueda_lineal(arr, x):\n    for i in range(len(arr)):\n        if arr[i] == x:\n            return i\n    return -1"}'
```

Response:
```json
{
  "nombre": "busqueda_lineal",
  "loops": 1,
  "recursion": false,
  "operaciones": 5,
  "complejidad": "O(n)",
  "exito": true
}
```

### Opción 3: Uso Programático

```python
from analizador import analizar_codigo
from mlp import MLP

codigo = """def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr"""

resultado = analizar_codigo(codigo)
print(f"Complejidad: {resultado['complejidad']}")

# Cargar modelo y predecir
mlp = MLP.cargar("modelo_mlp.json")
pred = mlp.predict(resultado['caracteristicas_mlp'])
```

## Estructura del Proyecto

```
actualizacion/
├── main.py                      # Interfaz interactiva principal
├── api_n8n.py                   # API REST Flask
├── mlp.py                        # Red neuronal multicapa
├── analizador.py                 # Análisis estático de complejidad
├── entrenamiento_combinado.py    # Entrenamiento de la MLP
├── algoritmos.py                 # Algoritmos de prueba
├── modelo_mlp.json              # Modelo entrenado guardado
├── recursos.csv                 # Dataset para entrenamiento
└── requirements.txt             # Dependencias
```

## Cómo Funciona

### 1. Análisis Estático

El módulo `analizador.py` examina el código fuente sin ejecutarlo:

- Cuenta loops for y while
- Detecta profundidad de anidamiento (O(n), O(n^2), etc.)
- Identifica recursión
- Busca patrones especiales (búsqueda binaria, merge sort)
- Extrae 8 características del código para la MLP

```python
# Características extraídas:
# 1. Profundidad de loops anidados
# 2. Número total de loops
# 3. Presencia de recursión
# 4. Número de llamadas recursivas
# 5. Patrón de búsqueda binaria
# 6. División del espacio (// 2)
# 7. Operaciones por línea
# 8. Patrón de merge
```

### 2. Red Neuronal Multicapa (MLP)

La red neuronal tiene la siguiente arquitectura:

```
Entrada (8 features) → Capa Oculta (16 neuronas) → Salida (4 clases)
```

Capas:
- Entrada: 8 características extraídas del código
- Oculta: 16 neuronas con activación sigmoid
- Salida: 4 neuronas (una por cada clase de complejidad)

Funciones de activación:
- Sigmoid: Convierte valores a rango [0, 1]
- Derivada de sigmoid: Para retropropagación

Proceso de aprendizaje:
- Forward pass: Propaga entrada hacia la salida
- Backward pass (retropropagación): Calcula gradientes y actualiza pesos
- Learning rate: 0.1 (velocidad de aprendizaje)

### 3. Clasificación de Complejidad

Las clases soportadas son:
- O(log n): Búsqueda binaria, logarítmicas
- O(n): Lineales, un loop
- O(n log n): Merge sort, quicksort promedio
- O(n^2): Bubble sort, nested loops

### 4. Persistencia del Modelo

El modelo entrenado se guarda en `modelo_mlp.json`:

```python
{
  "arquitectura": {
    "n_inputs": 8,
    "n_hidden": 16,
    "n_outputs": 4,
    "lr": 0.1
  },
  "pesos": {
    "w1": [...],  # Pesos entrada → oculta
    "b1": [...],  # Sesgos capa oculta
    "w2": [...],  # Pesos oculta → salida
    "b2": [...]   # Sesgos capa salida
  }
}
```

Ventajas:
- Entrena una sola vez
- Carga el modelo instantáneamente
- Reutilizable en diferentes sesiones

## Entrenamiento de la MLP

El script `entrenamiento_combinado.py` entrena la red:

```bash
python entrenamiento_combinado.py
```

Proceso:
1. Carga datos de `recursos.csv`
2. Extrae características del código de algoritmos conocidos
3. Normaliza características al rango [0, 1]
4. Entrena 5000 épocas
5. Evalúa accuracy por clase
6. Guarda modelo en `modelo_mlp.json`

Dataset:
- 112 muestras de 8 algoritmos diferentes
- Balanceado entre 4 clases de complejidad
- Características extraídas del código fuente

## Algoritmos Incluidos

El proyecto incluye estos 8 algoritmos de prueba:

1. **busqueda_lineal** - O(n): Recorre array secuencialmente
2. **busqueda_binaria** - O(log n): Divide espacio de búsqueda
3. **bubble_sort** - O(n^2): Compara pares consecutivos
4. **selection_sort** - O(n^2): Encuentra mínimo en cada iteración
5. **insertion_sort** - O(n^2): Inserta elementos ordenadamente
6. **merge_sort** - O(n log n): Divide y conquista
7. **quick_sort** - O(n log n): Particiona alrededor de pivote
8. (Potencialmente más)

## Archivos Principales

### main.py
Interfaz interactiva principal del usuario. Permite:
- Pegar código Python
- Analizar complejidad automáticamente
- Comparar análisis estático con predicción MLP
- Guardar y cargar modelos

### api_n8n.py
API REST con Flask para integración externa:
- GET /: Información del API
- GET /salud: Health check
- POST /analizar: Analiza código enviado en JSON

Diseñado para integración con N8N y otros sistemas.

### mlp.py
Implementación de la Red Neuronal Multicapa:
- Forward pass: Calcula salida
- Backward pass: Retropropagación del error
- Métodos de persistencia: guardar/cargar modelos
- Re-entrenamiento incremental

Características:
- Sin dependencias externas (solo math, random, json)
- Completamente interpretable
- Pesos guardables en JSON

### analizador.py
Análisis estático de complejidad:
- Detecta loops y anidamiento
- Identifica recursión
- Reconoce patrones especiales
- Extrae características para MLP

Funciones principales:
- `analizar_codigo()`: Análisis completo
- `extraer_caracteristicas_para_mlp()`: Features para red
- `contar_loops_anidados_reales()`: Profundidad de anidamiento

### entrenamiento_combinado.py
Entrenamiento de la red neuronal:
- Carga dataset combinado
- Normaliza características
- Entrena MLP
- Evalúa resultados
- Guarda modelo

### algoritmos.py
Implementaciones de algoritmos de prueba:
- Búsqueda lineal y binaria
- Sorting: bubble, merge
- Todos con complejidad bien definida

## Dependencias

Ver `requirements.txt`:
```
flask
flask-cors
requests
numpy
```

Solo Flask/CORS son estrictamente necesarios. La MLP se implementa desde cero.

## Precisión y Resultados

El modelo alcanza alta precisión en el dataset de entrenamiento:
- Accuracy global: 90%+
- Detección perfecta de patrones especiales
- Manejo robusto de variaciones de código

Limitaciones:
- Entrenado con 8 algoritmos específicos
- Código debe tener estructura clara
- No detecta complejidad O(1) o O(n^3+)

## Cómo Mejorar

1. Ampliar dataset con más algoritmos
2. Añadir más características al análisis
3. Aumentar neuronas en capa oculta
4. Ajustar learning rate
5. Entrenar más épocas
6. Validación cruzada para evitar overfitting

## Notas Técnicas

### Forward Pass
```
h = sigmoid(X * w1 + b1)
y = sigmoid(h * w2 + b2)
```

### Backward Pass
```
error_output = expected - output
delta_output = error_output * sigmoid'(output)
error_hidden = delta_output * w2.T
delta_hidden = error_hidden * sigmoid'(hidden)

w2 += lr * delta_output * h.T
b2 += lr * delta_output
w1 += lr * delta_hidden * X.T
b1 += lr * delta_hidden
```

### Normalización
```
X_normalized = (X - min) / (max - min)
```

## Ejemplo de Ejecución

```bash
$ python main.py

Ingresa código Python (termina con FIN):
def busqueda_binaria(arr, x):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1

FIN

================================
ANÁLISIS DE COMPLEJIDAD
================================

Nombre:                   busqueda_binaria
Loops detectados:         1
Recursión:                No
Operaciones:              5

Análisis estático:        O(log n)
Predicción MLP:           O(log n)

¿Coinciden?               SI

================================
```

## Integración con N8N

El API puede integrarse con N8N:

1. Iniciar API: `python api_n8n.py`
2. En N8N, crear nodo HTTP POST
3. URL: `http://localhost:5000/analizar`
4. Body: `{"codigo": "<código Python>"}`
5. Procesar respuesta JSON

Ejemplo de flujo N8N:
```
[Entrada de usuario] 
    → [HTTP POST a API]
    → [Procesar respuesta]
    → [Mostrar resultado]
```

## Licencia

Proyecto académico para análisis de algoritmos.

## Autor

David (Proyecto ADA - Análisis y Diseño de Algoritmos)

---

Última actualización: Diciembre 2025

Para preguntas o mejoras, revisar el código fuente en los módulos principales.
