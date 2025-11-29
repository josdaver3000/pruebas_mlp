# 🧠 Analizador de Complejidad Algorítmica con Red Neuronal

Analiza automáticamente la complejidad O(n) de algoritmos Python usando análisis estático y una Red Neuronal Multicapa.

## 📊 Características Principales

- ✅ **Detección automática de complejidad**: O(log n), O(n), O(n log n), O(n²)
- ✅ **Análisis estático inteligente**: Detección de loops, recursión y búsqueda binaria
- ✅ **Red Neuronal MLP**: Validación y predicción con 96.43% de precisión
- ✅ **API REST Flask**: Integrable con N8N y otros sistemas
- ✅ **Bot de Telegram**: Interfaz conversacional para análisis en tiempo real

## 🚀 Instalación

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
python api.py  # Escucha en http://localhost:5000
```

## 💻 Uso

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

**Resultado:**
```
Análisis estático:        O(log n)
Predicción MLP:           O(log n)
¿Coinciden?               ✓ SI
```

### Opción 2: API REST

```bash
python api.py
```

**Request:**
```bash
curl -X POST http://localhost:5000/analizar \
  -H "Content-Type: application/json" \
  -d '{"codigo": "def busqueda_lineal(arr, x):\n    for i in range(len(arr)):\n        if arr[i] == x:\n            return i\n    return -1"}'
```

**Response:**
```json
{
  "nombre": "busqueda_lineal",
  "loops": 1,
  "recursion": "No",
  "operaciones": 5,
  "complejidad": "O(n)",
  "exito": true
}
```

### Opción 3: Programáticamente

```python
from analizador import analizar_codigo

codigo = """def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr"""

resultado = analizar_codigo(codigo)
print(resultado['complejidad'])  # O(n^2)
```

## 🧠 Cómo Funciona

### 1. **Análisis Estático**
- Cuenta loops (for, while)
- Detecta recursión
- Identifica búsqueda binaria
- Cuenta operaciones básicas

### 2. **Extracción de Características**
Convierte el análisis en 8 características normalizadas:
- Loops ponderados
- Recursión ponderada
- Operaciones normalizadas
- Y 5 más...

### 3. **Red Neuronal MLP**
```
Entrada (8 features) → Capa Oculta (8 neuronas) → Salida (4 clases)
```
- **Arquitectura**: 8 → 8 → 4
- **Función de activación**: Sigmoid
- **Learning rate**: 0.1
- **Épocas de entrenamiento**: 3000

## 📁 Estructura del Proyecto

```
pruebas perceptron/actualizacion/
├── mlp.py                # Clase de la Red Neuronal
├── analizador.py         # Análisis estático del código
├── main.py               # Interfaz interactiva principal
├── entrenamiento.py      # Carga y normalización del dataset
├── api.py                # API REST Flask
├── tiempo.py             # Generador del dataset
├── algoritmos.py         # Algoritmos de referencia
├── recursos.csv          # Dataset de entrenamiento (112 muestras)
├── test_busqueda.py      # Tests
├── debug_recursion.py    # Herramientas de debug
└── README.md             # Este archivo
```

## 📊 Dataset

**112 muestras** generadas a partir de 8 algoritmos:
- Búsqueda lineal (O(n))
- Búsqueda binaria (O(log n))
- Bubble Sort (O(n²))
- Selection Sort (O(n²))
- Insertion Sort (O(n²))
- Merge Sort (O(n log n))
- Quick Sort (O(n log n))
- Dijkstra (O(n²))

Cada uno con 16 tamaños diferentes (10 → 660 elementos)

## 🎯 Precisión

- **Entrenamiento**: 96.43% (109/112 correctas)
- **Algoritmos probados**: 100% (búsqueda lineal, binaria, bubble sort, merge sort)

## 🔌 Integración con N8N y Telegram

Ver documentación completa en [N8N_TELEGRAM_SETUP.md](./N8N_TELEGRAM_SETUP.md)

Resumen rápido:
1. Instalar N8N: `npm install -g n8n`
2. Crear bot en Telegram con @BotFather
3. Ejecutar API: `python api.py`
4. Configurar workflow en N8N
5. ¡Listo! Tu bot responde en Telegram

## 🔍 Detección Automática

### Complejidad O(log n)
```python
# Detecta búsqueda binaria por patrones: low, high, mid
low, high = 0, len(arr) - 1
while low <= high:
    mid = (low + high) // 2  # ← Detecta división binaria
```

### Complejidad O(n)
```python
# 1 loop sin recursión
for i in range(len(arr)):  # ← 1 loop
    if arr[i] == x:        # ← Sin recursión
        return i
```

### Complejidad O(n log n)
```python
# 1 loop + recursión
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])      # ← Recursión
    right = merge_sort(arr[mid:])     # ← Recursión
    while ...:                        # ← 1 loop
```

### Complejidad O(n²)
```python
# 2+ loops
for i in range(n):        # ← Loop 1
    for j in range(n):    # ← Loop 2 = O(n²)
        if a[i] > a[j]:
            swap(a, i, j)
```

## 🛠️ Troubleshooting

| Problema | Solución |
|---|---|
| "ModuleNotFoundError" | Ejecuta: `pip install -r requirements.txt` |
| API no responde | Asegúrate que `python api.py` está corriendo |
| Recursión no detectada | Verifica que la función se llama a sí misma sin `def` en la línea |
| MLP predice mal | Normal: usa análisis estático como predicción principal |

## 📚 Referencias Teóricas

- **Análisis de complejidad**: [Big O Notation](https://en.wikipedia.org/wiki/Big_O_notation)
- **Redes Neuronales**: [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
- **Propagación hacia atrás**: [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)

## 👤 Autor

David

## 📄 Licencia

MIT

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

**¿Preguntas?** Abre un issue en GitHub.
