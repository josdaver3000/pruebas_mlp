
#* ===== RED NEURONAL MULTICAPA (MLP) CON PERSISTENCIA =====
#* Implementación de un perceptrón multicapa que permite:
#*   - Entrenar la red neuronal desde cero
#*   - Guardar los pesos en JSON para reutilización
#*   - Cargar modelos ya entrenados sin necesidad de re-entrenar
#*   - Re-entrenar incrementalmente con nuevas muestras

import math 
import random #* para inicialización aleatoria de pesos
import json #* para persistencia de modelos



#*==================== FUNCIONES DE ACTIVACIÓN ===================
#* Las funciones de activación introducen no-linealidad en la red
#* permitiendo que aprenda patrones complejos

def sigmoid(x):
    #! Sigmoid: convierte cualquier valor a un rango entre 0 y 1
    #! Fórmula: 1 / (1 + e^-x)
    #! Ventaja: diferenciable para retropropagación
    return 1.0 / (1.0 + math.exp(-x))


def dsigmoid(y):
    #! Derivada de sigmoid: necesaria para el backpropagation
    #! Si ya tenemos y = sigmoid(x), su derivada es: y * (1 - y)
    return y * (1 - y)

#*==================== CLASE MLP ===================
class MLP:
    #! Red neuronal con una capa oculta
    #! Arquitectura: entrada -> capa oculta -> salida

    def __init__(self, n_inputs, n_hidden, n_outputs, lr=0.05, seed=42):
        #! Inicializa la arquitectura de la red
        #! Args:
        #!   n_inputs: número de características de entrada
        #!   n_hidden: número de neuronas en la capa oculta
        #!   n_outputs: número de clases de salida
        #!   lr: learning rate (velocidad de aprendizaje)
        #!   seed: para reproducibilidad
        
        random.seed(seed)
        self.lr = lr
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        #! Pesos de entrada a capa oculta (matriz n_hidden x n_inputs)
        self.w1 = [[random.uniform(-1, 1) for _ in range(n_inputs)] for _ in range(n_hidden)]
        #! Sesgos de capa oculta
        self.b1 = [random.uniform(-1, 1) for _ in range(n_hidden)]

        #! Pesos de capa oculta a salida (matriz n_outputs x n_hidden)
        self.w2 = [[random.uniform(-1, 1) for _ in range(n_hidden)] for _ in range(n_outputs)]
        #! Sesgos de capa de salida
        self.b2 = [random.uniform(-1, 1) for _ in range(n_outputs)]

    def forward(self, inputs):
        #! Forward pass: propaga la entrada hacia adelante
        #! Calcula la salida de la red dado un input
        
        self.inputs = inputs[:]
        
        #! Capa oculta: entrada * w1 + b1, aplicar sigmoid
        self.h = []
        for i in range(self.n_hidden):
            #! Suma ponderada: cada neurona combina todas las entradas
            s = sum(w * x for w, x in zip(self.w1[i], inputs)) + self.b1[i]
            #! Aplica sigmoid para introducir no-linealidad
            self.h.append(sigmoid(s))
        
        #! Capa de salida: hidden * w2 + b2, aplicar sigmoid
        self.o = []
        for i in range(self.n_outputs):
            s = sum(w * hi for w, hi in zip(self.w2[i], self.h)) + self.b2[i]
            self.o.append(sigmoid(s))
        
        return self.o

    def backward(self, expected):
        #! Backward pass (retropropagación)
        #! Calcula gradientes y actualiza pesos basándose en el error
        
        #! Error en capa de salida: diferencia entre esperado y predicho
        error_o = [expected[i] - self.o[i] for i in range(self.n_outputs)]
        #! Delta de salida: error * derivada de sigmoid
        delta_o = [error_o[i] * dsigmoid(self.o[i]) for i in range(self.n_outputs)]
        
        #! Retropropagar error hacia capa oculta
        error_h = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            #! Suma ponderada del error de las neuronas de salida
            s = 0.0
            for i in range(self.n_outputs):
                s += delta_o[i] * self.w2[i][j]
            error_h[j] = s
        
        #! Delta de capa oculta: error * derivada de sigmoid
        delta_h = [error_h[j] * dsigmoid(self.h[j]) for j in range(self.n_hidden)]
        
        #! Actualizar pesos y sesgos de capa de salida
        for i in range(self.n_outputs):
            for j in range(self.n_hidden):
                self.w2[i][j] += self.lr * delta_o[i] * self.h[j]
            self.b2[i] += self.lr * delta_o[i]
        
        #! Actualizar pesos y sesgos de capa oculta
        for j in range(self.n_hidden):
            for k in range(self.n_inputs):
                self.w1[j][k] += self.lr * delta_h[j] * self.inputs[k]
            self.b1[j] += self.lr * delta_h[j]

    def train_epoch(self, X, Y):
        #! Entrena un época (una pasada sobre todo el dataset)
        #! Args:
        #!   X: lista de vectores de entrada
        #!   Y: lista de vectores esperados (one-hot encoded)
        
        total_loss = 0.0
        for x, y in zip(X, Y):
            #! Forward pass
            out = self.forward(x)
            #! Calcula error cuadrático
            total_loss += sum((y[i] - out[i]) ** 2 for i in range(self.n_outputs))
            #! Backward pass (retropropagación)
            self.backward(y)
        
        #! Retorna el error promedio
        return total_loss / len(X)

    def predict(self, x):
        #! Predicción: retorna el índice de la neurona de salida con mayor activación
        out = self.forward(x)
        #! Argmax: índice del valor máximo
        return max(range(len(out)), key=lambda i: out[i])
    
    #*==================== PERSISTENCIA DEL MODELO ===================
    
    def guardar(self, archivo="modelo_mlp.json"):
        #! Guarda los pesos y la arquitectura en un archivo JSON
        #! Permite cargar el modelo ya entrenado sin necesidad de re-entrenar
        
        modelo = {
            'arquitectura': {
                'n_inputs': self.n_inputs,
                'n_hidden': self.n_hidden,
                'n_outputs': self.n_outputs,
                'lr': self.lr
            },
            'pesos': {
                'w1': self.w1,
                'b1': self.b1,
                'w2': self.w2,
                'b2': self.b2
            }
        }
        
        try:
            with open(archivo, 'w') as f:
                json.dump(modelo, f, indent=2)
            return True
        except Exception as e:
            print(f"Error al guardar modelo: {e}")
            return False
    
    @classmethod
    def cargar(cls, archivo="modelo_mlp.json"):
        #! Carga un modelo previamente guardado
        #! Retorna una instancia de MLP con pesos inicializados
        
        try:
            with open(archivo, 'r') as f:
                modelo = json.load(f)
            
            #! Crear instancia con la arquitectura guardada
            arq = modelo['arquitectura']
            mlp = cls(
                n_inputs=arq['n_inputs'],
                n_hidden=arq['n_hidden'],
                n_outputs=arq['n_outputs'],
                lr=arq['lr']
            )
            
            #! Cargar los pesos entrenados
            pesos = modelo['pesos']
            mlp.w1 = pesos['w1']
            mlp.b1 = pesos['b1']
            mlp.w2 = pesos['w2']
            mlp.b2 = pesos['b2']
            
            return mlp
        
        except FileNotFoundError:
            print(f"Archivo '{archivo}' no encontrado.")
            return None
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            return None
    
    #*==================== RE-ENTRENAMIENTO INCREMENTAL ===================
    
    def entrenar_incremental(self, nuevos_X, nuevos_Y, epochs=1000):
        #! Re-entrena el modelo con nuevas muestras sin perder el aprendizaje anterior
        #! Útil para actualizar el modelo con datos nuevos
        
        print(f"\n Re-entrenamiento con {len(nuevos_X)} nuevas muestras...")
        
        for e in range(epochs):
            loss = self.train_epoch(nuevos_X, nuevos_Y)
            if e % 200 == 0:
                print(f"   Época {e:4d} | Loss: {loss:.8f}")
        
        print(f" Re-entrenamiento completado\n")
        return loss


#*==================== FUNCIONES DE UTILIDAD ===================

def existe_modelo_guardado(archivo="modelo_mlp.json"):
    #! Verifica si existe un archivo de modelo guardado
    import os
    return os.path.exists(archivo)






#* ==================== EJEMPLO DE USO ====================

if __name__ == "__main__":
    print("="*70)
    print("DEMOSTRACIÓN: Persistencia de MLP")
    print("="*70)
    
    #! Datos de ejemplo
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[1, 0], [0, 1], [0, 1], [1, 0]] 
    
    ARCHIVO = "modelo_test.json"
    
    #! 1. Intentar cargar modelo existente
    print("\n Intentando cargar modelo guardado...")
    mlp = MLP.cargar(ARCHIVO)
    
    if mlp:
        print(" Modelo cargado desde archivo")
    else:
        print(" No existe modelo, entrenando nuevo...")
        
        #! 2. Entrenar desde cero
        mlp = MLP(n_inputs=2, n_hidden=4, n_outputs=2, lr=0.5)
        
        for epoch in range(5000):
            loss = mlp.train_epoch(X, Y)
            if epoch % 1000 == 0:
                print(f"Época {epoch:5d} | Loss: {loss:.6f}")
        
        #! 3. Guardar modelo
        print("\n  Guardando modelo...")
        if mlp.guardar(ARCHIVO):
            print(f" Modelo guardado en '{ARCHIVO}'")
    
    #! 4. Probar predicciones
    print("\n Probando predicciones:")
    for x in X:
        pred = mlp.predict(x)
        print(f"   Input: {x} → Predicción: {pred}")
    
    #! 5. Re-entrenamiento (simulado)
    print("\n  Simulando re-entrenamiento incremental:")
    nuevos_X = [[0.1, 0.1], [0.9, 0.9]]
    nuevos_Y = [[0.9, 0.1], [0.9, 0.1]]
    
    mlp.entrenar_incremental(nuevos_X, nuevos_Y, epochs=1000)
    
    #! 6. Guardar modelo actualizado
    print("  Guardando modelo actualizado...")
    mlp.guardar(ARCHIVO)
    print(f" Modelo re-entrenado guardado")
