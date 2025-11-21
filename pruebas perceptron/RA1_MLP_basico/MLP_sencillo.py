import math #para aplicar las funciones matematicas
import random #para implementar los pesos con valores aleatorios

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x)) #transforma cualquier numero real en un valor entre 0 y 1
                                      #emplea una linealidad que permite que el mlp aprenda relaciones complejas

def dsigmoid(y): #representa la derivada de sigmoid(x), pero con el valor Y = sigmoid(x) ya calculado, para priorizar la eficiencia
    return y * (1 - y)
#esto se usara en el backpropagation para actualizar los pesos del modelo
    

class MLP:

    def __init__(self, n_inputs, n_hidden, n_outputs, lr=0.05, seed=42):
        random.seed(seed)
        self.lr = lr
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

# MODIFICACIÓN: debido a bajo aprendizaje y saturación inicial del MLP,
# se redujo el rango de inicialización para evitar gradientes débiles
# y se fijaron los bias en 0 para garantizar simetría inicial.
#=======================================codigo anterior======================================
        
        self.w1 = [[random.uniform(-1,1) for _ in range(n_inputs)] for _ in range(n_hidden)]
        self.b1 = [random.uniform(-1,1) for _ in range(n_hidden)]

        self.w2 = [[random.uniform(-1,1) for _ in range(n_hidden)] for _ in range(n_outputs)]
        self.b2 = [random.uniform(-1,1) for _ in range(n_outputs)]

#=============================================================================================
        
        # #pesos y sesgos de entrada en w1 y b1: dimencion = hidden x inputs
        # #w1 = matriz de pesos entre capa de entrada y capa oculta
        # #b1 = vector de bias de la capa oculta
        # self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(n_inputs)] for _ in range(n_hidden)]
        # self.b1 = [0.0 for _ in range(n_hidden)]

        # #pesos y sesgos de salida en w2 y b2: dimencion = outputs x hidden
        # #w2 =  matriz de pesos entre capa oculta y capa de salida
        # #b2 = vector de bias de la capa de salida
        # self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(n_hidden)] for _ in range(n_outputs)]
        # self.b2 = [0.0 for _ in range(n_outputs)]

    def forward(self, inputs):
        # inputs: list len n_inputs
        self.inputs = inputs[:]  # guardar para backprop
        # capa oculta
        self.h = []
        for i in range(self.n_hidden):
            s = sum(w*x for w,x in zip(self.w1[i], inputs)) + self.b1[i]
            self.h.append(sigmoid(s))
        # salida
        self.o = []
        for i in range(self.n_outputs):
            s = sum(w*hi for w,hi in zip(self.w2[i], self.h)) + self.b2[i]
            self.o.append(sigmoid(s))
        return self.o

    def backward(self, expected):
        # expected: list len n_outputs (targets entre 0 y 1)
        # error salidas
        error_o = [expected[i] - self.o[i] for i in range(self.n_outputs)]
        delta_o = [error_o[i] * dsigmoid(self.o[i]) for i in range(self.n_outputs)]
        # error ocultas
        error_h = [0.0]*self.n_hidden
        for j in range(self.n_hidden):
            s = 0.0
            for i in range(self.n_outputs):
                s += delta_o[i] * self.w2[i][j]
            error_h[j] = s
        delta_h = [error_h[j] * dsigmoid(self.h[j]) for j in range(self.n_hidden)]
        # actualizar w2
        for i in range(self.n_outputs):
            for j in range(self.n_hidden):
                self.w2[i][j] += self.lr * delta_o[i] * self.h[j]
            self.b2[i] += self.lr * delta_o[i]
        # actualizar w1
        for j in range(self.n_hidden):
            for k in range(self.n_inputs):
                self.w1[j][k] += self.lr * delta_h[j] * self.inputs[k]
            self.b1[j] += self.lr * delta_h[j]

    def train_epoch(self, X, Y):
        # X: list of input vectors; Y: list of target vectors
        total_loss = 0.0
        for x, y in zip(X, Y):
            out = self.forward(x)
            # mse
            total_loss += sum((y[i]-out[i])**2 for i in range(self.n_outputs))
            self.backward(y)
        return total_loss / len(X) #si se elimina / len(x) los resultados del loss seran independiendiente, por el contrario si se deja se usa el promedio


    def predict(self, x):
        out = self.forward(x)
        # devolver índice con mayor valor
        return max(range(len(out)), key=lambda i: out[i])

