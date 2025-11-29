import math
import random


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


class MLP:

    def __init__(self, n_inputs, n_hidden, n_outputs, lr=0.05, seed=42):
        random.seed(seed)
        self.lr = lr
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.w1 = [[random.uniform(-1, 1) for _ in range(n_inputs)] for _ in range(n_hidden)]
        self.b1 = [random.uniform(-1, 1) for _ in range(n_hidden)]

        self.w2 = [[random.uniform(-1, 1) for _ in range(n_hidden)] for _ in range(n_outputs)]
        self.b2 = [random.uniform(-1, 1) for _ in range(n_outputs)]

    def forward(self, inputs):
        self.inputs = inputs[:]
        
        self.h = []
        for i in range(self.n_hidden):
            s = sum(w * x for w, x in zip(self.w1[i], inputs)) + self.b1[i]
            self.h.append(sigmoid(s))
        
        self.o = []
        for i in range(self.n_outputs):
            s = sum(w * hi for w, hi in zip(self.w2[i], self.h)) + self.b2[i]
            self.o.append(sigmoid(s))
        
        return self.o

    def backward(self, expected):
        error_o = [expected[i] - self.o[i] for i in range(self.n_outputs)]
        delta_o = [error_o[i] * dsigmoid(self.o[i]) for i in range(self.n_outputs)]
        
        error_h = [0.0] * self.n_hidden
        for j in range(self.n_hidden):
            s = 0.0
            for i in range(self.n_outputs):
                s += delta_o[i] * self.w2[i][j]
            error_h[j] = s
        
        delta_h = [error_h[j] * dsigmoid(self.h[j]) for j in range(self.n_hidden)]
        
        for i in range(self.n_outputs):
            for j in range(self.n_hidden):
                self.w2[i][j] += self.lr * delta_o[i] * self.h[j]
            self.b2[i] += self.lr * delta_o[i]
        
        for j in range(self.n_hidden):
            for k in range(self.n_inputs):
                self.w1[j][k] += self.lr * delta_h[j] * self.inputs[k]
            self.b1[j] += self.lr * delta_h[j]

    def train_epoch(self, X, Y):
        total_loss = 0.0
        for x, y in zip(X, Y):
            out = self.forward(x)
            total_loss += sum((y[i] - out[i]) ** 2 for i in range(self.n_outputs))
            self.backward(y)
        return total_loss / len(X)

    def predict(self, x):
        out = self.forward(x)
        return max(range(len(out)), key=lambda i: out[i])