import numpy as np

class Network:
    def __init__(self):
        self.layers = []
        self.output = np.nan
        self.error = np.nan
        self.loss = np.nan

    def add(self, layer, x=None):
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[len(self.layers)-1].setInput(self.layers[len(self.layers)-2])
        else:
            if type(x) == np.ndarray:
                self.layers[len(self.layers) - 1].setInput(x)

    def forward(self, x=None):
        if type(x) == np.ndarray:
            self.layers[0].setInput(x)
        self.layers[len(self.layers)-1].forward()
        self.output = self.layers[len(self.layers)-1].output
        return self.output

    def calcError(self, X, Y):
        self.error = 0
        for i in range(len(X)):
            self.forward(X[i])
            self.error += np.sum(((Y[i]-self.output)**2.0)/2.0, axis=1)[0]
        self.error /= len(X)
        return self.error

    def calcLoss(self, X, Y):
        losses = []
        for i in range(len(X)):
            self.forward(X[i])
            losses.append(abs(Y[i]-self.output)[0][0])
        self.loss = max(losses)
        return self.loss

    def fit(self, X, Y, lr, epochs=10000, verbose=False):
        netInput = self.layers[0].inputs
        for epoch in range(epochs):
            for a in range(len(X)):
                self.forward(X[a])

                i = len(self.layers) - 1
                self.layers[len(self.layers)-1].target = Y[a]
                while i >= 0:
                    layerDeriv = self.layers[i].calcDeriv()
                    for j in range(len(self.layers[i].weights)):
                        for k in range(len(self.layers[i].weights[j])):
                            self.layers[i].weights[j][k] += -lr * layerDeriv[0][j] * self.layers[i].inputs[0][k]
                    for j in range(len(self.layers[i].biases[0])):
                        self.layers[i].biases[0][j] += -lr * layerDeriv[0][j]
                    i -= 1
            if verbose:
                print(f"Epoch: {epoch} | Loss: {self.calcLoss(X, Y)}")
        self.layers[0].inputs = netInput