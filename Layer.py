import numpy as np
from Activations import Activations

class Layer:

    def __init__(self, nNeurons, activation=Activations.linear, input=np.array([0.0])):
        if type(input) == Layer:
            self.inputs = input.forward()
            self.inputLayer = input
        else:
            self.inputs = np.array([input])
            self.inputLayer = None
        self.weights = (np.random.random((nNeurons, len(self.inputs[0]))) * 2) - 1
        self.biases = (np.random.random((1,nNeurons)) * 2) - 1
        self.activation = activation
        self.output = np.nan
        self.target = None
        self.outputLayer = None

    def setInput(self, input):
        if type(input) == Layer:
            inputs = input.forward()
            self.inputLayer = input
            self.inputLayer.outputLayer = self
        else:
            inputs = np.array([input])
            self.inputLayer = None

        if len(inputs[0])-len(self.inputs[0]) != 0:
            self.weights = (np.random.random((len(self.biases[0]), len(inputs[0]))) * 2) - 1
        self.inputs = inputs
        return self.inputs

    def forward(self):
        if self.inputLayer != None:
            self.inputs = self.inputLayer.forward()
        self.output = self.activation(np.dot(self.weights, self.inputs.T).T + self.biases)
        return self.output

    def calcDeriv(self):
        deriv = []
        if self.outputLayer == None and type(self.target) == np.ndarray:
            deriv = self.output-self.target
        else:
            if self.outputLayer != None:
                outDeriv = self.outputLayer.calcDeriv()
                outputs = self.forward()
                for i in range(len(self.biases[0])):
                    deriv.append([])
                    for j in range(len(self.outputLayer.biases[0])):
                        wno = self.outputLayer.weights[j][i]
                        bo = self.outputLayer.biases[0][j]
                        deriv[len(deriv)-1].append(Activations.getDerivative(self.outputLayer.activation)(outputs[0][i]*wno+bo)*wno)
                deriv = np.array(deriv).dot(outDeriv.T).T
        return deriv