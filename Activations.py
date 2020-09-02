import math
import numpy as np

class Activations:
    @staticmethod
    def linear(x):
        if type(x) == np.ndarray:
            for i in range(len(x[0])):
                x[0][i] = x[0][i]
        return x

    @staticmethod
    def relu(x):
        if type(x) == np.ndarray:
            for i in range(len(x[0])):
                x[0][i] = max(0.0, x[0][i])
        else:
            x = max(0.0, x)
        return x

    @staticmethod
    def sigmoid(x):
        if type(x) == np.ndarray:
            for i in range(len(x[0])):
                x[0][i] = 1/(1 + math.e**-x[0][i])
        else:
            x = 1/(1 + math.e**-x)
        return x

    @staticmethod
    def getDerivative(activation):
        if activation == Activations.linear:
            return Activations.Derivatives.dLinear
        if activation == Activations.relu:
            return Activations.Derivatives.dRelu
        if activation == Activations.sigmoid:
            return Activations.Derivatives.dSigmoid
        return -1

    class Derivatives:
        @staticmethod
        def dLinear(x):
            if type(x) == np.ndarray:
                npArr = [[]]
                for i in range(len(x[0])):
                    npArr[0].append(1)
                npArr = np.array(npArr)
                return npArr
            return 1

        @staticmethod
        def dRelu(x):
            if type(x) == np.ndarray:
                npArr = [[]]
                for i in range(len(x[0])):
                    npArr[0].append(0.0 if x[0][i] < 0.0 else 1.0)
                npArr = np.array(npArr)
                return npArr
            return 0.0 if x < 0.0 else 1.0

        @staticmethod
        def dSigmoid(x):
            if type(x) == np.ndarray:
                npArr = [[]]
                for i in range(len(x[0])):
                    npArr[0].append(Activations.sigmoid(x[0][i])*(1-Activations.sigmoid(x[0][i])))
                npArr = np.array(npArr)
                return npArr
            return Activations.sigmoid(x)*(1-Activations.sigmoid(x))