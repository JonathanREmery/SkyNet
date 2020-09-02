import numpy as np
from Network import Network
from Layer import Layer
from Activations import Activations

def main():

    np.random.seed(10)

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])

    network = Network()
    network.add(Layer(16, Activations.relu), X[1])
    network.add(Layer(1, Activations.sigmoid))

    network.fit(X, Y, 0.01, verbose=True)

    print(f"\nf(0, 1) = {network.forward()[0][0]}")
    print(f"Error: {network.calcError(X, Y)}\nLoss: {network.calcLoss(X, Y)}", end="")

if __name__ == '__main__':
    main()