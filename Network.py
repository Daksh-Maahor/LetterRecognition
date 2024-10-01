import numpy as np

class Network:
    def __init__(self) -> None:
        np.random.seed(1)
        
        self.synaptic_weights_0 = 2 * np.random.random((30, 784)) - 1
        self.synaptic_biases_1 = 2 * np.random.random((30, 1)) - 1
        
        self.synaptic_weights_1 = 2 * np.random.random((30, 30)) - 1
        self.synaptic_biases_2 = 2 * np.random.random((30, 1)) - 1
        
        self.synaptic_weights_2 = 2 * np.random.random((26, 30)) - 1
        self.synaptic_biases_3 = 2 * np.random.random((26, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, y):
        return y * (1-y)
    
    def predict(self, inputs): # inputs is a 784 by 1 matrix # returns a 26 by 1 matrix
        layer_1 = self.sigmoid(np.matmul(self.synaptic_weights_0, inputs) + self.synaptic_biases_1)
        
        layer_2 = self.sigmoid(np.matmul(self.synaptic_weights_1, layer_1) + self.synaptic_biases_2)
        
        output_layer = self.sigmoid(np.matmul(self.synaptic_weights_2, layer_2) + self.synaptic_biases_3)
        
        return output_layer