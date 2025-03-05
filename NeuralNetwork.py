import numpy as np
from typing import List, Tuple
import Activation

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes
        
        self.weights: List[np.ndarray] = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01 for i in range(len(layer_sizes) - 1)]
        
        self.biases: List[np.ndarray] = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes) - 1)]
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        pre_activations = []
        
        A = X
        
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            Z = A @ W + b
            pre_activations.append(Z)
            A = Activation.relu(Z)
            activations.append(A)
        
        Z = A @ self.weights[-1] + self.biases[-1]
        pre_activations.append(Z)
        A = Activation.relu(Z)
        activations.append(A)
        
        return activations, pre_activations