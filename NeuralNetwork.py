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
    
    def backward(self, X: np.ndarray, Y: np.ndarray, activations: List[np.ndarray], pre_activations: List[np.ndarray]) -> None:
        batch_size = X.shape[0]
        L = len(self.weights)
        
        dZ = activations[-1] - Y
        for i in reversed(range(L)):
            dW = activations[i].T @ dZ / batch_size
            db = np.sum(dZ, axis=0, keepdims=True) / batch_size
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            if i > 0:
                dZ = (dZ @ self.weights[i].T * Activation.relu_derivative(pre_activations[i - 1]))
    
    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 1000) -> None:
        for epoch in range(epochs):
            activations, pre_activations = self.forward(X)
            self.backward(X, Y, activations, pre_activations)
            
            if epochs % 100 == 0:
                loss = -np.sum(Y * np.log(activations[-1] + 1e-8)) / X.shape[0]
                print(f"Epoch {epoch}, Loss: {loss: .4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis = 1)