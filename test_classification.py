from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from NeuralNetwork import NeuralNetwork
import numpy as np

def main():
    # Generate synthetic data (binary classification)
    X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
    y = y.reshape(-1, 1)  # Reshape for OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)  # Convert labels to one-hot encoding

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    # Create and train the neural network
    nn = NeuralNetwork(layer_sizes=[4, 8, 2], learning_rate=0.1)
    nn.train(X_train, y_train, epochs=1000)

    # Evaluate on test data
    y_pred = nn.predict(X_test)
    y_true = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred == y_true)
    print(f"Test Accuracy: {accuracy:.2%}")
    

if __name__ == "__main__":
    main()