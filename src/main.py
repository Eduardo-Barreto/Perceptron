import numpy as np


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, X):
        # Hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Output layer
        self.final_input = (
            np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        )
        self.final_output = self.sigmoid(self.final_input)

        return self.final_output

    def compute_cost(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward_pass(self, X, y_true, y_pred):
        output_error = y_true - y_pred
        output_delta = output_error * self.sigmoid_derivative(y_pred)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += (
            self.hidden_output.T.dot(output_delta) * self.learning_rate
        )
        self.bias_output += (
            np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        )

        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += (
            np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
        )

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            y_pred = self.forward_pass(X)
            self.backward_pass(X, y, y_pred)
            cost = self.compute_cost(y, y_pred)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Cost: {cost}")


if __name__ == "__main__":
    mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)

    # XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    mlp.train(X, y, epochs=10000)

    print("Expected: ")
    print(y)
    print("Predicted: ")
    print(mlp.forward_pass(X))
