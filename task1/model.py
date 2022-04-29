import numpy as np


class Softmax:
    def __init__(self, input_size, category_size, vocab_size):
        self.input_size = input_size
        self.category_size = category_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(self.vocab_size, self.category_size) # initialize the weight matrix

    def regression(
        self,
        x,
        y,
        gradient,
        epochs,
        lr,
        mini_size=10
    ):
        assert self.input_size == len(x) and self.input_size == len(y), "Data size doesn't match!"
        if gradient == "SGD":
            times = epochs # keep three gradient strategies having fair times of gradient descent, more details in report.md
            for _ in range(times):
                i = np.random.randint(0, self.input_size - 1) # randomly choose a sample
                y_hat = self._softmax_vector(self.W.T.dot(x[i].reshape(-1, 1)))
                increment = x[i].reshape(-1, 1).dot((self._one_hot(y[i]) - y_hat).T)
                self.W += lr * increment
        elif gradient == "BGD":
            times = int(epochs / self.input_size)
            for _ in range(times):
                increment = np.zeros((self.vocab_size, self.category_size))
                for i in range(self.input_size):
                    y_hat = self._softmax_vector(self.W.T.dot(x[i].reshape(-1, 1)))
                    increment += x[i].reshape(-1, 1).dot((self._one_hot(y[i]) - y_hat).T)
                self.W += lr / self.input_size * increment
        elif gradient == "mini-batch":
            times = int(epochs / mini_size)
            for _ in range(times):
                increment = np.zeros((self.vocab_size, self.category_size))
                for _ in range(mini_size):
                    i = np.random.randint(0, self.input_size - 1)
                    y_hat = self._softmax_vector(self.W.T.dot(x[i].reshape(-1, 1)))
                    increment += x[i].reshape(-1, 1).dot((self._one_hot(y[i]) - y_hat).T)
                self.W += lr / mini_size * increment
        else:
            raise Exception("Unknown gradient strategy!")

    def _softmax_vector(self, x):
        """Apply softmax on a vector."""
        x = np.exp(x - np.max(x)) # substract maximum value to prevent overflow
        return x / x.sum()

    def _softmax_matrix(self, x):
        """Apply softmax on a matrix."""
        x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x / np.sum(x, axis=1, keepdims=True)

    def _one_hot(self, x):
        """Transform an 'int' into a one-hot vector."""
        vec = np.array([0] * self.category_size)
        vec[x] = 1
        return vec.reshape(-1, 1)

    def predict(self, x):
        prob = self._softmax_matrix(x.dot(self.W))
        return prob