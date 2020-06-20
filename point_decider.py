import numpy as np
np.random.seed(133)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class PointDecider(object):
    """
        A tiny neural network for advanced applications.
    """

    def __init__(self):
        self.bias = 0

    def train(self, train_x, train_y, learning_rate=None, iter_count=100, show_error=None):
        if learning_rate is None:
            learning_rate = 1

        self.weights = np.random.randn(train_x.shape[0], 1)

        for _ in range(iter_count):
            node_output = self.predict(train_x)
            assert node_output.shape == (1, train_x.shape[1])

            raw_diff = node_output - train_y
            error = 0.5 * (raw_diff ** 2)

            if show_error and _ % 10 == 0:
                print("Error", np.sum(error))

            derivative = node_output * (1 - node_output)
            d_o = derivative * raw_diff

            self.weights += learning_rate * -np.sum(d_o * train_x)
            self.bias += learning_rate * -np.sum(d_o)

    def predict(self, inputs):
        return sigmoid((inputs * self.weights) + self.bias)


def main():
    bias = 0
    train_x = np.array([[-10, -8, 5, 90]])
    train_y = np.array([[0, 0, 1, 1]])
    decider = PointDecider()
    decider.train(train_x, train_y, show_error=True)
    iter_count = 100
    inputs = np.array([[14, 16, -500, 10233]])
    raw_out = decider.predict(inputs)
    out = np.round(raw_out)
    print(out)


if __name__ == "__main__":
    main()
