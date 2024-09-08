import numpy as np
import random

class Layer:

    def __init__(self, node_count, weight_count, a_func):
        self.nodes = np.array([[1] * weight_count] * node_count) #np.array([[random.random() for _ in range(weight_count)] for _ in range(node_count)])
        self.activation = a_func
        self.biases = [.5] * node_count

    def forward(self, vals):
        assert vals.shape[0] == self.nodes.shape[1],"Shape Mismatch"
        mat = np.matmul(self.nodes, vals)
        after_bias = []
        for elmt, bias in zip(mat, self.biases):
            after_bias.append(elmt + bias)
        output = []
        for elmt in after_bias:
            output.append(self.activation(elmt))

        return np.array(output)



if __name__ == "__main__":
    def f(n):
        return 2*n
    l = Layer(5, 3, f)

    print(l.nodes)
    print(l.forward(np.array([1,1,1])))
