import numpy as np
import random


class ActivationFunction:
    function_list = ["sigmoid", "softmax"]
    
    def __init__(self, function):
        assert function in self.function_list

        self.function = function

    def value(self, val):
        retval = []
        if self.function == "sigmoid":
            for elmt in val:
                retval.append(1/(1 + np.exp(-val)))
        if self.function == "softmax":
            shift = val - np.max(val)
            exp = np.exp(val)
            retval = exp / np.sum(exp)
                

        return retval

    def derivative(self, val):
        retval = None
        if self.function == "sigmoid":
            retval = self.value(val) * (1 - self.value(val))

        return retval

class Data:

    def __init__(self, data):
        self.data = data


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

        output = self.activation.value(after_bias)

        return np.array(output)
    
    def update_weights(self, amounts):
        assert amounts.shape == self.nodes.shape

        self.nodes = np.subtract(self.nodes, amounts)


class Brain:
    
    def __init__(self, params, layers, a_func, l_rate):
        '''
        @param params: number of input parameters
        @param layers: list of ints indicating how many nodes per layer.
                       length of list is how many layers
            @param a_func: activation function
        @param l_rate: learning rate
        '''
        self.layers = []
        for layer in layers:
            if len(self.layers) == 0: weight_count = params
            else: weight_count = self.layers[-1].nodes.shape[0]
            self.layers.append(Layer(layer, weight_count, a_func))
        self.activation = a_func
        self.learning_rate = l_rate

    def guess(self, input_data):
        output = []

        for layer in self.layers:
            if len(output) == 0:
                 output.append(layer.forward(input_data))
            else:
                 output.append(layer.forward(output[-1]))

        return output

  

if __name__ == "__main__":
    def f(n):
        return 2*n
    a_func = ActivationFunction("softmax")
    l = Layer(5, 3, a_func)
    l2 = Layer(5, 5, a_func)
    l3 = Layer(1, 5, a_func)

    print(l.nodes)
    output1 = l.forward(np.array([1,1,1]))
    output2 = l2.forward(output1)
    output3 = l3.forward(output2)

    print(output1)
    print(output2)
    print(output3)

    b = Brain(3, [5, 5, 1], a_func, 1)
    print(b.guess(np.array([1,1,1])))
