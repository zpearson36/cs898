import csv
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
                retval.append(1/(1 + np.exp(-elmt)))
        if self.function == "softmax":
            shift = val - np.max(val)
            exp = np.exp(val)
            retval = exp / np.sum(exp)
                

        return retval

    def derivative(self, val):
        retval = []
        if self.function == "sigmoid":
            for elmt in val:
                retval.append(self.value(elmt) * (1 - self.value(elmt)))
        if self.function == "softmax":
            for i, s_i in enumerate(val):
                retval.append([])
                for j, s_j in enumerate(val):
                    tmp = 0
                    if i == j:
                        tmp = 1
                    retval[i].append(s_i * (tmp - s_j))
            

        return retval

class LossFunction:
    function_list = ["meanSquareError", "categoricalCrossEntropy"]

    def __init__(self, function):
        assert function in self.function_list
        
        self.function = function

    def value(self, measured, expected):
        loss = None
        if self.function == "meanSquareError":
            assert expected.shape == mesured.shape
            loss = np.square(measured - expected).mean()
        if self.function == "categoricalCrossEntropy":
            assert expected.shape == measured.shape
            loss = []
            # can be optimized: loss[i] == -log(actual[i])
            # where expected[i] is the correct class
            for real, guess in zip(expected, measured):
                loss.append(-real*np.log(guess))

        return loss

    def derivative(self, measured, expected):
        val = []
        if self.function == "meanSquareError":
            for real, guess in zip(expected, measured):
                val.append(2*(guess - real))

        if self.function == "categoricalCrossEntropy":
            exp = np.exp(measured)
            exp_sum = np.sum(exp)
            for real, guess in zip(exp, expected):
                if real == 1:
                    val.append((exp / exp_sum) -1)
                else:
                    val.append(exp / exp_sum)

        return val



class Data:
    data_formats = ["csv"]

    def __init__(self, data_dir, data_format):
        assert data_format in self.data_formats
        self.data_dir = data_dir
        self.format = data_format

    def show(self):
        d = np.genfromtxt(self.data_dir, delimiter=",")
        print(d)
        print(type(d))



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
    
    def __init__(self, params, layers, a_func, l_rate, l_func):
        '''
        @param params: number of input parameters
        @param layers: list of ints indicating how many nodes per layer.
                       length of list is how many layers
        @param a_func: activation function
        @param l_rate: learning rate
        @param l_func: loss function
        '''
        self.layers = []
        for layer in layers:
            if len(self.layers) == 0: weight_count = params
            else: weight_count = self.layers[-1].nodes.shape[0]
            self.layers.append(Layer(layer, weight_count, a_func))
        self.activation = a_func
        self.learning_rate = l_rate
        self.loss = l_func

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
    a_func = ActivationFunction("sigmoid")
    l_func = LossFunction("categoricalCrossEntropy")
    d = Data("../.ignore/mnist_train.csv", "csv")
    d.show()
    #l = Layer(5, 3, a_func)
    #l2 = Layer(5, 5, a_func)
    #l3 = Layer(1, 5, a_func)

    #print(l.nodes)
    #output1 = l.forward(np.array([1,1,1]))
    #output2 = l2.forward(output1)
    #output3 = l3.forward(output2)

    #print(output1)
    #print(output2.shape)
    #print(output3)

    #b = Brain(3, [5, 5, 1], a_func, .01, l_func)
    #print(b.guess(np.array([1,1,1])))
