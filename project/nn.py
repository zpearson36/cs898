import csv
import numpy as np
import random
import time

random.seed(time.time())

class ActivationFunction:
    function_list = ["sigmoid", "softmax", "relu", "tanh"]
    
    def __init__(self, function):
        assert function in self.function_list

        self.function = function

    def value(self, val):
        retval = []
        if self.function == "relu":
            for elmt in val:
                retval.append(max(0, elmt))
        if self.function == "sigmoid":
            for elmt in val:
                retval.append(1/(1 + np.exp(-elmt)))
        if self.function == "softmax":
            shift = val - np.max(val)
            exp = np.exp(val)
            retval = exp / np.sum(exp)
        if self.function == "tanh":
            for elmt in val:
                retval.append((np.exp(elmt) - np.exp(-elmt)) / (np.exp(elmt) + np.exp(-elmt)))
                

        return retval

    def derivative(self, val):
        retval = []
        if self.function == "relu":
            for elmt in val:
                if elmt == 0:
                    retval.append(0)
                else:
                    retval.append(1)
        if self.function == "sigmoid":
            for elmt in val:
                retval.append(self.value([elmt])[0] * (1 - self.value([elmt])[0]))
        if self.function == "softmax":
            for i, s_i in enumerate(val):
                retval.append([])
                for j, s_j in enumerate(val):
                    tmp = 0
                    if i == j:
                        tmp = 1
                    retval[i].append(s_i * (tmp - s_j))
        if self.function == "tanh":
            for elmt in val:
                retval.append(1 - self.value([elmt])[0]**2)
            

        return retval

class LossFunction:
    function_list = ["meanSquareError", "categoricalCrossEntropy"]

    def __init__(self, function):
        assert function in self.function_list
        
        self.function = function

    def value(self, measured, expected):
        loss = None
        if self.function == "meanSquareError":
            assert expected.shape == measured.shape, f"{expected.shape} | {measured.shape}"
            loss = np.square(measured - expected).mean()
        if self.function == "categoricalCrossEntropy":
            assert expected.shape == measured.shape, f"{expected.shape} | {measured.shape}"
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

    def __init__(self, node_count, weight_count, a_func,x):
        self.name = f"{x}"
        self.nodes = np.random.uniform(size=(node_count,weight_count))#np.array([[random.randrange(-1,1) for _ in range(weight_count)] for _ in range(node_count)])
        self.activation = a_func
        self.biases = []
        for _ in range(node_count):
            self.biases.append(.5)
        self.outputs_before_activation = []
        self.activated_derivative = []
        self.activated = []
        self.weight_contribs = []

    def forward(self, vals):
        assert vals.shape[0] == self.nodes.shape[1], "Shape Mismatch"
        self.outputs_before_activation.append([])
        for i, node in enumerate(self.nodes):
            self.weight_contribs.append([])
            for weight, val in zip(node, vals):
                t = np.multiply(weight, val)
                self.weight_contribs[i].append(t)

            self.outputs_before_activation[-1].append(sum(self.weight_contribs[i]) + self.biases[i])

        output = self.activation.value(self.outputs_before_activation[-1])
        self.activated_derivative.append(self.activation.derivative(self.outputs_before_activation[-1]))
        self.activated.append(output)
        #print()
        #print("weight_contribs:", self.weight_contribs)
        #print("unactivated:    ", len(self.outputs_before_activation))
        #print("unactivated2:   ", self.outputs_before_activation[-1])
        #print()
        return np.array(output)
    
    def update_weights(self, amounts):
        assert amounts.shape[0] == self.nodes.shape[1], "shape mismatch"
        assert amounts.shape[1] == self.nodes.shape[0], "shape mismatch"

        self.nodes = np.subtract(self.nodes, amounts)

    def reset_outputs(self):
        self.outputs_before_activation = []
        self.tmp = []
        self.weight_contribs = []
        self.activated = []
        self.activated_derivative = []


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
        x = 1
        for layer in layers:
            if len(self.layers) == 0: weight_count = params
            else: weight_count = self.layers[-1].nodes.shape[0]
            self.layers.append(Layer(layer, weight_count, a_func, x))
            x += 1
        self.activation = a_func
        self.learning_rate = l_rate
        self.loss = l_func

    def forward(self, input_data):
        output = []

        for layer in self.layers:
            if len(output) == 0:
                 output.append(layer.forward(input_data))
            else:
                 output.append(layer.forward(output[-1]))

        return output

    def train(self, input_data, classification):
        gradients = []
        for data, actual in zip(input_data, classification):
            output = self.forward(data)

            d_error = np.array(self.loss.derivative(output[-1], actual))
            layer_index = len(self.layers) - 1
            for layer in self.layers[::-1]:
                d_active = np.array(layer.activated_derivative[-1])
                if layer_index < len(self.layers) - 1:
                    d_error = np.dot(delta, self.layers[layer_index + 1].nodes)
                    delta = d_error * d_active
                else:
                    delta = np.array([d_error * d_active])
                activated = np.transpose(self.layers[layer_index - 1].activated)
                if layer_index == 0: activated = np.reshape(np.array(data), (len(data),1))
                grad_matrix = np.dot(activated, delta)
                gradients.insert(0, grad_matrix)
                layer.reset_outputs()
                layer_index -= 1

            # SGD
            for gradient, layer in zip(gradients, self.layers):
                for i, row in enumerate(gradient):
                    for j, col in enumerate(layer.nodes):
                        col[i] -= row[j] * self.learning_rate

    def clean_layers(self):
        for layer in self.layers:
            layer.reset_outputs()


if __name__ == "__main__":
    def f(n):
        return n**2
    a_func = ActivationFunction("tanh")
    l_func = LossFunction("meanSquareError")
    #l_func = LossFunction("categoricalCrossEntropy")
    #d = Data("../.ignore/mnist_train.csv", "csv")
    #d.show()
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
    _range = range(100)
    d = [[random.choice(_range)] for _ in range(50)]
    c = [[f(n[0])] for n  in d]

    xor_data = [[0,0],[0,1],[1,0],[1,1]]
    xor_class = [[0],[1],[1],[0]]
    d = xor_data
    c = xor_class


    b = Brain(2, [2, 1], a_func, 1e-1, l_func)
    for n in d:
        print(f"f({n}) =",b.forward(np.array(n))[-1])
        b.clean_layers()
    print()
    for _ in range(5000):
        b.train(np.array(d), np.array(c))
    for n in d:
        print(f"f({n}) =",b.forward(np.array(n))[-1])
        b.clean_layers()
    #b.train(np.array([[1,1]]),np.array([[1]]))
