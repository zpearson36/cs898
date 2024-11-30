import csv
import idx2numpy as id2np
import numpy as np
import random
import time

random.seed(time.time())
np.random.seed(np.int64(time.time()))

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
            retval = np.array(self.value(val)) * (np.eye(np.array(val).shape[0]) - np.array(self.value(val)).T)
        if self.function == "tanh":
            for elmt in val:
                retval.append(1 - self.value([elmt])[0]**2)
            

        return retval

class LossFunction:
    function_list = ["meanSquaredError", "categoricalCrossEntropy"]

    def __init__(self, function):
        assert function in self.function_list
        
        self.function = function

    def value(self, measured, expected):
        assert expected.shape == measured.shape, f"{expected.shape} | {measured.shape}"
        loss = None
        if self.function == "meanSquaredError":
            loss = np.square(measured - expected).mean()
        if self.function == "categoricalCrossEntropy":
            loss = -sum(expected * np.log(measured + 10**-100))

        return loss

    def derivative(self, measured, expected):
        val = []
        if self.function == "meanSquaredError":
            for real, guess in zip(expected, measured):
                val.append(2*(guess - real))

        if self.function == "categoricalCrossEntropy":
            val = -expected/(measured + 10**-100)
        return val

class MNIST:
    def  __init__(self):
        self.train_images = id2np.convert_from_file("data/train-images.idx3-ubyte")
        train_labels = id2np.convert_from_file("data/train-labels.idx1-ubyte")
        self.train_labels = []
        for label in train_labels:
            val =  [0 for _ in range(10)]
            val[label] = 1
            self.train_labels.append(val)

        self.test_images =  id2np.convert_from_file("data/t10k-images.idx3-ubyte")
        test_labels =  id2np.convert_from_file("data/t10k-labels.idx1-ubyte")
        self.test_labels = []
        for label in test_labels:
            val =  [0 for _ in range(10)]
            val[label] = 1
            self.test_labels.append(val)

    def train(self, split=1, flatten=True):
        train_images = self.train_images
        if flatten:
            train_images = train_images.reshape((train_images.shape[0], 784,))
            
        return train_images / 255, self.train_labels

    def test(self, flatten=True):
        test_images = self.test_images
        if flatten:
            test_images = test_images.reshape((test_images.shape[0], 784,))
        return test_images / 255, self.test_labels

class Layer:

    def __init__(self, node_count, weight_count, a_func, initialize="r_uniform", regularizer=1):
        if initialize == "r_uniform": self.nodes = np.random.uniform(size=(node_count,weight_count)) / regularizer
        if initialize == "r_normal": self.nodes = np.random.normal(size=(node_count,weight_count)) / regularizer
        if initialize == "ones": self.nodes = np.ones((node_count,weight_count))
        if initialize == "zeros": self.nodes = np.zeros((node_count,weight_count))
        self.activation = a_func
        self.biases = []
        for _ in range(node_count):
            self.biases.append(.5)
        self.outputs_before_activation = []
        self.activated_derivative = []
        self.activated = np.zeros((self.nodes.shape[0],1))
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

        output   = self.activation.value(self.outputs_before_activation[-1])
        d_output = self.activation.derivative(self.outputs_before_activation[-1])
        self.activated_derivative.append(d_output)
        output = np.array(output)
        output = np.reshape(output, (output.shape[0], 1))
        #print(output.shape)
       # print(self.activated.shape)
        self.activated += output
        return np.reshape(output, (output.shape[0],))
    
    def update_weights(self, amounts):
        assert amounts.shape[0] == self.nodes.shape[1], "shape mismatch"
        assert amounts.shape[1] == self.nodes.shape[0], "shape mismatch"

        self.nodes = np.subtract(self.nodes, amounts)

    def reset_outputs(self):
        self.outputs_before_activation = []
        self.weight_contribs = []
        self.activated = np.zeros((self.nodes.shape[0],1))
        self.activated_derivative = []

    def print_outputs(self):
        print(len(self.outputs_before_activation))
        print(len(self.weight_contribs))
        print(len(self.activated))
        print(len(self.activated_derivative))


class Brain:
    
    def __init__(self, layers=[], loss=LossFunction("meanSquaredError"), l_rate=.001):
        '''
        @param params: number of input parameters
        @param layers: list of ints indicating how many nodes per layer.
                       length of list is how many layers
        @param a_func: activation function
        @param l_rate: learning rate
        @param l_func: loss function
        '''
        self.layers = layers
        self.learning_rate = l_rate
        self.loss = loss

    def forward(self, input_data):
        output = []

        for layer in self.layers:
            if len(output) == 0:
                 output.append(layer.forward(input_data))
            else:
                 output.append(layer.forward(output[-1]))

        return output[-1]

    def train(self, input_data, classification, batch_size=1, verbose=True):
#        batch_size  =  1
        batch_count = 0
        d_error = np.zeros(classification.shape[1:])
        avg_error = np.zeros(classification.shape[1:])
        start  = time.time()
        for data, actual in zip(input_data, classification):
            batch_count  += 1
            if verbose:
                percent = batch_count/len(input_data)  * 100
                progress = "Training Progress: <"
                progress += "|" * int(percent)
                progress += "-" *  (100-int(percent))
                progress  += ">"
                print(progress, "{:.1f}% - error: {:.3f}".format(percent, sum(list(avg_error/input_data.shape[0])) / 10),end="\r")
            output = self.forward(data)
            #print(data,output,actual)
            #print(output,actual)
            #print(output)
            fff = np.array(self.loss.derivative(output, actual))
            avg_error += np.array(self.loss.derivative(output, actual))
            #print(np.array(output).shape, fff.shape, d_error.shape)
            d_error += fff 
            if batch_count % batch_size == 0 or batch_count == classification.shape[0]:
                gradients = []
                d_error /= batch_size
                layer_index = len(self.layers) - 1
                for layer in self.layers[::-1]:
                    #layer.print_outputs()
                    d_active = np.array(layer.activated_derivative[-1])
                    #print("d_active:",d_active.shape)
                    if layer_index < len(self.layers) - 1:
                        d_error = np.dot(delta, self.layers[layer_index + 1].nodes)
                        delta = d_error * d_active
                    else:
                        if self.loss.function == "categoricalCrossEntropy":
                            delta = output - actual
                            delta = np.reshape(delta, (delta.shape[0], 1)).T
                        else:
                            delta = np.array([d_error * d_active])
                    activated = self.layers[layer_index - 1].activated#np.transpose(self.layers[layer_index - 1].activated)
                    if layer_index == 0: activated = np.reshape(np.array(data), (len(data),1))
             #       print("delta:",delta.shape)
              #      print("activated:", activated.shape)

                    grad_matrix = activated @ delta
                    gradients.insert(0, grad_matrix)
                    layer.reset_outputs()
                    layer_index -= 1
                    d_error = np.zeros(classification.shape[1:])

                # SGD
                for gradient, layer in zip(gradients, self.layers):
                    for i, row in enumerate(gradient):
                        for j, col in enumerate(layer.nodes):
                            col[i] -= row[j] * self.learning_rate
        if verbose: print("\nepoch training time:",time.time() - start," - error:",avg_error / input_data.shape[0])

    def clean_layers(self):
        for layer in self.layers:
            layer.reset_outputs()

def test_xor():
    xor_data = [[0,0],[0,1],[1,0],[1,1]]
    #xor_class = [[0,1],[1,0],[1,0],[0,1]]
    xor_class = [[0],[1],[1],[0]]
    d = xor_data
    c = xor_class
    layers = [
            Layer(5, 2, ActivationFunction("tanh")),
            Layer(5, 5, ActivationFunction("tanh")),
            Layer(5, 5, ActivationFunction("tanh")),
            Layer(1, 5, ActivationFunction("sigmoid"))
            ]
    b = Brain(layers, LossFunction("categoricalCrossEntropy"), 1e-1)
    for n in d:
        print(f"f({n}) =",b.forward(np.array(n)))
        b.clean_layers()
    for _ in range(5000):
        b.train(np.array(d), np.array(c), batch_size=100, verbose=False)
    for n in d:
        print(f"f({n}) =",b.forward(np.array(n)))
        b.clean_layers()
    exit()


if __name__ == "__main__":
    # sanity check
    test_xor()
    l_func = LossFunction("categoricalCrossEntropy")

    mnist = MNIST()
    train_image, train_label = mnist.train()
    test_images, test_labels = mnist.test()


    layers = [
            Layer(100, 784, ActivationFunction("relu"), initialize='r_normal', regularizer=10),
            Layer(50, 100,  ActivationFunction("relu"), initialize='r_normal',  regularizer=10),
           # Layer(64, 128,   ActivationFunction("sigmoid")),
            Layer(10, 50, ActivationFunction("softmax"), initialize='r_normal',  regularizer=10)
            ]


    b = Brain(layers, l_func, l_rate=.001)
    for x,y in zip(test_images[:10], test_labels[:10]):
        out  =  list(b.forward(x))
        guess = out.index(max(out))
        label = y.index(1)
        print("guess:",guess,"actual:",label)
        b.clean_layers()
    start = time.time()
    for _ in range(10):
        #print("epoch",_+1)
        b.train(train_image[:1000], np.array(train_label[:1000]))
    print("total training time:", time.time() - start)
    for x,y in zip(test_images[:10], test_labels[:10]):
        out  =  list(b.forward(x))
        guess = out.index(max(out))
        label = y.index(1)
        print("guess:",guess,"actual:",label)
        b.clean_layers()
