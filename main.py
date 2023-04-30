import numpy as np
from queue import PriorityQueue
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derive_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def graph(layers):
    n = len(layers)*len(layers[0])
    graph = [[] for i in range(n)]
    for i in range(len(layers)-1):
        for u in layers[i]:
            for v in layers[i+1]:
                graph[u].append(v)
    return graph

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()    
 
class Neuron:
    def __init__(self, weights, bias):
        #weights is an np array, bias is an integer
        self.weights = weights
        self.bias = bias
    def feedForward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    def feedBackward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return derive_sigmoid(total)
    def update(self, weights, bias):
        self.weights = self.weights - (weights*0.1)
        self.bias = self.bias - (bias*0.1)
    def computeGradients(self, inputs, target):
        prediction = self.feedForward(inputs)
        derror_dprediction = 2 * (prediction - target)
        dprediction_dtotal = self.feedBackward(inputs)
        dtotal_dbias = 1
        dtotal_dweights = (0 * self.weights) + (1 * inputs)
        
        derror_dbias = (derror_dprediction * dprediction_dtotal * dtotal_dbias)
        derror_dweights = (derror_dprediction * dprediction_dtotal * dtotal_dweights)
        
        self.update(derror_dweights, derror_dbias)

class NeuralNetwork:
    def __init__(self, hiddenLayers, numNeurons):
        """_this neural network has_
            inputs (_array_)
            n hiddenLayers, each with numNeurons 
            each neuron (_Neuron_) is stored at layers[i][j]
            1 output layer with 1 neuron (output)
        """
        self.inputs = []
        self.hiddenLayers = hiddenLayers
        self.numNeurons = numNeurons
        self.layers = []
        self.layers_int = []
        self.layers_input = []
        for i in range(hiddenLayers):
            layer = []
            layer_int = []
            for j in range(numNeurons):
                if i == 0:
                    layer.append(Neuron(np.array([np.random.normal() for i in range(2)]), np.random.normal()))
                else:    
                    layer.append(Neuron(np.array([np.random.normal() for i in range(self.numNeurons)]), np.random.normal()))
                layer_int.append(i*numNeurons + j)
            self.layers.append(layer)
            self.layers_int.append(layer_int)
        self.graph = graph(self.layers_int)
        self.output = Neuron(np.random.rand(1, self.numNeurons), np.random.normal())
        self.outputlayer = []
    def feedForward(self, inputs):
        #BFS here
        self.inputs = inputs
        q = PriorityQueue() #priority_queue
        visited = [0 for i in range(self.hiddenLayers*self.numNeurons)]

        x = self.inputs #x has to be a np.array(list)
        outputs = []
        for i in range(len(self.layers[0])):
            visited[i] = 1
            q.put(i)
        while not q.empty():
            u = q.get()
            layer = u // self.numNeurons
            node = u - layer*self.numNeurons
         #   print(layer, node, x)
            output = self.layers[layer][node].feedForward(x)
            outputs.append(output)
         #   print("getting from q:", u, x, output, layer, node)
            self.layers_input.append(x)
            if len(outputs) == self.numNeurons:
                x = np.array(outputs)
                outputs = []
            for v in self.graph[u]:
                if visited[v] <= self.numNeurons:
                    visited[v] += 1
                    if visited[v] == 1:
                        q.put(v)
        self.outputlayer = self.output.feedForward(x)
        self.layers_input.append(x)
        # for i in range(len(self.layers_input)):
        #     print(i, self.layers_input[i])
        return self.outputlayer
    def train(self, data, y_trues):
        epochs = 10000
        for epoch in range(epochs):
            for x, y_true in zip(data, y_trues):
                self.feedForward(x)
                self.output.computeGradients(self.layers_input[-1], y_true)
                # for j in range(self.numNeurons):
                #     self.layers[-1][j].computeGradients(self.layers_input[i*self.numNeurons + j], y_true)
                for i in range(len(self.layers)-1, -1, -1):
                    for j in range(self.numNeurons):
                        self.layers[i][j].computeGradients(self.layers_input[i*self.numNeurons + j], y_true)
            if epoch % 1000 == 0:
                y_preds = np.apply_along_axis(self.feedForward, 1, data)
           #     print(type(y_preds), type(y_trues))
                loss = mse_loss(y_trues, y_preds)
                print("Epoch {0} loss: {1:0.3f}".format(epoch, loss))


# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
  [10, -1],  # Eve
  [30, 8],   # Frank
  [-10, -5], # Grace
  [20, 5],   # Henry
  [15, 3],   # Isabel
  [-12, -4], # Jack
  [-3, -3],  # Kate
  [8, 1],    # Luke
  [-5, -2],  # Mary
  [-8, -3],  # Nick
  [12, 3],   # Olivia
  [35, 10],  # Paul
  [3, 2],    # Quincy
  [0, -2],   # Rachel
  [7, 1],    # Samantha
  [-7, -2],  # Tom
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
  1, # Eve
  0, # Frank
  1, # Grace
  0, # Henry
  1, # Isabel
  0, # Jack
  1, # Kate
  0, # Luke
  1, # Mary
  0, # Nick
  1, # Olivia
  0, # Paul
  0, # Quincy
  1, # Rachel
  1, # Samantha
  0, # Tom
])

# Train our neural network!
network = NeuralNetwork(2, 3)
network.train(data, all_y_trues)
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 4])  # 155 pounds, 70 inches

print("Emily: %.3f" % network.feedForward(emily)) 
print("Frank: %.3f" % network.feedForward(frank)) 
                
                        
        
        
        
                            
                
                    
                
        
        
        
        
        
        
