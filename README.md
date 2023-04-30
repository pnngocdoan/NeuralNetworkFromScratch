# Building Neural Network From Scratch

A neural network that is implemented from scratch by OOP, numpy, priority queue, and breadth-first search (BFS). This neural network is not intended for machine learning or research purposes. It is my attempt to understand the network architecture from an aspiring software engineer's perspective. Notably, I integrated BFS to let the output from each neuron traverse to the neurons in the next layer, which can be seen in the feedForward() function in the NeuralNetwork class.

## Description

This neural network has:
- inputs: a numpy array of size 2
- n hidden layers, each with m neurons (the user decides the n and m) 
- each neuron's weights and bias are random in the range of (0, 1)
- 1 output layer with 1 neuron (output)

## Model Testing
In this particular example, we use a dataset of weights and heights to predict whether a person is female or male. If the output is close to 1, the tested person is female and vice versa. Each weight and height will be reduced by 135 and 66 respectively to help the model predict better.

Dataset to train the model:
```
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
```
For example, Alice weighs -2+135 = 133 pounds and has a height of -1+66 = 65 inches. 

Dataset to test:
```
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 4])  # 155 pounds, 70 inches
```

The predictions:
```
Emily: 0.659
Frank: 0.480
```

Although Emily is predicted to be closer to female (0.659 is closer to 1) and Frank is predicted to be closer to male (0.480 is closer to 0), they are not good predictions as both are around 0.5 more. 

## Resources
- https://victorzhou.com/blog/intro-to-neural-networks/
- https://realpython.com/python-ai-neural-network/



