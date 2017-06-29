# Glossary

## Neural Networks

#### Fully connected layers

Fully connected layers are usually the last layers in a neural network. ["In a fully connected layer, each neuron is connected to every neuron in the previous layer, and each connection has its own weight."](https://www.reddit.com/r/MachineLearning/comments/3yy7ko/what_is_the_difference_between_a_fullyconnected/cyhsdqd/) ["This layer basically takes an input volume and outputs an N dimensional vector where N is the number of classes that the program has to choose from."](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/).

These layers carry out the ["high-level reasoning in the neural network"](https://en.wikipedia.org/wiki/Convolutional_neural_network#Fully_connected_layer). The previous layer, which the fully connected layer is connected to, would represent [high level features](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/) that the fully connected layer is reducing to higher level features or the end classifications themselves.

From [Leonardo Araujo dos Santos': Artificial Intelligence](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/fc_layer.html): On forward propagation, the layer has 3 inputs (input signal, weights, bias) and 1 output (the classification or higher level feature). On back propagation, the layer has 1 input (the gradient), which has the same size as the initial output, and produces three outputs (derivatives of the input signal, weights, and bias), which are the same size as the original inputs.

#### SGD

Stochastic Gradient Descent