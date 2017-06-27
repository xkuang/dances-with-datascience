# Recurrent Neural Networks
## Introduction

### Sequences of Vectors
"The core reason that recurrent nets are more exciting is that they allow us to operate over sequences of vectors: Sequences in the input, the output, or in the most general case both." Andrej Karpathy [link](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

"Recurrent neural networks or RNNs are a family of *neural networks for processing sequential data*. Much as a convolutional network is a neural network that is specialized for processing a grid of values X such as an image, a recurrent neural network is a neural network that is specialized for processing a sequence of values x(1), . . . , x(τ)." Ian Goodfellow [The Deep Learning Book](http://www.deeplearningbook.org/contents/rnn.html)

![Sequence Types](images/RNNs-sequences-types.jpeg)
Figure. Each rectangle is a vector and arrows represent functions (e.g. matrix multiply). Input vectors are in red, output vectors are in blue and green vectors hold the RNN's state

## Key concept of an RNN
![Fig10.3 Deep Learning Book](images/DeepLearningBook_fig103.png)
Two representations of the basic RNN architecture. On the left, the RNN is drawn with recurrent connections and on the right, it is drawn as a time-series unfolded computational graph. Each x-value represents one time point, such as for example, a word in a sentence or measurement at a time t. The hidden layer, h, is updated based on each the input of each point, thereby efectively giving the network the capacity to carry memory from inputs at any point in the sequence. 

The input x is used to produce an output o via the hidden layer, h. The output is compared to the actual values y and loss function L measures how far o and y deviate from each other. 

[The Deep Learning Book](http://www.deeplearningbook.org/contents/rnn.html)

## Example Applications of RNNs:

### One-to-One Sequences
Description: from fixed-sized input to fixed-sized output. The Vanilla mode of processing without RNN.

Use cases: image classification.

### One-to-Many Sequences
Description: sequence output.

Use cases: image captioning, where the input is an image and the output is a sentence of words that describes the image.

### Many-to-One Sequences
Description: sequence input.

Use cases: Sentiment analysis, where a given body of text is classified as expressing positive or negative sentiment.

### Many-to-Many Sequences
#### Asynchronous Sequences
Description: Sequence input and sequence output.

Use cases: Machine translation, where an RNN reads a sentence in English and then outputs a sentence in a different language.

#### Synchronous Sequences
Description: Synced sequence input and output.

Use cases: Video classification, where we wish to label each frame of the video.

- insert more material here

## Outline

- insert more material here

## Advanced Topics 
### Attention
Attention can be applied to various advanced topics of RNNs. For example, it can be used to read and write from a memory vector using Neural Turing Machines or be leveraged to get the network to focus on a particular section of the input. [Need general introduction to attention systems here]

Read more [here.](http://distill.pub/2016/augmented-rnns/)


## Resources 
[Aggregation](https://github.com/kjw0612/awesome-rnn) of RNN resources
[Four part RNN tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) using Python and Theano by Denny Britz
[Review Paper on RNNs](https://arxiv.org/abs/1506.00019) by Lipton, Berkowitz, and Elkan. 2015.
