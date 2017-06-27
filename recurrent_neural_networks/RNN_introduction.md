# Recurrent Neural Networks
## Introduction

### Sequences of Vectors
"The core reason that recurrent nets are more exciting is that they allow us to operate over sequences of vectors: Sequences in the input, the output, or in the most general case both." Andrej Karpathy [link](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

![Sequence Types](/images/RNNs-sequences-types.jpeg)
Figure. Each rectangle is a vector and arrows represent functions (e.g. matrix multiply). Input vectors are in red, output vectors are in blue and green vectors hold the RNN's state

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
