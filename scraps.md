# Scraps

*(A place for random notes until we find somewhere better to put them)*

## CS231n Notes

#### [Lecture 5](https://www.youtube.com/watch?v=mzkOF4tULj8&list=PL16j5WbGpaM0_Tj8CRmurZ8Kk1gEBc7fg&index=5])

Don't be afraid to work with smaller data sets. You don't have to train all neural networks from scratch. For example, you take a convolutional neural network and train it on image-net.org images, then strip off the "classifier" layer, so that the remaining network is basically a fixed feature extractor. You can then take the "feature extractor", put on a new classifier that works with your smaller data set, and start training on that, either just the topmost layers, or the entire network. Furthermore, you can take a pre-trained neural network shared by others and refine the training.

ReLUs are a default recommendation for activation functions, but are not without problems. If, on the forward pass, a ReLU is not activated, on the backward propagation, they won't get updated with any gradient – they become "dead" ReLUs. Initializing ReLUs with slightly positive numbers makes them less likely to die. Leaky ReLUs (max(0.1x, x)) prevent a ReLU from "dying", by  introducing a small slope on the negative side. PReLUs give a parameterized (potentially learnable) slope on the negative side (max(ax,x)).

Debugging visualization: Pick a layer to start at and show histograms of the activations. This can show when weights start to collapse around a mean with a very small deviation, leading to the problem of vanishing gradients.

Xavier initialization (for tanh) results in a better distribution, effectively giving everythin a standard deviation of 1. Using Xavier initialization with ReLUs doesn't work well. Proper initialization is an active area of research – it can have a huge effect on the effectiveness of a neural network.

Batch normalization makes sure that every feature across a batch has unit guassion activations. Usually inserted after fully connected layers and before a non-linearity.

Sanity checks:

1. Disable regularization
2. Make sure loss comes out correctly (before training)
3. Turn up regularization. Does loss go up as expected?
4. Train on a small piece with regularization off to effectively guarantee overfitting. Check that the loss gets close to zero.