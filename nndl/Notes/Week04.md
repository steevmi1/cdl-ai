# Deep Neural Networks
Understand the key computations underlying deep learning, use them to build and train deep neural networks, and apply it to computer vision.
## Learning Objectives
- See deep neural networks as successive blocks put one after each other
- Build and train a deep L-layer Neural Network
- Analyze matrix and vector dimensions to check neural network implementations.
- Understand how to use a cache to pass information from forward propagation to back propagation.
- Understand the role of hyperparameters in deep learning
## Deep L-Layer neural network
Seen everything we need, just need to put it together to create a deep NN.

Depth -- number of hidden layers and the output layer.

Hard to know how many layers you need, so often need to try multiple models.

L -- the number of layers.
$n^{[l]}$ the number of nodes in a layer l.
$a^{[l]}$ the number of activations in layer l.
$W^{[l]}$ the weights in layer l.
$b^{[l]}$ the bias vector in layer l.
## Forward Propagation in a Deep Neural Network
x: $z^{[1]}$ = $W^{[1]}$ x + $b^{[1]}$
   $a^{[1]}$ = $g^{[1]}(z^{[1]})$
   $z^{[2]}$ = $W^{[2]}$ $a^{[1]}$ + $b^{[2]}$
   $a^{[2]}$ = $g^{[2]}(z^{[2]})$
   and so on for the remaining layers.

In general, $z^{[l]}$ = $W^{[l]}$ $a^{[l - 1]}$ + $b^{[l]}$, and $a^{[l]}$ = $g^{[l]}(z^{[l]})$. You still end up having a for-loop, to iterate through the number of layers.
## Getting your Matrix Dimensions Right
Looking at a NN model where

$n^{[0]}$ = 2
$n^{[1]}$ = 3
$n^{[2]}$ = 5
$n^{[3]}$ = 4
$n^{[4]}$ = 2
$n^{[5]}$ = 1

Based on this, $z^{[1]}$ will be an ($n^{[1]}$, 1) matrix, and x is a ($n^{[0]}$, 1) matrix, so this means that by the rules of matrix multiplication W needs to be a matrix of shape ($n^{[1]}$, $n^{[0]}$).

## Why Deep Representations
What is this actually doing? Look at NN that's trying to detect faces. First pass if you plot it looks like it's trying to find different edges. Second pass is putting them together to start building them up to find more complex things (like an eye or a nose), then start to put features together to form faces.

Circuit theory and deep learning -- there are certain functions that you can compute more efficiently by a "small" deep neural network that are exponentially harder to compute with shallow networks.

Still often start with a logistic regression, then a small, shallow network, then move on from there.
## Building Blocks of Deep Neural Networks
Will be useful to cache/store $z^{[l]}$ for use in figuring out backprop. Each step in the forward prop will take in $a^{[l-1]}$, and will provide $W^{[l]}$ and $b^{[l]}$, and send out $a^{[l]}$. We'll cache $z^{[l]}$, as this will make calculating the backprop easier.

For each backprop step, you input $da^{[l]}$, and output $da^{[l-1]}$, and also can calculate $dW^{[l]}$ and $db^{[l]}$.
## Forward and Backward Propagation
Forward prop -- each step should look familiar:
$z^{[l]} = W^{[l]} \cdot a^{[l]} + b^{[l]}$
$a^{[l]} = g^{[l]}(z^{[l]})$

Backward prop should also look familiar:
$dz^{[l]} = da^{[l]} * g^{[l]'}(z^{[l]})$ (element-wise product)
$dW^{[l]} = dz^{[l]} \cdot a^{[l-1]}$
$db^{[l]} = dz^{[l]}$
$da^{[l-1]} = W^{[l]T} \cdot dz^{[l]}$

$da^{[l]} = -\frac{y}{a} + \frac{1 - y}{1 - a}$
## Parameters versus Hyperparameters
What are hyperparameters? Things like $\alpha$ (learning rate), # of iterations, # of hidden layers L, # of hidden units ($n^{[1]}, n^{[2]}$, ....), choice of activation function. As you move on in study, also have things like mini-batch size, momentum, regularization parameters.

Often will iterate through various options for hyperparameters, to explore how your neural network performs and compare against other sets of parameters.
## What Does that have to do with the Brain
Only a superficial link -- neuron gets signals from other neurons, processes, and sends output. But we don't really know what a neuron does, it looks to be a fairly complex system. 
