# Shallow Neural Networks
*Learning Objectives*
- Understand hidden units and hidden layers
- Be able to apply a variety of activation functions in a neural network.
- Build your first forward and backward propagation with a hidden layer
- Apply random initialization to your neural network
- Become fluent with Deep Learning notations and Neural Network Representations
- Build and train a neural network with one hidden layer.
## Neural Networks Overview
Started last week with one layer of one node. Most neural networks have layers with multiple nodes, so you have to calculate Z and a for each layer, and also dw and db for each layer as well.
## Neural Network Representation
Simple network -- you have an input layer, one (or more) hidden layers, and then an output layer which gives you your $\hat{y}$.

Inputs -- were "X", but now is $a^{[0]}$, or your activation layer. The hidden layer calculates $a^{[1]}_{1}$, $a^{[1]}_{2}$, ... $a^{[1]}_{n}$. Each one of these will have a W and a b that go along with it.
## Computing a Neural Network's Output
For each node in your hidden layer, calculate
$z^{[l]}_{i} = W^{[l]T}_{i} x + b^{[l]}_{i}$
$a^{[l]}_{i} = \sigma(z^{[l]}_{i})$

We can vectorize this, each of the $W^{[i]T}$ turn from a column vector to a row vector, which can be stacked into a matrix.
## Vectorizing Across Multiple Examples
How do you vectorize this across multiple training examples? Naive way to work with m training examples is a for loop, and to go through each of your examples to calculate your zs and as.

Instead, you simply stack your input (column) vectors together, to form a matrix, and then calculate your matrix Zs and As.
## Explanation for Vectorized Implementation
Breakdown of matrix work in detail to show that vectorization method from the previous method is correct.
## Activation Functions
Don't have to use sigmoid function, can use other nonlinear functions such as tanh, which is really just a sigmoid function that's been shifted.

tanh works better for almost everything except the output layer, which still makes sense to use a sigmoid function (because you want your answer to be between 0 and 1).

The biggest drawback to sigmoid-style functions is that as z gets very large at either end of the scale, the slope is very small (almost 0), and that slows learning. Another popular choice is ReLU, which is a = max(0, z), or a "leaky" ReLU.

sigmoid: $a = \frac{1}{1 + e^{-z}}$
tanh: $a = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$
ReLU: a = max(0, z)
leaky ReLU: a = max(G * z, z), where G is some "small" number, typicall 0.01

Often no guidelines for things like what to use for activation function, how to initialize weights, so you often end up trying a number of different approaches.
## Why Do You Need Non-linear Activation Functions
Why non-linear? If we replace the non-linear function with a linear one, then the combination of two linear functions is itself a linear activation function.

This seems lacking in depth. Why is this a problem?
## Derivatives of Activation Functions
Derivative of the sigmoid function:

g'(z) = $\frac{1}{1 + e^{-z}}(1 - \frac{1}{1 + e^{-z}})$

which simplifies to g(z) (1 - g(z)).

Derivative of tanh(z) is $(1 - tanh(z))^{2}$

ReLU derivative is 0 if z<0, 1 if z>0, and undefined at z=0, but to make this simpler we just include 0 in one of the other cases. Leaky ReLU is the same, except that the derivative is 0.01 for z<0.
## Gradient Descent for Neural Networks
Cost function, J($w^{[1]}, b^{[1]}, w^{[2]}, b^{[2]}$) is $\frac{1}{m}\sum^{m}_{i = i}L(\hat{y}, y)$.

$\hat{y}$ is really $a^{[2]}$.

We already have the forward propagation, going back:
$dZ^{[2]} = A^{[2]} - Y$
$dW^{[2]} = \frac{1}{m}dZ^{[2]}A^{[2]T}$
$db^{[2]} = \frac{1}{m}$ np.sum($dZ^{[2]}$, axis = 1, keepdims = True)

Want the keepdims parameter to avoid getting one of those "funny" python not-a-vector values back.

Then, you have
$dZ^{[1]} = W^{[2]T}dZ^{[2]} * g'^{[1]}(Z^{[1]})$ (element-wise product)
$dW^{[1]} = \frac{1}{m} dZ^{[1]}X^{T}$
$db^{[1]} = \frac{1}{m}$ np.sum($dZ^{[1]}$, axis = 1, keepdims = True)

## Backpropagation Intuition (optional)

## Random Initialization
Turns out initializing b terms to zero is OK. If W is a zero matrix, then your hidden units end up having the same weight (symmetric). You're calculating the same value for each input for each node ($x_{1}$ gives the same answer for each node in your hidden layer). Instead, set $W^{[1]}$ to np.random.randn((2,2)) * 0.01, and this starts to weight your nodes differently. Pick 0.01 to make W small, to keep away from the "flat" points of your gradient descent function where the slope is very small.

Sometimes you will pick something different, if you're working with a larger neural network.
