# Hyperparameter tuning, Batch Normalization and Programming Frameworks
Learning Objectives

Master the process of hyperparameter tuning
# Hyperparameter tuning
## Tuning process
Learning rate ($\alpha$), momentum term ($\beta$), if you're using ADAM then you also have $\beta_{1}$, $\beta_{2}$, and $\epsilon$. Number of layers (L), # of hidden units, the decay factor for your learning rate if you're using that, mini-batch size.

Some are more important than others -- learning rate is most important, followed by momentum, mini-batch size, and number of layers. After that, layers and decay, the rest typically don't get changed from defaults.

Better to randomly select when you look at a combination of hyperparameters, you get better sampling/coverage of the space. Also can use coarse-to-fine scheme -- find a region where things look good, then narrow the ranges for the hyperparameters and then resample.
## Using an apropriate scale to pick hyperparameters
Sometimes when looking at something like number of layers, you just have to look at things regularly (e.g. if you want to look at something with 2, 3 or 4 layers).

In other cases, like learning rate, makes more sense to conver to a log scale to generate your random range. As an example if you want to look at a learning rate between 0.0001 and 1, then on a log scale you're looking at a range of [-4,0] ($10^{-4}$ = 0.0001), then you can do
r = -4 * np.random.randn()
$\alpha$ = $10^{r}$

Also tricky is hyperparameters for EWA. If you want $\beta$ to vary between 0.9 and 0.999, then a linear scale doesn't make as much sense. If we look at (1 - $\beta$), then this goes from 0.1 to 0.001, which if we look at a log scale means we're working between $10^{-1}$ and $10^{-3}$, which we can convert to the same system as we used for learning rate, calculate r, and then use that to determine 1 - $\beta$.
## Hyperparameters tuning in practice: Pandas vs. Caviar
Data can gradually change over time (also your compute environment), so periodically good to go back and re-evaluate your hyperparameters on occasion.

One model at a time approach (maybe computationally limited, so much data and models so large) versus training many models in parallel. First approach, look at day 1 results from day 0, adjust something, then look at results on day 2.
# Batch Normalization
## Normalizing activations in a network
We already know that we can normalize the inputs to speed up our learning. Question is if we can normalize a hidden layer $a^{[l]}$ so that we can learn w and b faster?

For a neural network, we usually will normalize the z values. Can do this the standard way

$$\mu = \frac{1}{m} \sum_{i} z^{i}$$
$$\sigma^{2} = \frac{1}{m} \sum_{i} (z^{i} - \mu)$$
$$z^{i}_{norm} = \frac{z^{i} - \mu}{\sqrt{(\sigma^{2} - \epsilon)}}$$

This normalizes with mean 0 (and variance 1), but sometimes we don't alwayts want this. In that case, what we can do is to compute

$$ \hat{z^{i}} = \gamma z^{i}_{norm} + \beta$$

Where $\gamma$ and $\beta$ are now learnable parameters for your model. You can set gamma and beta to give you a standard normalization.

## Fitting BatchNorm into a neural network
Each node in a neural network is computing two things -- z, and a. With BatchNorm, what we do is compute z, then apply BatchNorm to compute $\hat{z}$. Use $\hat{z}$ to compute a, then use this as our next input to compute the next z. Then, just keep repeating, after computing this next z, use BatchNorm to compute $\hat{z}$ for this step. This gives you a whole new set of $\beta$ and $\gamma$ parameters for each step, and you can optimize them (ADAM, and so on) the way you would any other.
## Why does BatchNorm work?
Why does this speed things up? First, like any other normalization. Also helps to handle the "covariate shift" effect, where you have your data distribution changing (a batch of black cats, then a batch of cats with other colorings). This works to prevent the early layers from having a larger effect than the later layers.
## BatchNorm at test time
BatchNorm normally works on minibatches of input data, what happens when we move to test? When you use it in training you use your batch size to divide and calculate your $\mu$ and $\sigma$. When you test, your batch size is 1, which doesn't work. Typically people will just keep a running average for the two values as they run through their test set which will give a reasonable value for them.
# Multi-class classification
## Softmax regression
What happens when we move to a multi-class problem (cats, and dogs, and birds, and "none of the above"...)? Now your final layer has four output units (has to match the number of classes you have), and each unit will be the probability that your object belongs to that class (and must sum to 1). The method that people use to accomplish this is softmax, where the activation function becomes
$$ t = e^{(z^{[l]})} $$
$$ a = \frac{e^{z^[l]}}{\sum_{j = 1}^{4}t_{i}} $$
For the first step, t is a temporary variable and is an element-wise exponentiation (if $z^{[l]}$ is a (4,1) vector then t is similarly a (4,1) vector).
## Training a softmax classifier
Softmax is really just logistic regression generalized to more than two classes. When you get to backprop, then you compute

$$ dz^{[l]} = \hat{y} - y$$
# Introduction to programming frameworks
## Deep learning Frameworks
So far we've just built everything from scratch. Works OK for learning and small examples, but as you build up it's not so efficient for you to be implementing everything yourself. This is where the frameworks come in, as they've taken care of most of the heavy lifting for you.
+ caffe/caffe2
+ CNTK
+ DL4J
+ Keras
+ Lasagne
+ mxnet
+ PaddlePaddle
+ TensorFlow
+ Theano
+ Torch
All are commonly used, and changing/evolving rapidly. In many cases selection is up to personal choice, as they're all approximately equivalent.
## TensorFlow
As an example, assume our cost function we're looking to optimize is
$$ J(w) = w^{2} - 10w + 25 $$
In tensorflow, this becomes
```python
import numpy as np
import tensorflow as tf
w = tf.Variable(0, dtype = tf.float32)
cost = tf.add(tf.add(w**2, tf.multiply(-10., w)), 25)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))
```
Advantages to this are that backprop are done for you automatically, and there's also overloading, so you could just define your cost function as
```python
cost = w**2 - 10*w + 25
```
You can also start to do things like
```python
for i in range(1000):
  session.run(train)
print(session.run(w))
```
placeholder variable in tensorflow -- something that will be provided "later". 
