# Practical Aspects of Deep Learning
## Learning Objectives
Recall that different types of initializations lead to different results

Recognize the importance of initialization in complex neural networks.

Recognize the difference between train/dev/test sets

Diagnose the bias and variance issues in your model

Learn when and how to use regularization methods such as dropout or L2 regularization.

Understand experimental issues in deep learning such as Vanishing or Exploding gradients and learn how to deal with them

Use gradient checking to verify the correctness of your backpropagation implementation

# Setting up your machine learning application

## train/dev/test Sets
Working on a neural network is an interative process, with lots of options. Choices around # of layers, # of hidden units, learning rates, activation functions, and other hyperparameters.

Often, intuitions from one domain (e.g. NLP) don't transfer to another.

Often, will split your data into three sets:
- training (bulk of data)
- hold-out/crossvalidation ("dev")
- test

Best practices used to be 70/30 (train/test) or 60/20/20 (train/dev/test).

Dev just needs to be big enough to try out different algorithms, so 20% may be too much data. Similarly, test can also be typically much smaller, so as your overall data set increases in size you can start considering 98/1/1 (or greater) splits.

Also can have mismatched data sets (train high-res images, dev/test come from cellphone cameras), so should try to make dev and test come from the same data set.
## Bias/Variance
High bias - underfitting.
High variance - overfitting.

Easy to see this in 2-D data, can just plot and look at. With higher-dimensional data, this isn't really a workable approach. Here, can look at train and dev error rates -- low train error/high dev error is high variance/overfitting, high train and dev error is about the same rate is high bias/underfitting, and high train error and even higher dev error is both high bias *and* high variance. Part of what constitutes "high" error is a comparison to human, so if you have ~ 0.01% error then 15% is high error, but if a human has 15% error then a 15% error may be considered low.
## Basic Recipe for Machine Learning
High bias (training data) - then try things like a bigger network, or training longer (or looking at different architectures).

Then, after fixing bias ask if you have high variance (dev set performace). Best way to fix this is more data (not always possible), otherwise can regularize the data. Sometimes different NN architectures.

Usually a bigger network never hurts, it just takes longer computationally.
# Regularizing Your Neural Network
## Regularization
Remember logistic regression. Here, you're trying to minimize the cost function J(w, b):
$J(w, b) = \frac{1}{m} \sum_{i = 1}^{m} L(\hat{y}^{(i)}, y^{(i)})$
which now becomes
$J(w, b) = \frac{1}{m} \sum_{i = 1}^{m} L(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}\Vert w \Vert^{2}_{2}$
where $\lambda$ is the regularization parameter and $\Vert w \Vert^{2}_{2}$ is the norm of w squared, which is just $\sum_{j = 1}^{n_{x}} w_{j}^{2}$, which works out to $w^{T} w$. Often referred to as L2 regularization.

W is usually a high dimension parameter, and b is just one parameter, so often it's just simpler to exclude this from regularization and just focus on w.

Now, if we move to a neural network then we have
$J(w^{[1]}, b^{[1]}....w^{[L]}, b^{[L]}) = \frac{1}{m} \sum_{i = 1}^{m} L(\hat{y}^{(i)}, y^{(i)})$
which becomes
$J(w^{[1]}, b^{[1]}....w^{[L]}, b^{[L]}) = \frac{1}{m} \sum_{i = 1}^{m} L(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \sum_{l = i}^{L}\Vert w^{[l]} \Vert^{2}_{F}$
and here the norm works out to be $\sum_{i = i}^{n^{[l - 1]}} \sum_{j= 1}^{n^{[l]}} (w_{ij}^{[l]})^{2}$, which is called the "Frobenius norm".

Now we also have to adjust for backprop as well, so that $dw^{[l]}$ = (from backprop) + $\frac{\lambda}{m} w^{[l]}$.
## Why Regularization Reduces Overfitting
What happens when you regularize? If you set $\lambda$ to be very large, that forces the weights to be very small [WHY?], which in a sense is equivalent to zeroing out nodes in your network, and force it to be simpler.
## Dropout Regularization
Set some probability for each node in your hidden nodes to be knocked out -- remove incoming and outgoing links to the node. Most common form is "inverted dropout".

   d3 = np.random.randn(a3.shape[0], a3.shape[1]) < keep_prob
   a3 = np.multiply(a3, d3)
   a3 /= keep_prob

Last step is needed because we're losing a number of nodes, so this will correct the value of z so that it's still in line with what's expected. Then, at test time you don't do the drop out.  
## Understanding Dropout
What's really happening is that because you can't rely on any one particular input, that forces the weights to be spread out. Also, can vary your keep_prob by layer, and lower it (or eliminate completely) for layers where you are less worried about overfitting.

The downside is that the cost function is less well defined/harder to calculate. Typically will turn off dropout to verify that learning rate is monotonically decreasing, then turn it on.
## Other Regularization Methods
Data augmentation -- do things like flip pictures or take random distortions and transformations of the input, which gives you more data to train on but is not as good as getting more, new data.

Early stopping -- plot training and dev error, and often you find dev error goes down for a while, then starts to increase. Find point when increase starts, and just stop your training around that point. Drawback is that you're stopping the optimization of your cost function early.
# Setting up your optimization problem

## Normalizing Inputs
Typically two steps. First, subtract the mean.
$\mu = \frac{1}{m} \sum_{i = 1}^{m} x^{(i)}$
$x := x - \mu$
Then, normalize the variance.
$\sigma^{2} = \frac{1}{m} \sum_{i = 1}^{m} x^{(i)} ** 2$ (element-wise square)
$x /= \sigma^{2}$

Also, need to use this for your test set as well.

Why do we do this? Often, one set of input parameters could have a large range, and a second set a very small one (0 < x1 < 1000 versus 0 < x2 < 1), which gives you a much more elongated space to "step" with gradient descent. Normalizing makes this a much more regularized space, so that you can take fewer (and larger) steps to get to the minima.
## Vanishing/Exploding Gradients
Look at deep networks -- if W is even just a little bit bigger than the identity matrix, then your activations increase exponentially. Likewise, if it's just even a little bit smaller than I, then your activations start to decrease exponentially. This ends up impacting your training, and forces you to take a lot longer to work at learning due to higher (or lower) activations. This is one of the big challenges when working with larger neural networks.
## Weight Initialization for Deep Networks
Often find that setting the variance of your initial weight matrix to be $\frac{1}{n}$, helps to force the weight matrix to be constrained so that it doesn't explode or vanish. ReLU it works out that it's a bit better to use $\frac{2}{n}$. Can also look at Xavier initialization ($\tanh \sqrt{\frac{1}{n^{[l - 1]}}}$)
## Numerical Approximations of Gradients

## Gradient Checking

## Gradient Checking Implementation Notes
