# Optimization Algorithms
Learning Objectives

- Remember different optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
- Use random minibatches to accelerate the convergence and improve the optimization
- Know the benefits of learning rate decay and apply it to your optimization

## Mini-batch Gradient Descent
Working with large amounts of data, so training by default is slow. Look for opportunities to speed this up wherever possible.

Need to process all your training examples, then take a step, then process again to take your next step. You can split your training set into "baby" training sets (called mini-batches), use a superscript with curly braces to denote each mini-batch.

$X^{(i)}$
$Z^{[l]}$
$X^{\{t\}}$

Now, you change gradient descent to loop over all of your $t$ minibatches, and for each one calculate Z and A for each mini-batch. Compute your cost for each mini-batch, and also your backprop.

One pass through a training set is called an epoch.
## Understanding mini-batch Gradient Descent
With batch gradient descent, you expect your cost to go down at each step (and if it goes up, even for one step, then something's wrong). With mini-batch, you may see cost go up from one step to the next, but overall trend should go down. Extremely noisy.

Selecting mini-batch size. If you pick a batch size of 1, then this is called stochastic gradient descent.

Batch gradient descent -- will take a long time, with that much data.

Stochastic gradient descent -- lose all your speedup due to vectorization.

Rules of thumb? If small training set (< 2000), then probably just use batch gradient descent. Typically go with a power of 2 for your batch size (64, 128, 256, ....). Also want to make sure that your mini-batch size will also fit into CPU (or GPU) memory.

Mini-batch size is just another hyperparameter....
## Exponentially Weighted Averages
Look at daily temperatures for London for last year. Plot is a bit noisy, but does look to have a curve in there. Can look at a weighted average to start removing some of the noise. Assume that you have an array, $\theta$, where each item in the array is the temperature for that day of the year (so 365 elements in the array). Then

$V_{0} = 0.0$
$V_{1} = 0.9 V_{0} + 0.1 \theta_{1}$
$V_{2} = 0.9 V_{1} + 0.1 \theta_{2}$
...
$V_{t} = 0.9 V_{t-1} + 0.1 \theta_{t}$

More generally, write this as

$V_{t} = \beta V_{t-1} + (1 - \beta) \theta_{t}$
In this case, each $V_{t}$ can be thought of as approximating $\frac{1}{1 - \beta}$ days temperature (so if $\beta$ is 0.9, then $V_{t}$ approximates 10 days of temperature).
## Understanding Exponentially Weighted Averages

## Bias Correction in Exponentially Weighted Averages

## Gradient Descent with Momentum

## RMSProp

## Adam Optimization Algorithm

## Learning Rate Decay

## The Problem of Local Optima
