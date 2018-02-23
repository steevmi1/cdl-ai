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
In this case, each $V_{t}$ can be thought of as approximating $\frac{1}{1 - \beta}$ days temperature (so if $\beta$ is 0.9, then $V_{t}$ approximates 10 days of temperature). As your window increases the curve continues to smooth, but you can also see it shifts (becomes a bit less accurate).
## Understanding Exponentially Weighted Averages
Remember $V_{t} = \beta V_{t-1} + (1 - \beta) \theta_{t}$. You can see that you keep substituting for the $V_{t-1}$ term, to expand this out. When you rearrange the equations, then the question becomes what power do you need to raise $\beta$ to to be approximately $\frac{1}{e}$, and that tells you how many days that your model is taking into account (so if $\beta$ is 0.9, then $0.9^{10}$ is approximately $\frac{1}{e}$, and you're working with approximately 10 days as part of your EWA). If $\beta$ is 0.98, then the exponent needs to be 50.
## Bias Correction in Exponentially Weighted Averages
Basic algorithm is to start with $V_{0}$ being equal to 0, which means that $V_{1}$ is really $0.02 \theta_{1}$, and $V_{2}$ is $0.98 * 0.02 \theta_{1}$ + $0.02 \theta_{2}$, and this really distorts your initial part of the curve. If you change the equation to divide $V_{t}$ by $(1 - \beta^{t})$, so in the case where we're looking at t = 2 that means that we now have $\frac{V_{2}}{1 - (0.98)^{2}}$, which is $\frac{V_{2}}{0.0396}$. As t is small, this tends to be more of a straight average of your $\theta$ values, but as t increases the denominator becomes closer and closer to 1 (so the correction essentially vanishes).
## Gradient Descent with Momentum
Sometimes your cost function is somewhat distorted, which means that a large learning rate causes you to diverge/overstep, and you can only get it to converge by using a small learning rate, but as a result you take a lot of little steps to oscillate towards your solution (which takes a lot of time because you're moving more "up and down" and not left-to-right towards your solution). To work around this, we can use a bias-corrected EWA for our dW and db, which should help to smooth out our steps for gradient descent. Then, we use the corrected EWAs multiplied by our learning rate, and use that to correct W and b at each step.
## RMSProp
Another method to slow down your up/down and not impact (or speed up) your left-right is for every iteration t:
Compute dW, db as normal
$S_{dw} = \beta S_{dw} + (1 - \beta) dW^{2}$
$S_{db} = \beta S_{db} + (1 - \beta) db^{2}$
$W = W - \alpha \frac{dW}{\sqrt{S_{dw}}}$
$b = b - \alpha \frac{db}{\sqrt{S_{db}}}$
What you're really looking for is for dW to be small, which makes your W adjustments large (number divided by a small number is a larger number), and your db to be large, which will shrink your adjustments to b.

Do have a concern, if your $S_{dw}$ is small enough, you end up dividing by 0 and "blowing up" your adjustment to W. Next lecture will show how to add numerical stability so that you avoid this.
## Adam Optimization Algorithm
$V_{dw} = 0$, $S_{dw} = 0$, $V_{db} = 0$, $S_{db} = 0$
On iteration t:
Compute dW, db using current mini-batch
$V_{dw} = \beta_{1} V_{dw} + (1 - \beta_{1}) dW$, $V_{db} = \beta_{1} V_{db} + (1 - \beta_{1}) db$
$S_{dw} = \beta_{2} S_{dw} + (1 - \beta_{2}) dW^{2}$, $S_{dw} = \beta_{2} S_{dw} + (1 - \beta_{2}) dW^{2}$
$V^{corrected}_{dw} = \frac{V_{dw}}{1 - \beta_{1}^{t}}$, $V^{corrected}_{db} = \frac{V_{db}}{1 - \beta_{1}^{t}}$
$S^{corrected}_{dw} = \frac{S_{dw}}{1 - \beta_{2}^{t}}$, $S^{corrected}_{db} = \frac{S_{db}}{1 - \beta_{2}^{t}}$
$W = W - \alpha \frac{V^{corrected}_{dw}}{\sqrt{S^{corrected}_{dw}} + \epsilon}$
$b = b - \alpha \frac{V^{corrected}_{db}}{\sqrt{S^{corrected}_{db}} + \epsilon}$

This method ends up with a number of different hyperparameters to be tuned. $\alpha$ (learning rate) will need to be tuned "manually", $\beta_{1}$ is typically started at 0.9, $\beta_{2}$ is typically started at 0.999, and $\epsilon$ is usually started at $10^{-8}$. Usually defaults for last three parameters are good enough and don't need to be tuned. Beta parameters are first and second moments respectively.
## Learning Rate Decay
Another option for improving performance is to slowly decrease your learning rate over time ("decay"). Conceptually, you want to take bigger steps at the beginning, and as you get closer to your solution you want to start taking smaller and smaller steps.

1 epoch is one pass through your training data. For each epoch, set your learning rate to be $\alpha = \frac{1}{1 + decay\_rate * epoch\_number} \alpha_{0}$.

Other strategies to do the decay -- can do an exponential rates, discrete steps, RMS-style decays.

Decaying the learning rate is usually lower in the list of hyperparameters to tune.
## The Problem of Local Optima
Local optima used to be viewed as a problem, but as you start working in higher dimensional spaces it turns out that local optima that could potentially "trap" you and mask the true optima end up being really saddle points -- you have 10,000 (or more) points that need to converge, which isn't something that often happens.

One problem that does exist is in plateaus, where a bunch of your points are close enough to 0 that you see a slowdown because it takes a number of steps to get away from the plateau region.
