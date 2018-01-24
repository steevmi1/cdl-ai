# Logistic Regression as a Neural Network

## Binary Classification
Forward/back propagation.
Binary classification -- is it a cat (1) or not (0)?
Take RGB pixel matrix, turn into a straight vector. 64x64 image with three channels gives you 12288 features (called nx).

Notation:
m training examples.
X -- matrix, with each column corresponding to one of your input vectors with m columns and nx rows. Y -- 1 row matrix, with each column your y^i label.

## Logistic Regression
Given x, what is the prediction $\hat{y}(Pr(y = 1|x))$?

$\hat{y} = \sigma(W^{T} x + b)$
W is an nx-length vector of weights of features, b is a real number.
The $\sigma$ function is $\frac{1}{1 + e^z}$

Need to learn W and b well enough to give you accurate $\hat{y}$.

## Logistic Regression Cost Function
Loss function -- how far off is your predicted value ($\hat{y}$) from the real answer? Squared error doesn't work well for gradient descent, go with

$L(\hat{y}, y) = -(y log(\hat{u}) + (1 - y) log(1 - \hat{y}))$

If y == 1, then you want log($\hat{y}$) to be large, so $\hat{y}$ should be large.
If y == 0, then you want log(1 - $\hat{y}$) to be large, so $\hat{y}$ should be small

Cost function -- loss function over all your models.
J(W, b) = $\frac{1}{m} \sum_{i = 1}^{m}(L(y, \hat{y}))$ over all your training inputs

## Gradient Descent
Gradient descent -- find W, b to minimize J(W, b). Start, evaluate, and then take a small step in the "downward" direction of steepest descent to move you towards the minima.

Repeat {
  W := W - $\alpha(\frac{dJ(W)}{dW})$
  b := b - $\alpha(\frac{dJ(b)}{db})$
}

Technically we have a second term (b) in this, so the derivatives are actually parital derivatives, but for simplicity we'll gloss over that for now.

$\alpha$ is our learning rate, how big of a "step" we take. Too small, and it will take a long time to converge. Too large, and you may not converge because you keep stepping "over" the minimum, and never reach it. The derivative will give you + or -, and controls if you need to shrink or increase W and b to move towards your final solution.

## Derivatives

## More Derivative Examples

## Computational Graph

## Derivatives with a Computational Graph

## Logistic Regression Gradient Descent

## Gradient Descent on m Examples
