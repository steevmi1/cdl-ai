# Logistic Regression as a Neural Network

## Binary Classification
Forward/back propagation.
Binary classification -- is it a cat (1) or not (0)?
Take RGB pixel matrix, turn into a straight vector. 64x64 image with three channels gives you 12288 features (called nx).

Notation:
m training examples.
X -- matrix, with each column corresponding to one of your input vectors with m columns and nx rows. Y -- 1 row matrix, with each column your y^i label.

## Logistic Regression
Given x, what is the prediction ($\hat{y}$) that y = 1 given x?

y-hat = sigmoid(W^T * x + b)
W is an nx-length vector of features, b is a real number.
The sigmoid function is 1 / (1 + e^z)

Need to learn W and b well enough to give you accurate y-hat.

## Logistic Regression Cost Function
Loss function -- how far off is your predicted value (y-hat) from the real answer? Squared error doesn't work well for gradient descent, go with

$L(\hat{y}, y) = -(y log(\hat{u}) + (1 - y) log(1 - \hat{y}))$

If y == 1, then you want log($\hat{y}$) to be large, so $\hat{y}$ should be large.
If y == 0, then you want log(1 - $\hat{y}$) to be large, so $\hat{y}$ should be small

Cost function -- loss function over all your models.
J(W, b) = $\frac{1}{m} \sum_{i = 1}^{m}(L(y, \hat{y}))$ over all your training inputs

## Gradient Descent
Gradient descent -- find W, b to minimize J(W, b).

Repeat {
  W := W - $\alpha$(d J(W) / d W)
}

## Derivatives

## More Derivative Examples

## Computational Graph

## Derivatives with a Computational Graph

## Logistic Regression Gradient Descent

## Gradient Descent on m Examples
