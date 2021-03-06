# Logistic Regression as a Neural Network
*Learning Objectives*
- Build a logistic regression model, structured as a shallow neural network
- Implement the main steps of an ML algorithm, including making predictions, derivative computation, and gradient descent.
- Implement computationally efficient, highly vectorized, versions of models.
- Understand how to compute derivatives for logistic regression, using a backpropagation mindset.
- Become familiar with Python and Numpy
- Work with iPython Notebooks
- Be able to implement vectorization across multiple training examples
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
Don't need extensive Calculus, but do need some.

Equation for a straight line, slope is rise over run, which is also the derivative.

## More Derivative Examples
Look at more complex equations than just a straight line. Polynomial equations. Slope now differs depending on your inputs (e.g. if f(a) = $a^{2}$, your slope becomes 2a).

f(a) = $a^{3}$, then slope becomes $3a^{2}$, and so on.

f(a) = ln(a) [natural log function], then derivative of this is $\frac{1}{a}$.

Take away -- derivative is the slope of a function, and can vary based on whwere you are on the line.

## Computational Graph
Assume you have J(a,b,c) which is the function 3(a + bc), you solve this in three steps.

u = bc
v = a + u
J = 3v

Turn this into a graph, three input nodes (a, b, c), b and c combine to form node u, u and a combine to form v, v forms J. Going backwards in this graph is equivalent to calculating the derivative.

## Derivatives with a Computational Graph
Start with the computational graph from the previous lecture. If we start with $\frac{dJ}{dv}$, then we get a slope of 3 (which makes sense, since J = 3v). If we then look at changing a and how that impacts J, then the calculus chain rule says that this becomes $\frac{dJ}{dv}$ times $\frac{dv}{da}$, and the slope for the second term is 1 (but $\frac{dJ}{da}$ is still 3....).

In code, refer to this as "d" and the variable you're interested in just for simplicity/brevity.

Finish working backwards, $\frac{dJ}{db} = \frac{dJ}{du} \, \frac{du}{db}$, and works out to 6, and similarly $\frac{dJ}{dc}$ works out to 9 for this set of inputs.

## Logistic Regression Gradient Descent
Recap:
$z = w^{T}\,x + b$
$\hat{y} = a = \sigma(z)$
$L(a, y) = -\,(y\,log(a) + (1 - y)\,log(1 - a))$

If we assume two features ($x_{1}$ and $x_{2}$), then we have two ws ($w_{1}$ and $w_{2}$) and b. These combine so that you have
$z = w_{1}x_{1} + w_{2}x_{2} + b$
You can now calculate forward prop, and using calculus calculate the derivatives to figure out the backwards prop. Derivates are multiplied by $\alpha$ to give you your adjustment at each step of the way.

## Gradient Descent on m Examples
More concretely, one step is:
J = 0; dw1 = 0; dw2 = 0; db = 0

for i = 1 to m:
    $z^{(i)} = w^{T}x + b$
    $a^{(i)} = \sigma(z)$
    J += $-[y^{(i)}log(a^{(i)}) - (1 - y^{(i)})log(1 - a^{(i)})]$
    $dz^{(i)} = a^{(i)} - y^{(i)}$
    dw1 += $x_{1}^{(i)}dz^{(i)}$
    dw2 += $x_{2}^{(i)}dz^{(i)}$
    db += $dz^{(i)}$

J /= m
dw1 /= m
dw2 /= m
db /= m

dw1, dw2, and db are accumulators. Then, at the end use dw1, dw2, db to update w1, w2 and b, and then go back to take your next gradient descent step.

Big drawback is as you add features, your for-loop will become a big bottleneck. Need to attack this using vectorization, to improve efficiency.
# Python and Vectorization

## Vectorization
If you have a pair of vectors to multiply, by hand you need a for-loop. In python, can use np.dot() function.

Can get some speedups by letting numpy handle things (vectorization for the CPU), instead of rolling your own or relying on brute force.
## More Vectorization
Rule of thumb -- wherever possible avoid explicit for-loops. numpy has many built-in functions that already vectorize things (np.exp(), np.dot()). Go back and look at our example from GD on m Examples, we have two for-loops. Outer one, and a second one where we figure out dw1, dw2, db. Can use vectorization to replace the inner for-loop, where we calculate dw1, dw2, db.

## Vectorizing Logistic Regression
z = np.dot(W.T, X) + b

Does all the vectorized operations all at once. Even where b is a single value, python uses broadcasting to turn it into a 1xm vector that can be added.

Now, we just need to figure out how to vectorize the second step, where we calculate the sigmoid for z.

## Vectorizing Logistic Regression's Gradient Output
np.sum(dZ) / m -- to calculate db.

dw becomes 1/m * X * dZ.T

Five equations, and can do a single iteration of gradient descent without having to use a for-loop. May still need to do this if you want to do a number of iterations.
## Broadcasting in Python
Look at a 3 x 4 matrix, want to sum each column and then divide each element in the column by the sum.

Called reshape when not strictly necessary, but a habit to deal with case when you're not entirely sure what shape your output is.

Python coerces inputs into whatever format makes sense -- if you add m x n matrix and a 1 x n vector, then python will turn the 1 x n matrix into m x n by simply adding rows of the original vector until it's the right size.
## A note on python/numpy vectors
Can get some strange results (add a row to a column, get a matrix).

a = np.random.randn(5) gives you what's called a "rank 1 arrayt", which is neither a row vector nor a column vector. Better off being explicit, and creating things as rows or columns explicitly on creation. Also can use assert statements like
assert(A.shape == (5,1))
as a quick check to verify that you are working with what you think you are. Can also use "reshape()" to help ensure things are in the right shape.
## A quick tour of Jupyter/ipython notebooks
Running notebooks from Coursera. Kill kernel if necessary, run all blocks even if they print nothing. Submit button right at the top when you're done.
## Explanation of logistic regression cost function (optional)
Derivation of the cost function.
