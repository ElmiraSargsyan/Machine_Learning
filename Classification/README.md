# Classification with Logistic Regression and Perceptron using Gradient descent, SGD and mini-Batch GD

## LogisticRegression
Logistic regression is implemented using only numpy.


### LogisticRegression

The model to fit classified data and predict on the new data.

#### _generate_initial_weights

Initialize weights somehow (it would be better, if the initialization was random.)

#### loss

Calculates logistic regression loss on the given data.


#### gradloss

Calculates the gradient of loss function respect to model's weights.

#### hesianloss

Calculates the hessian matrix of loss function respect to model's weights. It is used in Newton-Raphson method.

### Gradient Descent

Stochastic, minibatch, batch gradient descents. You might want to shuffle data first for stochastic and minibatch gradient descents. Batch gradient descent fucntion should yield the gradient only once. So should do Newton Raphson method's function.
Gradient descent function yields updates, not weight vectors. So after gradient descent yields $u$, the model should update weights like $w^{(new)} = w^{(old)} - u$.

## Perceptron

You should replace weight initialization, implement calculations of loss function, gradient of the loss function and predict method. Implementation is similar to LogisticRegression model.

## Plotting 

Run `plot.py` to see the training process of the model. You can choose the model, solution method and some extra parameters, e.g. `python plot.py --update-method stochastic_gradient_descent --num-datapoints 10`. Use `python plot.py --help` to see the full list of instructions.
