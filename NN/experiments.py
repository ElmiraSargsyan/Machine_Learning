"""
If you have done all parts of the homework (which really has nonzero probability)
and want to have more fun, try this experiments.


EXPERIMENT 1: 
Use NN with one hidden layer for simple 1d regression problem, generate random 
(x,y) points and fit your model on it. Compare **Tanh** and **ReLu** activations
for hidden layer (note that you do not need any nonlinearity for output layer).
Also tweak hidden layer size.


EXPERIMENT 2:
Train a multilayer model on MNIST and reach ~95% accuracy on test set. Now 
randomly remove neurons from your network (i.e. set neuron parameters zero). 
Plot test set accuracy versus number of neurons removed. Do the same experiment 
with 
1) Dropout layer added
2) BatchMeanSubtraction layer added.
Comment on results.


EXPERRIMENT 3:
Train a model on MNIST with multiclass criterion. Now backpropagate label through
the network, i.e. find the input that would produce given output. Your label was
one-hot-encoded (only one digit was on the picture). This time backpropagate
label, which has 2 or more ones and see the input picture of the network.

You can do them in separate files. Good luck.
"""

