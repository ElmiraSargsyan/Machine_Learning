"""
We are using well known *MNIST*(http://yann.lecun.com/exdb/mnist/) as our dataset.
Lets start with *cool visualization*(http://scs.ryerson.ca/~aharley/vis/). 
The most beautiful demo is the second one, if you are not familiar with 
convolutions you can return to it in further lectures lectures. 
"""

import os
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from modules import Dense, SoftMax, Sequential, Sigmoid, ReLU, SoftPlus, Dropout, BatchMeanSubtraction
from criterions import MultiLabelCriterion
from optimizers import sgd_momentum, adam_optimizer

from utils.batch_generator import get_batches
from utils.metrics import accuracy_score

import matplotlib.pyplot as plt

# Fetch MNIST dataset and create a local copy.
if os.path.exists('mnist.npz'):
    # data = np.load('mnist.npz',)
    with np.load('mnist.npz', 'r', allow_pickle=True) as data:
        X = data['X']
        y = data['y']
else:
    X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
    # X, y = mnist.data / 255.0, mnist.target
    np.savez('mnist.npz', X=X, y=y)

print("data shape:", X.shape, y.shape)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
YOUR TASKS:
- **Compare** `ReLU`,`Sigmoid`, `SoftPlus` activation functions. 
You would better pick the best optimizer params for each of them, but it is 
overkill for now. Use an architecture of your choice for the comparison and let
it be fixed.

- **Try** inserting `BatchMeanSubtraction` between `Dense` module and 
  activation functions.

- Plot the losses both from activation functions comparison and 
  `BatchMeanSubtraction` comparison on one plot. Please find a scale (log?) 
  when the lines are distinguishable, do not forget about naming the axes, 
  the plot should be goodlooking. You can submit pictures of this plots.

- Write your personal opinion on the activation functions, think about 
  computation times too. Does `BatchMeanSubtraction` help?

- **Finally**, use all your knowledge to build a super cool model on this 
  dataset, do not forget to split dataset into train and validation. Use 
  **dropout** to prevent overfitting, play with **learning rate decay**. 
  You can use **data augmentation** such as rotations, translations to boost 
  your score. Use your knowledge and imagination to train a model. 

- Print your accuracy at the end of the code. Also write down the best accuracy 
  that you could get on test set. It should be around 90%.

- Hint: logloss for MNIST should be around 0.5.

- Suggestions: it can be easyer to use jupyter notebook for experimenting,
  but final results MUST be in this file (or multiple files)

Write down all your answers at the end of the file as a comment.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def acc(net, X, y):
  return 100*accuracy_score(net.forward(X).argmax(axis=-1), y.argmax(axis=-1))
  


def fit(X, Y, X_val, y_val, net, n_epoch, batch_size, criterion, optimizer, optimizer_config, optimizer_state):
  loss_history = []
  #acc_history = []
  for i in range(n_epoch):

    
    for x_batch, y_batch in get_batches(X, Y, batch_size):

      net.zeroGradParameters()
      
      # Forward
      predictions = net.forward(x_batch)
      loss = criterion.forward(predictions, y_batch)
  
      # Backward
      dp = criterion.backward(predictions, y_batch)
      net.backward(x_batch, dp)
      
      # Update weights
      optimizer(net.getParameters(), 
                  net.getGradParameters(), 
                  optimizer_config,
                  optimizer_state)      
        
    loss_history.append(loss)
    #acc_history.append(100*accuracy_score(predictions.argmax(axis=-1), y_batch.argmax(axis=-1)))
    """
    print(f"Epoch: {i}")
    print("train accuracy:  %.2f " %acc(net, X_train, y_train))
    print("validation accuracy:  %.2f " %acc(net, X_val, y_val))
    """
    
  return loss_history 

# One hot encoding labels
enc = OneHotEncoder()
y = np.array(enc.fit_transform(y.reshape(-1, 1)).todense()) 

# Shuffling data and spliting into training valistation and test sets
N = X.shape[0]    
idx = np.arange(N)
np.random.shuffle(idx)

X = X[idx]
y = y[idx]

s1 = int(0.7*N)
s2 = int(0.85*N)

X_train, y_train = X[0:s1], y[0:s1]
X_val, y_val = X[s1:s2], y[s1:s2]
X_test, y_test = X[s2:], y[s2:]
"""
print('Training', X_train.shape, y_train.shape)
print('Validation', X_val.shape, y_val.shape)
print('Test', X_test.shape, y_test.shape)
"""
"""
###Best model
criterion = MultiLabelCriterion()

# Optimizer params
optimizer_config = {'learning_rate' : 1e-2, 'momentum': 0.9}
optimizer_state = {}

# Looping params
n_epoch = 50
batch_size = 128

nn = Sequential()
nn.add(Dense(784, 784))
nn.add(Dropout(0.6))
nn.add(ReLU())
nn.add(Dense(784, 50))
nn.add(ReLU())
nn.add(Dense(50, 10))
nn.add(SoftMax())


loss_history = fit(X_train, y_train, X_val, y_val, nn, n_epoch, batch_size, criterion, optimizer_config, optimizer_state)
print()
print()
print("Test accuracy:  %.2f " %acc(nn, X_val, y_val))

np.save("weights.npy", nn.getParameters)

"""
#####Experiments
# Loss function
criterion = MultiLabelCriterion()

# Activation function
activations = {"ReLU": ReLU, "Sigmoid": Sigmoid, "SoftPlus": SoftPlus}


# Optimizer params
# for sgd
#optimizer = sgd_momentum
#optimizer_config = {'learning_rate' : 1e-2, 'momentum': 0.9}
# for adam
optimizer = adam_optimizer
optimizer_config = {'learning_rate' : 1e-2, 'beta1': 9e-1, 'beta2' : 999e-3, 'epsilon': 10e-8}

optimizer_state = {}

# Looping params
n_epoch = 20
batch_size = 1024

for activation_name, activation in activations.items():
  nn1 = Sequential()
  nn1.add(Dense(784,100))
  nn1.add(activation())
  nn1.add(Dense(100,50))
  nn1.add(activation())
  nn1.add(Dense(50,10))
  nn1.add(SoftMax())

  print("****************************************************")
  print(f"Training NN with {activation_name} without Batch Normalization")
  print("****************************************************")

  loss_history1 = fit(X_train, y_train, X_val, y_val, nn1, n_epoch, batch_size, criterion, optimizer, optimizer_config, optimizer_state)

  nn2 = Sequential()
  nn2.add(Dense(784,100))
  nn2.add(BatchMeanSubtraction(1))
  nn2.add(activation())
  nn2.add(Dense(100,50))
  nn2.add(BatchMeanSubtraction(1))
  nn2.add(activation())
  nn2.add(Dense(50,10))
  nn2.add(SoftMax())

  #print("****************************************************")
  print(f"Training NN with {activation_name} with Batch Normalization")
  print("****************************************************")
  
  loss_history2 = fit(X_train, y_train, X_val, y_val, nn2, n_epoch, batch_size, criterion, optimizer, optimizer_config, optimizer_state)

  
  print()
  print(f"Test accuracy - {activation_name} without Batch Normalization:  %.2f " %acc(nn1, X_val, y_val))
  print(f"Test accuracy - {activation_name} witհ Batch Normalization:  %.2f " %acc(nn2, X_val, y_val))
  print()

  plt.figure(figsize = (15, 10))
  plt.title(f"Training Loss - {activation_name}")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.plot(loss_history1, label = 'without Batch Normalization')
  plt.plot(loss_history2, label = 'with Batch Normalization')
  plt.legend()
  plt.savefig(activation_name + ".png")
plt.show()


# ...

# Your answers here
"""
Results with Gradient Descent

Test accuracy - ReLU without Batch Normalization:  95.77 %
Test accuracy - ReLU witհ Batch Normalization:  95.87 %

Test accuracy - Sigmoid without Batch Normalization:  71.80 %
Test accuracy - Sigmoid witհ Batch Normalization:  72.00 %

Test accuracy - SoftPlus without Batch Normalization:  95.78 %
Test accuracy - SoftPlus witհ Batch Normalization:  95.73 %

Results with Adam

Test accuracy - Sigmoid without Batch Normalization:  92.98 %
Test accuracy - Sigmoid witհ Batch Normalization:  93.27 %

# All the plots can be found in the "Plots" folder.

The best accuracy I could get was 97.27 %
Training weights are stored in the weights.npy file
the file was too large for pushing into GitHub.
File can be found here within 1 week https://we.tl/t-dQ8fhOnnyy 
"""