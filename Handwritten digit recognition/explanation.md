 ## Data Transformation
 Each handwritten 28x28 pixel image is (28, 28) matrix, first of all, this matrix was flatten and the pixeles which are always zero for all training dataset waere dropped and the indexes saved to self.mask. \
 For normalizing the data I just divided each value of the pixel by 255.0 (since the image is grayscale, so, every value is between 0 and 255)
 
 Also, bias was added.(inserted dummy ones at the beginning of the image array)
 
 ## Saving weights
 While training all the weights of the model were added to the `weights_Model_Name.csv` file and the path to this file is given as an argument. \
 if this path is not None the weights will be initialized from the file, else randomly.
 
 ## Results
 
 ### LogisticRegression
 
 LogisticRegression is done with many hyperparameters, all the plots are stored in the plots folder. \
 Also, I have splitted the training data into 2 sets and validated my training on 20% of data. Some results(train and test accuracies) are added bellow. 
 
I would like to mention one thing about the results, it can be seen that, both with regualrization and without the model is not overfitted, but in any case regularization parameter(lambda) was added to the LogisticRegression.
 
*`Number of epochs - 500 | Batch size - 1024 | Learning rate - 1e-1 | Regularization lambda - 0`* \
Train Accuracy - 97.78% \
Test Accuracy - 96.77%

*`Number of epochs - 500 | Batch size - 512 | Learning rate - 1e-1 | Regularization lambda - 0`* \
Train Accuracy - 97.83% \
Test Accuracy - 96.77%

*`Number of epochs - 500 | Batch size - 512 | Learning rate - 1e-1 | Regularization lambda - 1e-2`* \
Train Accuracy - 97.08% \
Test Accuracy - 96.73%

*`Number of epochs - 500 | Batch size - 1024 | Learning rate - 1e-1 | Regularization lambda - 1e-2`* \
Train Accuracy - 97.21% \
Test Accuracy - 97.15%

*`Number of epochs - 500 | Batch size - 1024 | Learning rate - 1e-2 | Regularization lambda - 1e-2`* \
Train Accuracy - 97.25% \
Test Accuracy - 97.07%


 The behaviour of the plot in all cases were similar, so I could chose the number of epochs from any of them. \
 **How to chose the number of epochs?** \
 Here is one of the plots.
 
<p align="center">
<img src="Plots/LogisticRegression with minibatch_gradient_descent ____________ Learning Rate - 0.1 Lambda - 0.01 Batch size - 512.png" alt="LogisticRegression - Epochs and Loss relationship" width="1000" class="center"/> 
</p>

From the plot can be seen that the loss aproximately remains the same after 200 epochs, so, 200 as the number of epochs was fixed and added as a default argument. \
But, for avoiding retraining and wasting the time, epochsdefault value was set to 0 and init_weights to True, which means that the weights of themodel will automatically be loaded from the corresponding file.
 
Accuracy of the LogisticRegression on the whole training data was  **97.24%**

 ### Perceptron

Perceptron is fine when the given data is linearly seperable, but in real life it is not :D \
Therefore, the accuracy of the perceptron is lower than accuracy from Logistic Regression

Moreover, it takes too long to train, in my case I got approximately 96% accuracy after 1000 epochs 

The Perceptron algorithm goes for a maximum fit to the training data and is therefore susceptible to overfitting. 
But with an increase of training data size, overfitting usually decreases.


<p align="center">
<img src="Plots/Perceptron_1000_epochs.png" alt="Perceptron - Epochs and Loss relationship" width="1000" class="center"/> 
</p>

From the plot can be seen that the loss decreases gradually, which means that model is trying to fit every single datapoint

*`Number of epochs - 100`* \
Train Accuracy - 92.53% \
Test Accuracy - 93.17%

*`Number of epochs - 600`* \
Train Accuracy - 96.07% \
Test Accuracy - 96.26%

*`Number of epochs - 1000`* \
Train Accuracy - 96.62% \
Test Accuracy - 96.26%

Accuracy of the Perceptron on the whole training data was  **96.98%**

## Multiclass Classification
Multiclass classification can be done by using One vs Rest method. 

Some methods are added to the files in the `Multiclass` folder, but it is not final and not tested.