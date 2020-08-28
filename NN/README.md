# Basic Artificial Neural Networks

This is the only assignment of this week. The goal of this homework is simple, yet an actual implementation may take some time. We are going to write an Artificial Neural Network (almost) from scratch. The software design of was heavily inspired by [Torch](http://torch.ch) which is one of the most convenient neural network environments when the work involves defining new layers. 

This homework has the following parts. 

1. Differentiation exercises, (5%)
2. Modules of neural network, (50%)
3. Testing the framework on synthetic datasets, (5%)
4. Making powerful model on MNIST dataset, 
   - getting score above 95% in test set  (20%)
   - comparing different activation functions (10%)
   - examining the effects of Dropout and BatchNorm (10%)
5. Autoencoders (optional)
6. Other interesting experiments (optional)



## Assignment guide
Neural Networks (NN) became popular due to many facts. One of them is *extensibility*. NN is composed of modules (blocks), where each module 
implements some functionality. By combining these modules one can build state-of-the-art NNs with existing NN packages. Recent NN wonderful ideas 
often require just defining a new module or slightly changing an existing one. This section should help you to understand what the modules are and 
what other abstractions are used in NNs. 

At first, let's think of NN as of black box model (we don't care or know how it works inside, but when we ask it to do something it politely does). What functionality then should the black box implement to be practical? Well, the same as other discriminative models! 
- it should be able to give a predictions (let's call it **output**) if provided with **input** data
- it should be learnable (there should be a mean to adapt model to the given data)

The first point implies the black box should implement a function (we call it **forward**).

        text{output = NN.forward(input)

The second point means the model should be able to compute a gradient with respect to (w.r.t.) its parameters and return them to us. We would use this gradient to perform parameters update. The computation of the gradient is done during **backward** call.

        NN.backward(input, criterion (output, target))

and gradients retrieved with, lets say:

        gradParameters = NN.getGradParameters()

the **criterion** should tell quantively how wrong your model is if predicting **output** when **target** expected (loss function). 

After the *lecture* it should be clear how we use the gradient: we use one of the **optimizers** (*sgd*, *adaGrad*, *Adam*, *nag*) to perform parameter update.

#### Summary
At this point we have seen three important abstractions: 
- black box
- criterion
- optimizer

#### Workflow
The workflow is split into 3 steps (yeah, kind of abstractions):
- forward pass
- backward pass
- parameters update

Let's detail further the workflow.

Forward pass: 

    output = NN.forward(input)
    loss =  criterion.forward(output, target)

Backward pass: 
    
    NNGrad = criterion.backward(output, target)
    NN.backward(input, NNGrad)

Parameters update:

    gradParameters = NN.getGradParameters()
    optimizer.update(currentParams, gradParameters)

There can be slight technical variations, but the high level idea is always the same. It should be clear about forward pass and parameters update, the most struggling is to understand backprop. 

## White box
Last thing before discussing backprop is to whiten our black box, we are old enough to know the truth. 

As said in introduction NN is composed of modules and surprisingly these modules are NNs too by definition! Remember, left or right child in binary tree is also a tree, and the leaves are trees themselves. Kind of the same logic it is here too, but is about directed acyclic graphs (you can think of a chain for the first time). You can find "starter" and "final" nodes in these graphs (start and end of a chain), the data goes through the graph according to the directions, each node applies its **forward** function till the last node is reached. On backward pass the graph is traversed form "final" nodes to "starter" and each node applies **backward** function to whatever previous node passed. 

So the cool thing is: each node is a NN, every connected subgraph is NN. We defined everything we need already, you just need a set of "simple" NNs which are used as building blocks for complex models! That is exactly what the NN packages implements for you and what you are going to do in homework.

## Backpropagation

**Be careful!** In this section the variable $x$ designates the parameters in NN and not the input data. Think that we fixed the data now, and loss is a functionof parametrs, we try to find the best parameters to lower the loss.

Let's define as f(x) the function NN applies to input data and g(o) is a criterion. Then

        L(x) = g(f(x); target)

L(x) is just a number. We aim to find grad(L). Obviously, if f,g:R->R are real functions, using chain rule:  

        dL/dx = dg/df * df/dx

and practical formula:

        dL/dx(x0) = dg/df(f(x0)) * df/dx(x0)

In multidimensional case barely the same. It is the sum of 1-dimensional chains.

        dL/dx_i = Sum( dg/df_j * df_j/dx_i  :  j from 1 to m )

Actually that is all you need to write backprop functions! Go to differenciation notebook to for some practice before homework.

## Structure of the framework

        ├───criterions       
        ├───modules
        ├───optimizers
        ├───tests
        ├───utils
        ├──toy_example.py
        ├──digit_classification.py
        ├──autoencoder(optiional).py
        └──experiments(optional).py

Each module of the NN is implemented in separate file in ./modules directory. In modules/module.py file the base abstract class Module is defined. This is the abstract building block of NN. All other modules are inherited from that class.

The same thing with criterions.

In optimizers folder we have two files. SGD optimizers with momentum is already implemented. Adam optimizer is left to you.

Tests are for testing as usual.

In utils folder I put all utility functions that will be frequently needed, e.g. metrics, data_generators.

## Where to start
Yes, you definitely worried where to start this weird homework. I would suggest you the following:

1. Read all the files, with only exception "tests" directory (which is optional :) read all the comments, they are the key!
2. Implement easiest parts/modules first. This way you will make less bugs and enjoy the results of your code.
  - for the working NN you need Dense layer, one activation layer (pick the easiest one) and one criterion (e.g. MSECriterion, note that you can use that for classification task as well, only for testing purposes). Build your model with these blocks in "toy_example.py" file and check the performance of the model.
  - If everything works fine and you pass tests for these modules, you can continue implement other modules.
  - implement SoftMax and CrossEntropy.
  - then implement remaining parts in folders ./modules and ./criterions. Also implement Adam, it performs better then SGD.
3. Now you are ready to do MNIST task.

This is just a suggestion.

## Technical notes:
np.multiply, np.add, np.divide, np.subtract instead of *,+,/,- for better memory handling

Suppose you allocated a variable    

    a = np.zeros(...)

So, instead of

    a = b + c  # will be reallocated, garbage collector needed to free

I would go for: 

    np.add(b,c,out = a) # puts result in `a`

But it is completely up to you.

## Finally
The unit testing command is:
```
$ python -m unittest discover -s tests
```

Good luck and enjoy the homework!

