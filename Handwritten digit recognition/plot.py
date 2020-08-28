import argparse
import numpy as np

from matplotlib import pyplot as plt

from practical3.perceptron import Perceptron
from practical3.logistic_regression import LogisticRegression
from data_reader import read_data
from practical3.gradient_descent import stochastic_gradient_descent
from practical3.gradient_descent import minibatch_gradient_descent
from practical3.gradient_descent import batch_gradient_descent
from practical3.gradient_descent import newton_raphson_method


def parse_args(*argument_array):
    parser = argparse.ArgumentParser(description="Plotting model"
                                                 "results upon time")
    parser.add_argument('--model', default='LogisticRegression',
                        choices=['LogisticRegression', 'Perceptron'])
    parser.add_argument('--update-method', type=str,
                        choices=['stochastic_gradient_descent',
                                 'minibatch_gradient_descent',
                                 'batch_gradient_descent',
                                 'newton_raphson_method'],
                        default='minibatch_gradient_descent')
    parser.add_argument('--learning-rate', type=float, default=1e-1)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--data-path', default='data')
    parser.add_argument('--init-weights', type=bool, default=False)
    parser.add_argument('--l', type=float, default=1e-2)
    
    args = parser.parse_args(*argument_array)
    
    args.weights_file = f"weights_{args.model}.csv"
    if args.model == "Perceptron":   
        args.fig_title = f"{args.model}_{args.epochs}_epochs"
    else:
        args.fig_title = f"{args.model} with {args.update_method} ____________ Learning Rate - {args.learning_rate} Lambda - {args.l} Batch size - {args.batch_size}"
    
    if args.update_method is not None:
        args.update_method = eval(args.update_method)
    
    args.model = eval(args.model)
    args.update_params = {}
    if args.learning_rate is not None:
        args.update_params['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        args.update_params['batch_size'] = args.batch_size
    return args

def main(args):
    data, labels = read_data(args.data_path)
    mask = (labels == 4) | (labels == 9)
    data = data[mask]
    labels = (labels[mask] - 4) / 5
    if args.model is Perceptron:
        labels = labels * 2 - 1
        kwargs = {}
    else:
        kwargs = {'update_params': args.update_params}
        kwargs['l'] = args.l
        if args.update_method is not None:
            kwargs['update_method'] = args.update_method
    if args.epochs is not None:
        kwargs['epochs'] = args.epochs
    
    kwargs['weights'] = args.weights_file if args.init_weights else None
    
    

    model = args.model(**kwargs)
    
    plt.ion()
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    ln, = ax.plot([], [], c='b')
    losses = []
    xlim = 1
    
    plt.title(args.fig_title)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    
       
    
    #Spliting data to train and test for avoiding overfitting
     
    
    shuffle_index = np.random.permutation(data.shape[0])
    data_shuffled, labels_shuffled = data[shuffle_index], labels[shuffle_index]

    train_proportion = 0.8
    train_test_cut = int(len(data)*train_proportion)

    data, test_data, labels, test_labels = \
        data_shuffled[:train_test_cut], \
        data_shuffled[train_test_cut:], \
        labels_shuffled[:train_test_cut], \
        labels_shuffled[train_test_cut:]    
    
    
    for ind, w in enumerate(model.fit(data, labels)):
        #if ind > xlim:
        xlim += 1
        losses.append(model.loss(data, labels))
        ax.set_xlim([0, xlim])
        
        ax.set_ylim([min(losses) * 0.9, max(losses) * 1.1])
        ln.set_xdata(list(range(1, len(losses)+1)))
        ln.set_ydata(losses)
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        np.savetxt(args.weights_file, w,  delimiter=',')
        np.savetxt('mask.csv', model.mask,  delimiter=',')
    
    print("[LOSS] ", losses[-1])
    print("[TRAINING ACCURACY]", model.accuracy(data, labels))    
    print("[TEST ACCURACY]", model.accuracy(test_data, test_labels))

    plt.savefig(args.fig_title + ".png")
    plt.show(block=True)
    
    
    
if __name__ == '__main__':
    main(parse_args())
