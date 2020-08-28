import argparse
import time

import numpy as np
from matplotlib import pyplot as plt

from perceptron import Perceptron
from logistic_regression import LogisticRegression
from data_generator import generate_data
from gradient_descent import stochastic_gradient_descent
from gradient_descent import minibatch_gradient_descent
from gradient_descent import batch_gradient_descent
from gradient_descent import newton_raphson_method


def parse_args(*argument_array):
    parser = argparse.ArgumentParser(description="Plotting model"
                                                 "results upon time")
    parser.add_argument('--model', default='LogisticRegression',
                        choices=['LogisticRegression', 'Perceptron'])
    parser.add_argument('--sleep-time', default=0.1, type=float)
    parser.add_argument('--update-method', type=str,
                        choices=['stochastic_gradient_descent',
                                 'minibatch_gradient_descent',
                                 'batch_gradient_descent',
                                 'newton_raphson_method'],
                        default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-datapoints', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args(*argument_array)
    if args.update_method is not None:
        args.update_method = eval(args.update_method)
    args.model = eval(args.model)
    args.update_params = {}
    if args.learning_rate is not None:
        args.update_params['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        args.update_method['batch_size'] = args.batch_size
    return args

def getline(x1, y1, x2, y2, x):
    return (x - x1) / (x2 - x1) * (y2 - y1) + y1

def plotw(ln, w, xlims, ylims):
    x1, y1 = 0, -w[0] / w[2]
    x2, y2 = -w[0] / w[1] , 0
    yleft = getline(x1, y1, x2, y2, xlims[0])
    yright = getline(x1, y1, x2, y2, xlims[1])
    xleft = getline(y1, x1, y2, x2, ylims[0])
    xright = getline(y1, x1, y2, x2, ylims[1])
    xs = []
    ys = []
    if xlims[0] <= xleft <= xlims[1]:
        xs.append(xleft)
        ys.append(ylims[0])
    if xlims[0] <= xright <= xlims[1]:
        xs.append(xright)
        ys.append(ylims[1])
    if ylims[0] <= yleft <= ylims[1]:
        ys.append(yleft)
        xs.append(xlims[0])
    if ylims[0] <= yright <= ylims[1]:
        ys.append(yright)
        xs.append(xlims[1])
    ln.set_xdata(xs)
    ln.set_ydata(ys)

def main(args):
    blues, reds = generate_data(args.num_datapoints)
    data = np.vstack([blues, reds])
    data = np.array([[1] + list(i) for i in data])
    labels = np.array([0] * len(blues) + [1] * len(reds))
    if args.model is Perceptron:
        labels = labels * 2 - 1
    inds = np.arange(len(labels))
    np.random.shuffle(inds)
    labels, data = labels[inds], data[inds]

    plt.ion()
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    if args.model is Perceptron:
        kwargs = {}
    else:
        kwargs = {'update_params': args.update_params}
        if args.update_method is not None:
            kwargs['update_method'] = args.update_method
    if args.epochs is not None:
        kwargs['epochs'] = args.epochs
    model = args.model(data.shape[1], **kwargs)
    ax.scatter(*blues.T, c='blue')
    ax.scatter(*reds.T, c='red')
    ln, = plt.plot([], [], c='black')
    lmaxx = np.max(data[:, 1])
    lmaxy = np.max(data[:, 2])
    lminx = np.min(data[:, 1])
    lminy = np.min(data[:, 2])
    plt.xlim([lminx - 1, lmaxx + 1])
    plt.ylim([lminy - 1, lmaxy + 1])
    losses = []
    for w in model.fit(data, labels):
        losses.append(model.loss(data, labels))
        plotw(ln, w, [lminx - 1, lmaxx + 1], [lminy - 1, lmaxy + 1])
        fig.canvas.draw()
        time.sleep(args.sleep_time)
        if args.model is LogisticRegression:
            xs = np.linspace(lminx - 1, lmaxx + 1, 100)
            ys = np.linspace(lminx - 1, lmaxx + 1, 100)
            X, Y = np.meshgrid(xs, ys)
            Z = np.array([model.calculate_probabilities(
                              np.array([np.ones(100), X_, Y_]).T)
                          for X_, Y_ in zip(X, Y)])
            plt.pcolormesh(X, Y, Z, shading='gouraud', cmap=plt.cm.seismic,
                           vmin=0, vmax=1, zorder=0)

    plt.show(block=True)
    plt.plot(losses)
    plt.show(block=True)

if __name__ == '__main__':
    main(parse_args())
