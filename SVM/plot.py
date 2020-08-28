import json
import argparse

import numpy as np
from matplotlib import pyplot as plt

from svm import SVM
from kernel import linear_kernel, polynomial_kernel, gaussian_kernel
from seed import state

def parse_args(*argument_array):
    parser = argparse.ArgumentParser(description="SVM model presentation")
    parser.add_argument('--data-type', type=str,
                        default='blobs', choices=['blobs', 'circles'])
    parser.add_argument('--kernel', type=str,
                        default='linear', choices=['gaussian',
                                                   'polynomial',
                                                   'linear'])
    parser.add_argument('--config', type=str, default=None,
                        help='Path to json file with '
                             'kernel and data parameters')
    parser.add_argument('--c', type=float, default=None)
    args = parser.parse_args(*argument_array)

    return args


def generate_blobs(samples=50,
                   std=0.5,
                   centers=[(1, 2), (5, 6)]):
    np.random.set_state(state)
    data = []
    labels = []
    for i, c in enumerate(centers):
        _d = np.random.multivariate_normal(c, [[std, 0], [0, std]], samples)
        data.extend(_d)
        labels.extend(np.ones(_d.shape[0]) * i)
    labels = np.array(labels)
    data = np.array(data)
    labels[labels == 0] = -1
    return np.array(data), np.array(labels)


def generate_circles(samples=50,
                     std=0.7,
                     r=6,
                     center=(1, 2)):
    np.random.set_state(state)
    t = np.linspace(0, 2 * np.pi, samples)
    x = r * np.cos(t) + np.random.normal(scale=std, size=len(t)) + center[0]
    y = r * np.sin(t) + np.random.normal(scale=std, size=len(t)) + center[1]
    first_circle = np.vstack((x, y)).T

    return (np.vstack([first_circle, np.random.multivariate_normal(center,
                                                                   [[std, 0],
                                                                    [0, std]],
                                                                   samples)]),
            np.hstack([np.ones(samples), -np.ones(samples)]))


def plot(data, labels, clf=None):
    kernel = clf.kernel.__name__.replace('_', ' ') or ''
    params = ' '.join(f'{k}={v}' for k, v in clf.kernel_params.items()) or 'default'
    c = '' if clf.C is None else f', c={clf.C}'
    title = f'SVM with {kernel}' + c + ',kernel params - ' + params +'\n x - support vector points'

    plt.figure(figsize=(15, 8))
    plt.title(title)

    for d, l in zip(data, labels):
        if np.any(clf.sv_s == d):
            m = 'x'
        else:
            m = 'o'
        plt.scatter(d[0], d[1],
                    c='red' if l == 1 else 'green',
                    marker=m,
                    cmap=plt.cm.coolwarm,
                    label='x - support vector points')

    if clf is not None:
        xx, yy = np.meshgrid(
            np.arange(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1, 0.02),
            np.arange(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1, 0.02))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
    plt.show()


def main(args):
    kernels_mapping = {
        'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        'gaussian': gaussian_kernel
    }

    clf = SVM(C=args.c,
              kernel=kernels_mapping[args.kernel],
              kernel_params=args.params['kernel'])

    if args.data_type == 'blobs':
        data, labels = generate_blobs(**args.params['data'])
    else:
        data, labels = generate_circles(**args.params['data'])

    clf.fit(data, labels)
    plot(data, labels, clf)


if __name__ == '__main__':
    args = parse_args()
    args.params = {'kernel': {}, 'data': {}}

    if args.config:
        try:
            with open(args.config, 'r') as pfile:
                args.params.update(json.load(pfile))
        except Exception as e:
            print('Wrong file path or format for params.')

    main(args)
