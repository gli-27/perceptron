#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
#TODO: understand that you should not need any other imports other than those already in this file; if you import something that is not installed by default on the csug machines, your code will crash and you will lose points

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "./adult/" #TODO: if you are working somewhere other than the csug server, change this to the directory where a7a.train, a7a.dev, and a7a.test are on your machine

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray(ys), np.asarray(xs) #returns a tuple, first is an array of labels, second is an array of feature vectors

def perceptron(train_ys, train_xs, dev_ys, dev_xs, args):
    weights = np.zeros(NUM_FEATURES)
    #TODO: implement perceptron algorithm here, respecting args
    error = True
    dev_accuracy = 0.0
    dev_acc = [0.0, 0.0, 0.0, 0.0, 0.0]
    devmode_weights = [np.zeros(NUM_FEATURES) * 5]
    times = 0
    while args.iterations != 0 and error:
        error = False
        for i in range(0, train_ys.size):
            result_yi = np.sign(np.dot(weights, train_xs[i]))
            if result_yi != train_ys[i]:
                error = True
                weights = weights + args.lr*train_ys[i]*train_xs[i]
        args.iterations -= 1
        if not args.nodev:
            accuracy = test_accuracy(weights, dev_ys, dev_xs)
            if args.devmode == 0:
                if accuracy<dev_accuracy:
                    break
                else:
                    dev_accuracy = accuracy
            elif args.devmode == 1:
                if times == 10:
                    weights = devmode_weights[-1]
                    break
                elif accuracy >= dev_acc[-1] and times < 10:
                    times = 0
                    dev_acc.pop(0)
                    devmode_weights.pop(0)
                    dev_acc.append(accuracy)
                    devmode_weights.append(weights)
                else:
                    times += 1
    return weights

def perceptron_plot(train_ys, train_xs, iteration, lr):
    weights = np.zeros(NUM_FEATURES)
    #TODO: implement perceptron algorithm here, respecting args
    while iteration > 0:
        for i in range(0, train_ys.size):
            result_yi = np.sign(np.dot(weights, train_xs[i]))
            if result_yi != train_ys[i]:
                weights = weights + lr*train_ys[i]*train_xs[i]
        iteration -= 1
    return weights

def plot(train_ys, train_xs, dev_ys, dev_xs, test_ys, test_xs, args):
    dev_acc1 = []
    test_acc1 = []
    dev_acc2 = []
    test_acc2 = []
    for i in range(1, 101):
        iteration = i
        weight_lr1 = perceptron_plot(train_ys, train_xs, iteration, 1)
        weight_lr2 = perceptron_plot(train_ys, train_xs, iteration, 0.2)
        dev_acc1.append(test_accuracy(weight_lr1, dev_ys, dev_xs))
        test_acc1.append(test_accuracy(weight_lr1, test_ys, test_xs))
        dev_acc2.append(test_accuracy(weight_lr2, dev_ys, dev_xs))
        test_acc2.append(test_accuracy(weight_lr2, test_ys, test_xs))
    x = list(range(1, 101))
    plt.subplot(1, 2, 1)
    plt.plot(x, test_acc1, color='blue')
    plt.plot(x, dev_acc1, marker="*", linewidth=3, linestyle="--", color='green')
    plt.title("Accuracy-Iterations, lr = 1")
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.legend(["test_acc (lr = 1)", "dev_acc (lr = 1)", "test_acc (lr = 0.2)", "dev_acc (lr = 0.2)"])

    plt.subplot(1, 2, 2)
    plt.plot(x, test_acc2, color='blue')
    plt.plot(x, dev_acc2, marker="*", linewidth=3, linestyle="--", color='green')
    plt.title("Accuracy-Iterations, lr = 0.2")
    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.legend(["test_acc (lr = 1)", "dev_acc (lr = 1)", "test_acc (lr = 0.2)", "dev_acc (lr = 0.2)"])
    plt.show()

def test_accuracy(weights, test_ys, test_xs):
    accuracy = 0.0
    #TODO: implement accuracy computation of given weight vector on the test data (i.e. how many test data points are classified correctly by the weight vector)
    c = 0
    for i in range(0, test_ys.size):
        result_ys = np.sign(np.dot(weights, test_xs[i]))
        if result_ys == test_ys[i]:
            c += 1
    accuracy = c/test_ys.size
    return accuracy

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate to use for update in training loop.')
    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')
    parser.add_argument('--plot', action='store_true', default=False, help='if provided, plot analysis will be processed')
    parser.add_argument('--devmode', type=int, default=0, help='choose the mode of the using of dev data, --devmode 0 means simple use, when accuracy decrese then stop, --devmode 1 means no more five next iteration increase the accuracy, then stop.')
    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.train_file: str; file name for training data.
    args.dev_file: str; file name for development data.
    args.test_file: str; file name for test data.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs = parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)
    weights = perceptron(train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(weights, test_ys, test_xs)
    print(train_ys)
    print('Test accuracy: {}'.format(accuracy))
    print('Feature weights (bias last): {}'.format(' '.join(map(str, weights))))

    if args.plot:
        plot(train_ys, train_xs, dev_ys, dev_xs, test_ys, test_xs, args)

if __name__ == '__main__':
    main()
