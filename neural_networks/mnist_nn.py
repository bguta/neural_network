import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
import matplotlib.animation as animate
import random
import sys
import math as mt
from functools import reduce
sys.path.insert(0, "../")
from nn_models import model2 as md
from data import formatImage
import time


inputSize = 28 * 28  # the pixels space
outputSize = 10  # the number of choices for objects

answers = [
    [0] * 0 + [1] + [0] * (outputSize - 1),
    [0] * 1 + [1] + [0] * (outputSize - 2),
    [0] * 2 + [1] + [0] * (outputSize - 3),
    [0] * 3 + [1] + [0] * (outputSize - 4),
    [0] * 4 + [1] + [0] * (outputSize - 5),
    [0] * 5 + [1] + [0] * (outputSize - 6),
    [0] * 6 + [1] + [0] * (outputSize - 7),
    [0] * 7 + [1] + [0] * (outputSize - 8),
    [0] * 8 + [1] + [0] * (outputSize - 9),
    [0] * 9 + [1] + [0] * (outputSize - 10)

]

"""
this is the data, download it from
https://makeyourownneuralnetwork.blogspot.ca/2015/03/the-mnist-dataset-of-handwitten-digits.html

"""
dataset = [
    [
        "../data/mnist/mnist_train.csv",
    ],
    [
        "../data/mnist/mnist_test.csv",
    ]
]


def makeData():
    """Make the data and return a dict that contains the data and goal."""
    print("Importing Data...")

    data = []
    goal = []

    with open(dataset[0][0], "r") as file:
        info = file.readlines()
        random.shuffle(info)

        for line in info:
            vec = line.split(",")
            goal.append(answers[int(vec[0])])
            vec = np.asfarray(vec[1:]).reshape(inputSize, 1)
            data.append(vec)

    test_data = []
    test_goal = []

    with open(dataset[1][0], "r") as file:
        info = file.readlines()
        random.shuffle(info)

        for line in info:
            vec = line.split(",")
            test_goal.append(answers[int(vec[0])])
            vec = np.asfarray(vec[1:]).reshape(inputSize, 1)
            test_data.append(vec)

    return {"data": data, "goal": goal, "tData": test_data, "tGoal": test_goal}

lr = 0.001
eps = 50


def main():
    """ Make the mnist neural net."""
    trainingSet = makeData()

    composition = [inputSize, 500, 10,
                   outputSize]  # the network composition

    nn = md.Network(composition)
    print("Learning rate: " + str(lr))
    nn.eta = lr
    epcs = eps

    test(trainingSet["tData"], trainingSet["tGoal"], nn)

    train(trainingSet["data"], trainingSet["goal"], nn,
          numEpochs=epcs, plot=False, t=trainingSet)

    test(trainingSet["tData"], trainingSet["tGoal"], nn)

    while True:
        name = input(
            "please enter the name of the file to save to (ex. net1) or enter q to exit: ")

        if name == "q":
            break

        nn.save(name)
        print("SAVED")
        break


def train(data, goal, net, numEpochs=100, plot=True, t=None):
    print("Starting to train...")

    prevE = 0
    i = 0

    # plt.ion()  # start the graph
    """
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    fig.suptitle('Error Plot')
    plt.xlabel('Epoch Number')
    plt.ylabel('Error')
    """
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        fig.set_size_inches(10, 10)
        ax1.set_title("RMS Error")
        ax1.set(ylabel="Error")
        ax2.set_title("Learning Rate")
        ax2.set(ylabel="Rate")
        plt.xlabel("Epoch number")
        plt.tight_layout()
        animate.FuncAnimation(fig, addPoint)  # animate the function

    xs, y1 = [], []  # the points
    y2 = []
    decay = net.eta / numEpochs
    load = len(data)
    for epochs in range(numEpochs):
        xs.append(epochs + 1)

        err = 0  # the incured error

        te_in = time.time()

        print("Epoch (" + str(epochs + 1) + "/" + str(numEpochs) + ")")

        for j in range(load):

            prograssBar(j + 1, load)
            err += net.train(data[j][:], goal[j])

        print("\n")
        dE = err - prevE  # change in error
        print("Time to go through epoch #" + str(i + 1) +
              " (sec): " + str(time.time() - te_in))

        if plot:
            y1.append(err)  # the y point
            y2.append(net.eta)
            addPoint(xs, y1, ax1, shape="-")
            addPoint(xs, y2, ax2, colour="b")
            plt.draw()
            plt.pause(0.0001)

        print("Learning rate: " + str(net.eta))
        print("Error: " + str(err))
        print("Change in error: " + str(dE) + "\n")

        if err <= 200:
            break

        if not (dE <= 0.001 and dE >= -0.001):
            if err >= 200:
                searchThanConv(net, epochs)
        if t is not None:
            test(t["tData"], t["tGoal"], net)  # test the net

        prevE = err
        i += 1


def test(data, goal, net):
    """ Test the model. """
    correct = 0
    print("Testing...")
    for i in range(len(data)):

        classification = np.argmax(net.test(data[i][:]))
        answer = np.argmax(goal[i])

        if(answer == classification):
            correct += 1

    print("Score: " + str(correct * 100.0 / len(data)) + " %\n")
    return correct * 100.0 / len(data)


def addPoint(xs, ys, axis, colour="r", shape="o"):
    """
    animate the plot of the error

    @param xs a list of x points
    @param ys a list of y points
    @param axis i.e suplot
    """
    axis.plot(xs, ys, colour + shape)
    return True


def prograssBar(val, final):
    """
    Show the prograss.

    @param val
    the current value of the prograss (you should increase this yourself)

    @param final
    the final goal
    """
    maxlen = 50
    step = final // maxlen

    print("\r[ " + "#" * (val // step) + " ] " +
          str(int(val * 100.0 / final)) + "% ", end="")


def searchThanConv(net, epoch, eta=lr, searchE=int(eps * 0.8), alpha=10):
    """
    search then converge- (STC) learning rate
    schedules (Darken and Moody, 1990b, 1991)

    This is constant at eta for a given number of epochs (searchE)
    than it begins to decrease

    @param alpha - a constatnt
    @pararm eta - the original learning rate
    @param searchE - the number of epochs to maintain eta for
    @param net - the neural net
    @param epoch - the epoch number

    visit here for more info: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.2884&rep=rep1&type=pdf
    """
    net.eta = eta * (1 + (alpha / eta) * (epoch / searchE)) / \
        (1 + (alpha / eta) * (epoch / searchE) + (epoch**2 / searchE))


if __name__ == "__main__":
    main()
