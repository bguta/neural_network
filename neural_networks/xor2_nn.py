import sys
import numpy as np
sys.path.append('../')
from nn_models import model2 as m2  # noqa


'''
This is a test of the model of the neural network using an
xor truth table dataset

__author__ = Bereket Guta
'''


def main():
    # the makeup of the network
    makeUp = [2, 30, 10, 10, 1]

    # activate the network
    net = m2.Network(makeUp)

    # set the learning rate and momentum to best reach a minimum

    # the input set
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # the output for each input
    outputs = [[0], [1], [1], [0]]

    # start
    print("Training")
    while True:
        err = 0
        for i in range(len(inputs)):
            vec = np.array(inputs[i])
            vec = vec.reshape(len(inputs[i]), 1)
            out = np.array(outputs[i])
            out = out.reshape(len(outputs[i]), 1)

            err += net.train(vec, out)
        print("error: " + str(err))
        if err <= 0.01:
            break

    # testing it out with inputs
    while True:
        q = input("Ready:")
        if q == "q":
            break
        a = float(input("type 1st input: "))
        b = float(input("type 2nd input: "))

        print(str(net.test([a, b])))


if __name__ == '__main__':
    main()
