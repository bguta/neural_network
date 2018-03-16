import sys
sys.path.append('../')
from nn_models import model


'''
This is a test of the model of the neural network using an
xor truth table dataset

__author__ = Bereket Guta
'''


def main():
    # the makeup of the network
    makeUp = [2, 100, 1]

    # activate the network
    net = model.Network(makeUp)

    # set the learning rate and momentum to best reach a minimum
    model.Neuron.eta = 0.09
    model.Neuron.alpha = 0.001

    # the input set
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # the output for each input
    outputs = [[0], [1], [1], [0]]

    # start
    print("Training")
    while True:
        err = 0
        for i in range(len(inputs)):
            net.setInput(inputs[i])
            net.feedForward()
            net.backPropagate(outputs[i])
            err += net.getError(outputs[i])
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

        net.setInput([a, b])
        net.feedForward()
        print(str(net.getResults()))


if __name__ == '__main__':
    main()
