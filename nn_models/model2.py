import numpy as np
import math as mt

"""
This is an implementation of a neural network using mostly numpy

@author Bereket Guta
"""


class Network:
    """
    This creates a nn model with a given makeup.
    It initilizes each layer with a set of neurons including one bias
    (bias is for the non output layers)
    neuron to add another dimension to the fit.

    This sets the connection for each neuron that is not a bias or an input,
    such that it is connected to all the neurons in the previous layer.

    @param topology
    a list containing the number of neurons per layer
    ex. [3,3,3] means three layers with three neurons. Each
    must contain numbers greater or equal to 1. First number is
    always the input and the last is the output
    """
    eta = 0.01

    def __init__(this, topology):
        """Create the network."""

        # this list contains a set of lists such that each
        # index corresponds to a layer of the network
        this.network = []

        # the bias for each layer in this.network that is not the output
        # bias[i] is the bias for network[i+1] i.e the layer it points to
        this.bias = []

        this.inputL = np.zeros((topology[0], 1))  # create an input vector

        this.network.append(this.inputL)

        this.outputL = np.zeros((topology[-1], 1))  # the output vecotr layer

        # set up the list of hidden layers and import them into the network
        for hl in range(1, len(topology) - 1):
            # the hidden layer vectors
            this.network.append(np.zeros((topology[hl], 1)))

        this.network.append(this.outputL)  # add the output

        # for each layer that is not an input
        for i in range(1, len(this.network)):
            # add a column vector of 1.0
            this.bias.append(np.ones((this.network[i].size, 1)))

        # this list contains a set of lists such that each index i contains the
        # weight for the (i) -> (i +1) in the list this.network
        this.weights = []
        # setupt the weight matrix randomly
        for i in range(1, len(this.network)):
            this.weights.append(np.random.randn(this.network[i].size,
                                                this.network[i - 1].size))

    # propagation
    def feedForward(this):
        """
        Feed forward the sum of the previous
        layers activations multiplied by their corresponding
        weights fed into the sigmoid.

        equations:
        raw_out = Weight_matrix * prev_Output + Bias_vector
        output = activation(raw_out)


        """
        # set the output for every layer other than the input
        for i in range(1, len(this.network)):
            # the weight from the previous layer to this layer, the output of
            # the previous layer and the bias that is points from the
            # previous layer to this layer
            """

            print(i)
            print(str(this.network[i - 1].shape))
            """
            out = np.add(np.dot(this.weights[i - 1], this.network[i - 1]),
                         this.bias[i - 1])
            # run the activation function

            this.network[i][:, 0] = [sigmoid(x) for x in out]

    def backPropagate(this, goal):
        """
        Stochastic gradient descent

        This sets the error for output neurons based on the goal
        it then propagates this error backwards through the network

        It changes the weights based on the gradient
        and output of the layer before it.
        it also changes the bias based on the outputs

        equations:
        gradient = eta * Error * prev_dOutput
        dWeight = gradient (dot) transpose(prev_output)
        dBias = gradient
        """
        g = np.zeros(this.network[-1].shape)
        g[:, 0] = goal

        assert this.network[-1].size == len(goal), (
            "goal is not the same length as output" +
            " layer rather it is %r" % len(goal))

        this.Error = np.subtract(g, this.network[-1])  # get the difference

        restLayers = this.network[:-1]
        errs = []
        errs.append(this.Error)

        i = len(this.weights) - 1  # start with the last layer
        for layer in restLayers[::-1]:  # reverse the array to go backwards
            prevError = errs.pop()

            layer_error = np.dot(np.transpose(this.weights[i]), prevError)

            dPrev = np.zeros(this.network[i + 1].shape)
            dPrev[:, 0] = [sigmoid(x, derivitave=True)
                           for x in this.network[i + 1]]

            # calc the cahnge in weights
            gradients = np.multiply(
                Network.eta, np.multiply(
                    prevError, dPrev))

            dWeight = np.dot(gradients, np.transpose(layer))

            assert dWeight.shape == this.weights[
                i].shape, "sizes are not equal"
            # add the change to the weight
            this.weights[i] = np.add(this.weights[i], dWeight)

            # add the change in bias
            this.bias[i] = np.add(this.bias[i], gradients)

            errs.append(layer_error)
            i -= 1

    # setters and getters methods
    def setInput(this, inputs):
        '''
        @param inputs
        a list of numbers that correspond to the input for each input neurons.
        must have the same length as the number of neurons in the input layer
        i.e the number passed in topology
        '''
        assert len(this.network[0]) == len(inputs), (
            "input is not the same length as input"
            "layer rather it is " + str(len(inputs)))

        # set the inputs
        this.network[0][:, 0] = inputs  # assign the input layer the inputs

    def getError(this, goal):
        '''

        This calculates the error by summing the difference squared of
        each neuron in the output layer with its
        goal and then taking root of the mean.

        @param goal
        a list containing the desired outputs of the network for a given input.
        must have the same length as the number of neurons in the output layer

        @returns
        the err of the network calculated using rms of all the
        output neuron errors
        '''
        err = 0
        assert len(this.network[-1]) == len(goal), (
            "goal is not the same length as output layer" +
            " rather it is %r" % len(goal))

        for i in range(len(goal)):
                # find the difference between the output and the goal neuron
            e = (goal[i] - this.network[-1][i])
            err += e ** 2  # add the error squared to the sum

        err /= len(goal)
        err = mt.sqrt(err)
        return err

    def getResults(this):
        """
        This gets the results of the output layer

        @returns
        A list of numbers containing the output of the
        neurons in the output layer
        """
        output = []

        for n in this.network[-1]:  # output layer

            '''
            optional if threshold is desired

            if n > 0.5:
                output.append(1.0)
            else:
                 output.append(0.0)
            '''

            output.append(n)

        return list(output)


def sigmoid(x, derivitave=False):     # the activation function
    if(derivitave):
        return x * (1.0 - x)

    if(x < -709.0):  # to avoid overflow
        return 0.0

    if(x > 1000):    # Really big
        return 1.0

    return 1 / (1 + np.exp(x * -1.0))
