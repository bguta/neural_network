"""Model of a simple feed forward neural network using mostly numpy.

@author Bereket Guta.
"""

import numpy as np
import math as mt
from numba import jit, vectorize, float64, boolean
import pickle
from functools import reduce


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

    def __init__(this, topology, dropout=False):
        """Create the network."""

        this.eta = 0.01  # the learning rate
        this.topology = topology  # the makeup of the network

        # if this is a classification problem or not i.e dog or cat ; this
        # determines the use of softmax
        if this.topology[-1] >= 2:
            this.isClassify = True
        else:
            this.isClassify = False

        this.dropout = dropout  # if we are going to drop neurons while training

        # this list contains a set of lists such that each
        # index corresponds to a layer of the network
        this.network = []

        # the bias for each layer in this.network that is not the output
        # bias[i] is the bias for network[i+1] i.e the layer it points to
        this.bias = []

        # create an input vector
        this.inputL = np.zeros((topology[0], 1), dtype="float64")

        this.network.append(this.inputL)

        # the output vecotr layer
        this.outputL = np.zeros((topology[-1], 1), dtype="float64")

        # set up the list of hidden layers and import them into the network
        for hl in range(1, len(topology) - 1):
            # the hidden layer vectors
            this.network.append(np.zeros((topology[hl], 1), dtype="float64"))

        this.network.append(this.outputL)  # add the output

        # for each layer that is not an input
        for i in range(1, len(this.network)):
            # add a column vector of 1.0
            this.bias.append(np.ones(this.network[i].shape, dtype="float64"))

        # this list contains a set of lists such that each index i contains the
        # weight for the (i) -> (i +1) in the list this.network
        this.weights = []
        # setupt the weight matrix randomly
        np.random.seed()  # set the seed randomly
        for i in range(1, len(this.network)):
            # random method
            """
            this.weights.append(np.random.randn(this.network[i].size,
                                                this.network[i - 1].size))
            """
            # xavier method
            this.weights.append(np.random.uniform(-4 * mt.sqrt(6 / (this.network[i].size + this.network[i - 1].size)),
                                                  4 * mt.sqrt(6 / (this.network[i].size + this.network[i - 1].size)), (this.network[i].size, this.network[i - 1].size)))
        this.probabilities = []
        for h1 in this.network[1:-1]:
            this.probabilities.append(h1)
    # propagation

    def feedForward(this, drop=True, pr=0.5, train=True):
        """
        Feed forward the sum of the previous
        layers activations multiplied by their corresponding
        weights fed into the sigmoid.

        You should not call this directly, rather call test/train

        equations:
        raw_out = Weight_matrix * prev_Output + Bias_vector
        output = activation(raw_out)


        """

        # if drop:  # drop the inputs
        #     this.inputs = np.array(this.network[0])
        #     this.network[0][:, 0] = dropout(this.network[0])[:, 0]

        # sig = np.vectorize(sigmoid)  # make the sigmoid a vector function
        # set the output for every layer other than the input
        last = len(this.network)
        for i in range(1, last):
            # the weight from the previous layer to this layer, the output of
            # the previous layer and the bias that is points from the
            # previous layer to this layer

            outp = np.array(this.network[i])

            if drop:
                if train:
                    np.dot(this.weights[i - 1], this.network[i - 1], out=outp)

                else:
                    np.dot(this.weights[i - 1] * pr,
                           this.network[i - 1], out=outp)
            else:
                np.dot(this.weights[i - 1], this.network[i - 1], out=outp)

            outp = matAdd(outp, this.bias[i - 1])
            """
            out = forwardMul(this.weights[i - 1], this.network[i - 1],
                             this.bias[i - 1])
            """
            # run the activation function

            this.network[i][:, 0] = sigmoid(outp, False)[:, 0]

            if i > 1 and drop and train and i < last:  # undrop
                this.network[i - 1][:, 0] = this.prev[:, 0]

            if i >= 1 and i < last - 1 and drop and train:  # drop
                this.prev = np.array(this.network[i])

                this.probabilities.pop(i - 1)
                this.probabilities.insert(i - 1, np.random.binomial(
                    1, pr, size=this.network[i].shape))

                this.network[i] *= this.probabilities[i - 1]

    def backPropagate(this, goal, pr=0.5, drop=True):
        """
        Stochastic gradient descent

        This sets the error for output neurons based on the goal
        it then propagates this error backwards through the network

        It changes the weights based on the gradient
        and output of the layer before it.
        it also changes the bias based on the outputs

        You should not call this directly, rather call train

        equations:
        gradient = eta * Error * deriv_prev_Output
        dWeight = gradient (dot) transpose(this_output)
        dBias = gradient
        """

        assert this.network[-1].size == len(goal), (
            "goal is not the same length as output" +
            " layer rather it is %r" % len(goal))

        this.Error = subt(goal, this.network[-1])  # get the difference

        restLayers = this.network[:-1]
        errs = []
        errs.append(this.Error)

        # sig = np.vectorize(sigmoid)  # vectorize the sigmoid

        i = len(this.weights) - 1  # start with the last layer
        last = i
        for layer in restLayers[::-1]:  # reverse the array to go backwards
            prevError = errs.pop()

            layer_error = np.array(layer)
            np.dot(this.weights[i].transpose(), prevError, out=layer_error)

            #dPrev = np.zeros(this.network[i + 1].shape)
            dPrev = sigmoid(this.network[i + 1], True)

            # calc the change in weights
            """
            gradients = np.multiply(
                Network.eta, np.multiply(
                    prevError, dPrev))
            """
            gradients = multi(this.eta, multi(prevError, dPrev))

            if drop:
                if i < last:
                    gradients *= this.probabilities[i]
                else:
                    gradients *= pr

            dWeight = np.array(this.weights[i])
            np.dot(gradients, layer.transpose(), out=dWeight)

            assert dWeight.shape == this.weights[
                i].shape, "sizes are not equal"
            # add the change to the weight
            this.weights[i] = matAdd(this.weights[i], dWeight)

            # add the change in bias
            this.bias[i] = matAdd(this.bias[i], gradients)

            errs.append(layer_error)
            i -= 1

    # setters and getters methods
    def setInput(this, inputs):
        '''
        @param inputs
        a list of numbers that correspond to the input for each input neurons.
        must have the same length as the number of neurons in the input layer
        i.e the number passed in topology

        You should not call this directly, rather call test/train
        '''
        assert len(this.network[0]) == len(inputs), (
            "input is not the same length as input"
            "layer rather it is " + str(len(inputs)))

        # set the inputs
        # assign the input layer the inputs
        # this.network[0][:, 0] = list(inputs)
        assert this.network[0].shape == inputs.shape
        this.network[0][:, 0] = inputs[:, 0]

    def getError(this, goal):
        '''

        This calculates the error by summing the difference squared of
        each neuron in the output layer with its
        goal and then taking root of the mean.

        You should not call this directly, rather call train

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
        """
        for i in range(len(goal)):
                # find the difference between the output and the goal neuron
            e = (goal[i] - this.network[-1][i])
            err += (e ** 2) / 2  # add the error squared to the sum

        err /= len(goal)
        err = mt.sqrt(err)
        """

        return float(RMS(goal, this.network[-1]).sum(axis=0))

    def getResults(this, prob=True):
        """
        This gets the results of the output layer.

        You should not call this directly, rather call test

        @returns
        A list of numbers containing the output of the
        neurons in the output layer
        """

        if this.isClassify and prob:
            return list(softmax(this.network[-1]))

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

    def train(this, inputs, output):
        """
        Trains the model by inputing the inputs and feeding forward and back propagating and
        then returns the error of this forward and backward iteration.

        @param inputs
        a list of numbers that correspond to the input for each input neurons.
        must have the same length as the number of neurons in the input layer
        i.e the number passed in topology

        @param output
        a list of numbers that represent the desired output


        @returns
        the err of the network calculated using rms of all the
        output neuron errors

        """
        g = np.array(output, dtype="float64")
        g = g.reshape(this.network[-1].shape)
        i = np.array(inputs, dtype="float64")
        i = i.reshape(this.network[0].shape)

        this.setInput(i)
        this.feedForward(drop=this.dropout)
        this.backPropagate(g, drop=this.dropout)
        return this.getError(g)

    def test(this, inputs, useSoftmax=True):
        """
        Test the model with a given input. This returns the output of the network given
        this input


        @param inputs
        a list of numbers that correspond to the input for each input neurons.
        must have the same length as the number of neurons in the input layer
        i.e the number passed in topology

        @returns
        the output of the network

        """
        i = np.array(inputs, dtype="float64")
        i = i.reshape(this.network[0].shape)
        this.setInput(i)
        this.feedForward(drop=False)
        return this.getResults()

    def save(this, name):
        """Save the model to a file.

        This saves the model's weights and biases as well its makeup i.e. how many layers/neurons it has.

        @param name
        the name of the file for this particular network without ".pkl"
        e.g "net_10_10_1"
        """
        weight_data = this.weights
        bias_data = this.bias
        topology_data = this.topology

        data = [topology_data, weight_data, bias_data]

        with open(name + ".pkl", 'wb') as file:  # open and write the file
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    def load(this, name):
        """Load and initalize this model using saved weights and biases.

        This network must have the same topology as the one saved in the file.
        This method loads a file the is saved in strictly the same format as the
        save method.

        @param name
        the name of the file for this particular network without ".pkl"
        e.g "net_10_10_1"

        @throws assert errors if the topolgies are not equal
        """
        with open(name + ".pkl", 'rb') as file:
            data = pickle.load(file)

        # assert equality of the saved data with this networks
        assert len(data[0]) == len(this.topology), (
            "this topology is not equal to saved network's topology. The saved network has topology: " + str(data[0]))
        for i, j in zip(data[0], this.topology):
            assert i == j, (
                "The topologies are not equal The saved network has topology: " + str(data[0]))

        this.weights = data[1]
        this.bias = data[2]


tar = "cpu"


@vectorize(["float64(float64, boolean)"], target=tar)
def sigmoid(x, derivitave=False):     # the activation function
    if(derivitave):
        return x * (1.0 - x)

    if(x < -709.0):  # to avoid overflow
        return 0

    if(x > 1000):    # Really big
        return 1.0

    return 1 / (1 + mt.exp(x * -1.0))


@vectorize(["float64(float64, float64)"], target=tar, nopython=True)
def multi(x, y):
    """
    mutiply element wise

    """
    return x * y


@vectorize(["float64(float64, float64)"], target=tar, nopython=True)
def matAdd(x, y):
    """
    add the vectors element wise

    """
    return x + y


@vectorize(["float64(float64, float64)"], target=tar, nopython=True)
def subt(x, y):
    """
    subtract element wise

    """
    return x - y


@vectorize(["float64(float64, float64)"], target=tar, nopython=True)
def RMS(goal, output):
    """
    Calc the RMS for a single element
    """
    return ((goal - output) ** 2) / 2


def softmax(vector, derivitave=False):
    """
    A soft max function to be used in getResults()

    @param vector
    the vector which softmax will be applied to

    @returns
    a new vector such that each element has the softmax function applied to it
    """
    # D = -1.0 * np.max(vector)
    D = 0
    # if derivitave:
    #     return np.diag(vector) - np.dot(vector, vector.T)

    bottom = reduce(lambda x, y: x + mt.exp(y + D), vector, 0.0)
    v = []
    for n in vector:
        v.append(mt.exp(n + D) / bottom)
    return v
