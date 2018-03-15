import numpy as np

"""
This is an implementation of a neural network using mostly numpy
"""
BIAS = 1.0


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

    def __init__(this, topology):
        """Create the network."""

        # this list contains a set of lists such that each
        # index corresponds to a layer of the network
        this.network = []

        # the bias for each layer in this.network that is not the output
        # bias[i] is the bias for network[i+1] i.e the layer it points to
        this.bias = []

        this.inputL = [0] * \
            (topology[0])  # the input layer plus the bias

        this.network.append(this.inputL)

        this.outputL = [0] * topology[-1]  # the output layer

        # set up the list of hidden layers and import them into the network
        for hl in range(1, len(topology) - 1):
            this.network.append([0] * (topology[hl]))

        this.network.append(this.outputL)  # add the output

        # for each layer that is not an input
        for i in range(1, len(this.network)):
            # add a column vector of 1.0
            this.bias.append([1.0] * len(this.network[i]))

        # this list contains a set of lists such that each index i contains the
        # weight for the (i) -> (i +1) in the list this.network
        this.weights = []
        # setupt the weights randomly
        for i in range(1, len(this.network)):
            this.weights.append(
                np.random.randn(
                    len(this.network[
                        i]),
                    len(this.network[i - 1])))

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
            rawOutput = list(np.add(np.dot(this.weights[i - 1],
                                           this.network[i - 1]),
                                    this.bias[i - 1]))
            # run the activation function
            this.network[i] = [sigmoid(x) for x in rawOutput]

    def backPropagate(this):
        """

        """

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

        for i in range(len(inputs)):
            this.network[0][i] == inputs[i]


def sigmoid(x, derivitave=False):     # the activation function
    if(derivitave):
        return x * (1.0 - x)

    if(x < -709.0):  # to avoid overflow
        return 0.0

    if(x > 1000):    # Really big
        return 1.0

    return 1 / (1 + np.exp(x * -1.0))
