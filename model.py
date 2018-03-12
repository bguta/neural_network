import math
import numpy as np


class Connection:
    '''
    @param connectedNeuron
    the neuron that is connected
    '''
    def __init__(this, connectedNeuron):
        this.connectedNeuron = connectedNeuron
        assert this.connectedNeuron is not None, \
            "empyt connected neuron in connection"

        this.weight = np.random.normal()
        this.dWeight = 0.0  # delta weight of the connection


class Neuron:
    '''
    @param layer
    is a set of other neurons connected to this neuron
    i.e the previous layer


    where:
    η = learning rate
    α = momentum rate
    '''
    eta = 0.001  # learning rate
    alpha = 0.01  # momenutm factor

    def __init__(this, layer):

        this.connections = []  # set of connections

        # this is to calc error of output
        this.error = 0.0
        this.gradient = 0.0

        this.output = 0.0  # the value of the n

        # make its connections
        if layer is None:
            pass  # for the input and bias neurons this will be none
        else:
            for neuron in layer:
                c = Connection(neuron)  # makes it into a connection object
                this.connections.append(c)  # add it to its connections

    # setter and getter methods
    def getError(this):
        # this is used for testing not in the model
        return this.error

    def addError(this, err):  # this will sum the errors
        this.error += err

    def sigmoid(this, x):     # the activation function

        if(x < -709.0):
            return 0.0

        if(x > 1000):
            return 1.0

        return 1 / (1 + math.exp(x * -1.0))

    def dSigmoid(this, x):
        # derivitave of the sigmoid; used for the
        # gradient during backpropagation
        return x * (1.0 - x)

    def setError(this, err):
        this.error = err

    def setOutput(this, out):
        this.output = out

    def getOutput(this):
        return this.output

    # propogation

    def feedForward(this):
        '''
        Checks if the there are any previously connected neurons; if there are
        than it uses their output to determine this output ; if not it is an
        input or bias neuron
        and does not need to feedforward i.e change its output
        '''
        sumOutput = 0
        if len(this.connections) == 0:
            return

        # get the output of each connection while multiplying by its
        # connection weight add to the sum of this output
        for link in this.connections:
            sumOutput += link.connectedNeuron.getOutput() * link.weight

        # run the activation function over the sum of the connected outputs

        this.output = this.sigmoid(sumOutput)

    def backPropagate(this):
        '''
        This sets the gradient and loops through the previous connections of
        this neuron. For each connection it calculates the change
        in weight and adjusts the weight
        using the value. It finally adds the resulting error to the connected
        neuron. It resets the error for this neuron once done the loop

        formulas:
        δweight= η x gradient x output of connected neuron
            + α x previous δweight
        gradient = error x d/dy(output)
        error += (weight * gradient)

        where:
        η = learning rate
        α = momentum rate (this will let the weights move in a
            certain direction avoiding fulxs)

        '''
        # calc the gradient ; this will decide the direction of the change of
        # the weight
        this.gradient = this.error *\
            this.dSigmoid(this.output)
        for link in this.connections:

            # calc the change in weight of the connection
            link.dWeight = Neuron.eta * (
                link.connectedNeuron.output * this.gradient) + (
                this.alpha * link.weight)
            # set the new weight using the change in weight
            link.weight += link.dWeight

            # set the error for the connected neuron based on the weight and
            # the gradient for this neuron
            link.connectedNeuron.addError(link.weight * this.gradient)

        # reset the error
        this.error = 0.0


class Network:
    """
    This creates a nn model with a given makeup.
    It initilizes each layer with a set of neurons including one bias
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
        this.layers = []  # a list of lists that contains the layers of neurons

        for neuronSet in topology:
            layer = []  # the neurons in this particular layer

            for i in range(neuronSet):

                if(len(this.layers) == 0):

                    # the input layer has no previous connections
                    n = Neuron(None)
                    layer.append(n)
                else:
                    # get the last layer as the connections for the neuron
                    n = Neuron(this.layers[-1])
                    layer.append(n)

            # add a bias neuron to make a better fit
            layer.append(Neuron(None))
            layer[-1].setOutput(1.0)  # set the output of the bias to 1

            # add this layer to the list of layers
            this.layers.append(layer)

    # setter and getter functions
    def setInput(this, inputs):
        '''
        @param inputs
        a list of numbers that correspond to the input for each input neurons.
        must have the same length as the number of neurons in the input layer
        i.e the number passed in topology
        '''
        assert len(this.layers[0]) - 1 == len(inputs), (
            "input is not the same length as input"
            "layer rather it is " + str(len(inputs)))

        for i in range(len(inputs)):
            this.layers[0][i].setOutput(inputs[i])

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
        assert len(this.layers[-1]) - 1 == len(goal), (
            "goal is not the same length as output layer" +
            " rather it is %r" % len(goal))

        for i in range(len(goal)):
                # find the difference between the output and the goal neuron
            e = (goal[i] - this.layers[-1][i].getOutput())
            err += e ** 2  # add the error squared to the sum

        err /= len(goal)
        err = math.sqrt(err)
        return err

    # propagation
    def feedForward(this):
        """
        This calls the feed forward function for each neuron
        not in the input layer.
        Depending on the size of the network this can take a long time
        """
        # set the output for every layer other than the input
        for layer in this.layers[1:]:
            for n in layer:
                n.feedForward()

    def backPropagate(this, goal):
        """
        This sets the error for output neurons based on the goal
        and calls the backpropagate function for each neuron
        looping from output to the input layer

        @param goal
        a list containing the desired outputs of the network for a given input.
        must have the same length as the number of neurons in the output layer

        """

        assert len(this.layers[-1]) - 1 == len(goal), (
            "goal is not the same length as output" +
            " layer rather it is %r" % len(goal))

        for i in range(len(goal)):
            # sets the error for each neuron in the output layer based on the
            # desired goal
            this.layers[-1][i].setError(goal[i] -
                                        this.layers[-1][i].getOutput())
        # reverses the order i.e from output to input
        for layer in this.layers[::-1]:
            for n in layer:
                n.backPropagate()

    def getResults(this):
        """
        This gets the results of the output layer

        @returns
        A list of numbers containing the output of the
        neurons in the output layer
        """
        output = []

        for n in this.layers[-1]:  # output layer

            '''
            optional if threshold is desired

            if n.getOutput > 0.5:
                output.append(1.0)
            else:
                 output.append(0.0)
            '''

            output.append(n.getOutput())

        output.pop()  # remove the bias neuron

        return output
