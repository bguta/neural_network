import math
import numpy as np


class Connection:
    '''
    @param connectedNeuron
        the neuron that is connected
    '''
    def __init__(this, connectedNeuron):
        this.connectedNeuron = connectedNeuron
        this.weight = np.random.normal()
        this.dWeight = 0.0  # delta weight of the connection


class Neuron:
    '''
    @param layer
        is a set of other neurons connected to this neuron
        i.e the previous layer


    where
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

    def addError(this, err):  # this will sum the errors
        this.error += err

    def sigmoid(this, x):     # the activation function
        return 1 / (1 + math.exp(-x * 1.0))

    def dSigmoid(this, x):    # derivitave of the sigmoid; used for the gradient during backpropagation
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
        than it uses their output to determine this output ; if not it is an input or bias neuron
        and does not to feedforwar
        '''
        sumOutput = 0
        if len(this.connections) == 0:
            return

        # get the output of each connection while multiplying by its connection weight
        # add to the sum of this output
        for link in this.connections:
            sumOutput += link.connectedNeuron.getOutput() * link.weight

        # run the activation function over the sum of the connected outputs
        this.output = this.sigmoid(sumOutput)

    def backPropagate(this):
        '''
            This sets the gradient and loops through the previous connections of this neuron.
            For each connection it calculates the change in weight and adjusts the weight
            using the value. It finally adds the resulting error to the connected neuron.
            It resets the error for this neuron once done the loop

            formulas:
            δweight= η x gradient x output of connected neuron + α x previous δweight
            gradient = error x d/dy(output)
            error += (weight * gradient)

            where
                η = learning rate
                α = momentum rate (this will let the weights move in a certain direction avoiding fulxs)

        '''
        # calc the gradient ; this will decide the direction of the change of
        # the weight
        this.gradient = this.error *\
            this.dSigmoid(this.output)
        for link in this.connections:

            # calc the change in weight of the connection
            link.dWeight = Neuron.eta * (
                link.connectedNeuron.output * this.gradient) + (this.alpha * link.weight)
            # set the new weight using the change in weight
            link.weight += link.dWeight

            # set the error for the connected neuron based on the weight and
            # the gradient for this neuron
            link.connectedNeuron.addError(link.weight * this.gradient)

        # reset the error
        this.error = 0


class Network:
    '''
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
    '''
    def __init__(this, topology):

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
                    n = Neuron(this.layer[-1])
                    layer.append(n)

            # add a bias neuron to make a better fit
            layer.append(Neuron(None))
            layer[-1].setOutput(1.0)  # set the output of the bias to 1

            # add this layer to the list of layers
            this.layer.append(layer)

    # setter and getter functions
    def setInput(this, inputs):
        '''
        @param inputs
            a list of numbers that correspond to the input for each input neurons.
            Must have the same length as the number of neurons in the input layer i.e the number passed in topology
        '''
        for i in range(len(inputs)):
            this.layer[0][i].setOutput(inputs[i])
