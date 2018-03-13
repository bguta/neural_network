import model
import pytest
import math


class TestModel:
    """Test for the model of a neural network."""

    def test_Input_bias_Neuron(this):
        """Test for the bias/input neuron functionality."""
        this.neru = model.Neuron(None)  # an input neuron

        # output tests
        this.neru.setOutput(1.0)
        assert this.neru.getOutput() == 1.0

        this.neru.setOutput(10000000000)
        assert (this.neru.getOutput() == 10000000000)

        this.neru.setOutput(-100000000000)
        assert (this.neru.getOutput() == -100000000000)

        # error sum tests
        assert (this.neru.getError() == 0.0)

        this.neru.setError(5.0)
        assert this.neru.getError() == 5.0

        this.neru.addError(5.0)
        assert this.neru.getError() == 10.0

        this.neru.addError(-5.0)
        assert this.neru.getError() == 5.0
        assert (this.neru.getError() != 11.0)

        # activation function tests

        assert (this.neru.sigmoid(1) - 0.73105857863) <= 0.0001
        assert (this.neru.sigmoid(-1000) == 0)
        assert (this.neru.sigmoid(1000) - 1.0 <= 0.0001)

        assert (this.neru.dSigmoid(0) == 0)
        assert this.neru.dSigmoid(1000.0) == 1000.0 * (1.0 - 1000.0)

        # feed forward test
        this.neru.setOutput(1.0)
        this.neru.feedForward()
        assert (this.neru.getOutput() == 1.0)

        # back propagate
        this.neru.backPropagate()
        assert (this.neru.getError() == 0.0)

    def test_hiddenNeuron(this):
        """Test the hidden layer neuron."""

        this.layers = []
        this.neru = model.Neuron(None)  # the inpt neru
        # the input layer
        this.prevLayer = [this.neru]
        this.layers.append(this.prevLayer)

        this.n = model.Neuron(this.layers[0])  # hidden neuron

        # output tests
        this.n.setOutput(-1000)
        assert this.n.getOutput() == -1000

        this.n.setOutput(0)
        assert (this.n.getOutput() == 0)

        this.n.setOutput(1)
        assert (this.n.getOutput() == 1)

        # error sum tests
        assert (this.n.getError() == 0.0)

        this.n.setError(-1.0)
        assert this.n.getError() == -1.0

        this.n.addError(5.0)
        assert this.n.getError() == 4.0

        this.n.addError(-5.0)
        assert this.n.getError() == -1.0
        assert (this.n.getError() != 100000.0)

        # activation function tests

        assert (this.n.sigmoid(1) - 0.73105857863 <= 0.0001)
        assert (this.n.sigmoid(-1000) == 0)
        assert (this.n.sigmoid(1000) - 1.0 <= 0.0001)

        assert (this.n.dSigmoid(0) == 0)
        assert this.n.dSigmoid(1000.0) == 1000.0 * (1.0 - 1000.0)

        # feed forward test
        this.n.setOutput(1.0)
        this.n.feedForward()
        assert (this.n.getOutput() == this.n.sigmoid(0))

        # back propagate
        this.n.setError(0.0)
        this.n.backPropagate()
        assert (this.n.getError() == 0.0)
        assert (this.n.connections[0].connectedNeuron.getError() == 0)

    def test_Connection(this):
        """Test for the connection class."""
        this.n1 = model.Neuron(None)
        this.n2 = model.Neuron(None)

        # one way connection 1 ---> 2
        this.n1.connections.append(model.Connection(this.n2))
        assert this.n2 == this.n1.connections[0].connectedNeuron
        with pytest.raises(AssertionError):
            this.n1.connections.append(model.Connection(None))

    def test_Net(this):
        """ Test for the functionality of the network"""

        # a simple AND neural net
        this.AND_nn = model.Network([2, 1, 1])
        this.input = [[1, 1], [1, 0], [0, 1], [0, 0]]
        this.output = [[1], [0], [0], [0]]

        for j in range(len(this.input)):
            this.AND_nn.setInput(this.input[j])  # set the input
            for i in range(len(this.input[j])):
                assert this.AND_nn.layers[0][i].getOutput() == this.input[j][i]
