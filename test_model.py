import model
import pytest


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

        this.n = model.Neuron(this.layers[0])

    def test_Connection(this):
        """Test for the connection class"""
