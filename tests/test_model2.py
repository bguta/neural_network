

import pytest
import numpy as np
from nn_models import model2 as m2


class Test:
    """ Test the model."""

    def test_createNet(this):
        topology = [2, 2, 1]  # 3 layers
        this.net = m2.Network(topology)

        # OR truth table
        inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
        outputs = [[1], [1], [1], [0]]

        # train
        this.net.train(inputs[0], outputs[0])
        for i, j in zip(list(this.net.network[0]), inputs[0]):
            assert i == j
