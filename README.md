[![Build Status](https://travis-ci.org/bguta/neural_network.svg?branch=master)](https://travis-ci.org/bguta/neural_network)
# Neural Network

This is a simple model of a neural network. 

I started with a model that mostly consisted of classes of objects that 
represented neurons, connections as well as a Network. This model worked with small scale networks but failed with
larger number of layers/neurons. 

Model 2 uses a less objects and more matrix calculations with the aid of the numpy package. This model works better
on both small and larger number of layers when compared to model 1. However, it does still tend to run slow when there
is a large number of neurons in a layer (~800). May need to consider using multi threading or a faster algorithm.

## Prereqs

Python 3.6 ; use pipenv to get the required packages

to install pipenv with pip on the command line use

```
pip install pipenv
```

## Getting started

On Github, click on the green button that says "Clone or download", then "Download ZIP".

Extract the zipped/compressed folder and open the resulting folder.

Run xor_nn.py to see an example or create your own code using the xor network as a template

## Testing

Currently testing the models. Model 2 is currently the best, but I still need to check it throughly for bugs


