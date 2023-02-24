import numpy as np
# import h5py
from keras.models import load_model
from NNet.python.nnet import *
from NNet.utils.writeNNet import writeNNet


def convert(kerasFile, inputMins, inputMaxes):

    model = load_model(kerasFile)

    # Get a list of the model weights
    model_params = model.get_weights()

    # Split the network parameters into weights and biases, assuming they alternate
    weights = model_params[0:len(model_params):2]
    biases = model_params[1:len(model_params):2]

    # Transpose weight matrices
    weights = [w.T for w in weights]

    # Mean and range values for normalizing the inputs and outputs. All outputs are normalized with the same value
    means = np.zeros(shape=(1+len(inputMins)))
    ranges = np.ones(shape=(1+len(inputMins)))

    # Tensorflow pb file to convert to .nnet file
    nnetFile = kerasFile[:-2] + 'nnet'

    # Convert the file
    writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, nnetFile)
