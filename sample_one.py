import numpy as np
import theano
import h5py
from theano import tensor
from blocks import roles
from blocks.model import Model
from blocks.extensions import saveload
from blocks.filter import VariableFilter
from utils import MainLoop
from config import config
from model import nn_fprop
from utils import get_stream
import argparse
import sys
import os
import pandas as pd
import time
import signal
from pandas.parser import CParserError
import matplotlib.pyplot as plt
from fuel.datasets import H5PYDataset

locals().update(config)

def my_longitude(ndarray):
    return map(lambda x: (x - 2.1) / 4.0 - 73.925, ndarray)

def my_latitude(ndarray):
    return map(lambda x: (x - 2.0) / 4.0 + 40.775, ndarray)


def sample(x_curr, fprop):

    hiddens = fprop(x_curr)
    #print x_curr.shape
    #print x_curr[-1:,:,:]
    probs = hiddens.pop().astype(theano.config.floatX)
    # probs = probs[-1,-1].astype('float32')
    #print "the output shape is: "
    #print probs.shape
    #print "getting the last element..."
    #probs = probs[-1,:].astype('float32')
    #print probs
    #print probs.shape
    #print my_longitude(probs[0])
    #print my_latitude(probs[1])
    #hierarchy_values = [None] * (len(predict_funcs) + 1)
    # probs = probs / probs.sum()
    # sample = np.random.multinomial(1, probs).nonzero()[0][0]
    # print(sample)
    return probs, hiddens


if __name__ == '__main__':
    # Load config parameters
    locals().update(config)
    float_formatter = lambda x: "%.5f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    parser = argparse.ArgumentParser(
        description='Generate the learned trajectory',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    print('Loading model from {0}...'.format(save_path[network_mode]))

    x = tensor.tensor3('features', dtype=theano.config.floatX)
    y = tensor.tensor3('targets', dtype=theano.config.floatX)
    out_size = len(output_columns)
    in_size = len(input_columns)

    cost, mu, sigma, mixing, output_hiddens, mu_linear, sigma_linear, mixing_linear, cells = nn_fprop(x, y, in_size, out_size, hidden_size[network_mode], num_layers, layer_models[network_mode], 'MDN', training=False)
    main_loop = MainLoop(algorithm=None, data_stream=None, model=Model(cost),
                         extensions=[saveload.Load(save_path[network_mode])])

    for extension in main_loop.extensions:
        extension.main_loop = main_loop
    main_loop._run_extensions('before_training')
    bin_model = main_loop.model
    print 'Model loaded. Building prediction function...'
    hiddens = []
    initials = []

    for i in range(num_layers):
        brick = [b for b in bin_model.get_top_bricks() if b.name == layer_models[network_mode][i] + str(i) + '-'][0]
        hiddens.extend(VariableFilter(theano_name=brick.name + '_apply_states')(bin_model.variables))
        hiddens.extend(VariableFilter(theano_name=brick.name + '_apply_cells')(cells))
        initials.extend(VariableFilter(roles=[roles.INITIAL_STATE])(brick.parameters))

    fprop = theano.function([x], hiddens + [output_hiddens])

    #predicted = np.array([1,1], dtype=theano.config.floatX)
    #x_curr = [[predicted]]

    #print(x[2].eval())
    file = h5py.File(hdf5_file[network_mode], 'r')
    input_dataset = file['features']  # input_dataset.shape is (8928, 200, 2)
    input_dataset = np.swapaxes(input_dataset,0,1)
    print input_dataset.shape


    ############### for second network #####################
    in_size = hidden_size[network_mode]
    out_size = hidden_size[network_mode]
    nsamples = len(input_dataset[-1, :, -1])
    # inputs = np.empty((nsamples, seq_length, in_size), dtype='float32')
    inputs = np.empty((nsamples/24, 24, in_size), dtype='float32')
    outputs = np.empty((nsamples/24, 24, out_size), dtype='float32')
    print nsamples


    #sample_results = sample([x], fprop, [component_mean])
    #x_curr = [input_dataset[0,:,:]]

    for i in range(nsamples):
        x_curr = input_dataset[:,i:i+1,:]
        #print x_curr
        input_sec_network, newinitials = sample(x_curr, fprop)  # the shape of input_sec_network is (200,)
        ############  make data for the second network ########################
        #print input_helper[i].shape
        inputs[i/24, i%24, :] = input_sec_network
        #input_helper[i] = input_sec_network

        # for initial, newinitial in zip(initials, newinitials):
        #    initial.set_value(newinitial[-1].flatten())
           #print newinitial[-1].flatten()
    outputs = inputs
    print 'inputs shape:', inputs.shape
    print 'outputs shape:', outputs.shape


    f = h5py.File(hdf5_file[1], mode='w')
    features = f.create_dataset('features', inputs.shape, dtype='float32')
    targets = f.create_dataset('targets', outputs.shape, dtype='float32')
    features[...] = inputs
    targets[...] = outputs
    features.dims[0].label = 'batch'
    features.dims[1].label = 'sequence'
    targets.dims[0].label = 'batch'
    targets.dims[1].label = 'sequence'

    nsamples_train = int(372 * train_size[network_mode])
    split_dict = {
        'train': {'features': (0, nsamples_train), 'targets': (0, nsamples_train)},
        'test': {'features': (nsamples_train, 372), 'targets': (nsamples_train, 372)}}

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()
