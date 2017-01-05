import numpy as np
import theano
import h5py
from theano import tensor
from blocks import roles
from blocks.model import Model
from blocks.extensions import saveload
from blocks.filter import VariableFilter
from utils import get_stream, track_best, MainLoop
from config import config
from model import nn_fprop
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import argparse
from sklearn import mixture

locals().update(config)

def my_longitude(ndarray):
    return map(lambda x: (x - 2.1) / 4.0 - 73.925, ndarray)

def my_latitude(ndarray):
    return map(lambda x: (x - 2.0) / 4.0 + 40.775, ndarray)

def sample(x_curr, fprop):

    hiddens = fprop(x_curr)
    probs = hiddens.pop().astype(theano.config.floatX)

    return probs

def gaussian_2d(x, y, x0, y0, sig):
    return 1/(2*np.pi*sig) * np.exp(-0.5*((x-x0)**2 + (y-y0)**2) / sig)


def gengmm(nc):
    g = mixture.GMM(n_components=nc)  # number of components

    return g


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

    file = h5py.File(hdf5_file[network_mode], 'r')

    x = tensor.tensor3('features', dtype=theano.config.floatX)
    y = tensor.tensor3('targets', dtype=theano.config.floatX)
    x = x.swapaxes(0,1)
    y = y.swapaxes(0,1)

    cost, mu, sigma, mixing, output_hiddens, mu_linear, sigma_linear, mixing_linear, cells = nn_fprop(x, y, in_size[network_mode], out_size[network_mode], hidden_size[network_mode], num_layers, layer_models[network_mode][0], 'MDN', training=False)
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

    fprop_mu = theano.function([x], [mu])
    fprop_sigma = theano.function([x], [sigma])
    fprop_mixing = theano.function([x], [mixing])


    input_dataset = file['features']  # input_dataset.shape is (8928, 200, 2)
    input_dataset = np.swapaxes(input_dataset,0,1)
    print input_dataset.shape

    output_mu = np.empty((200, 2, components_size[network_mode]), dtype='float32')
    output_sigma = np.empty((200, components_size[network_mode]), dtype='float32')
    output_mixing = np.empty((200, components_size[network_mode]), dtype='float32')

    for i in range(20):
        x_curr = input_dataset[:,i:i+1,:]
        print x_curr.shape
        mu_ = sample(x_curr, fprop_mu)  # the shape of input_sec_network is (200,)
        sigma_ = sample(x_curr, fprop_sigma)
        mixing_ = sample(x_curr, fprop_mixing)
        ############  make data for the second network ########################
        mu_ = mu_[-1]

        output_mu[i, 0, :] = my_longitude(mu_[0])
        output_mu[i, 1, :] = my_latitude(mu_[1])
        output_sigma[i, :] = sigma_[-1]
        output_mixing[i, :] = mixing_[-1]


    index = 13
    g = gengmm(600)
    g.means_=output_mu[index,:,:]
    g.weights_=output_mixing[index,:]
    g.covariances_ = output_sigma[index,:]

    delta = 0.01
    x = np.arange(-74.2, -73.65, delta)
    y = np.arange(40.55, 41.0, delta)
    X, Y = np.meshgrid(x, y)

    i=0
    Z2= g.weights_[i]*gaussian_2d(X, Y, g.means_[0, i], g.means_[1, i], g.covariances_[i])

    for i in xrange(1,600):
        Z2 = Z2+ g.weights_[i]*gaussian_2d(X, Y, g.means_[0, i], g.means_[1, i], g.covariances_[i])
    plt.contour(X, Y, Z2)

    plt.title('Gaussian Mixture Model')
    plt.show()
