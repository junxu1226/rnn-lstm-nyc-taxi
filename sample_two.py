import numpy as np
import theano
import h5py
from theano import tensor
from blocks import roles, initialization
from blocks.initialization import Constant, Uniform
from blocks.model import Model
from blocks.extensions import saveload
from blocks.filter import VariableFilter
from blocks.bricks import Linear, Rectifier, cost
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
import csv
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

locals().update(config)

def my_longitude(ndarray):
    return map(lambda x: (x - 2.1) / 4.0 - 73.925, ndarray)

def my_latitude(ndarray):
    return map(lambda x: (x - 2.0) / 4.0 + 40.775, ndarray)

def initialize(to_init):
    for bricks in to_init:
        bricks.weights_init = initialization.Uniform(width=0.08)
        bricks.biases_init = initialization.Constant(0)
        bricks.initialize()


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

    # loading the first model
    network_mode = 0

    print('Loading model from {0}...'.format(save_path[network_mode]))
    # file = h5py.File(hdf5_file[network_mode], 'r')

    x = tensor.tensor3('features', dtype=theano.config.floatX)
    y = tensor.tensor3('targets', dtype=theano.config.floatX)
    x = x.swapaxes(0,1)
    y = y.swapaxes(0,1)

    cost, mu, sigma, mixing, output_hiddens, mu_linear, sigma_linear, mixing_linear, cells = nn_fprop(x, y, in_size[0], out_size[0], hidden_size[0], len(layer_models[0]), layer_models[0], 'MDN', training=False)
    main_loop = MainLoop(algorithm=None, data_stream=None, model=Model(cost),
                         extensions=[saveload.Load(save_path[0])])

    for extension in main_loop.extensions:
        extension.main_loop = main_loop
    main_loop._run_extensions('before_training')
    bin_model_one = main_loop.model
    print 'first model loaded. Building prediction function...'
    hiddens = []
    initials = []


    for i in range(len(layer_models[0])):
        brick = [b for b in bin_model_one.get_top_bricks() if b.name == layer_models[0][i] + str(i) + '-'][0]
        hiddens.extend(VariableFilter(theano_name=brick.name + '_apply_states')(bin_model_one.variables))
        hiddens.extend(VariableFilter(theano_name=brick.name + '_apply_cells')(cells))
        initials.extend(VariableFilter(roles=[roles.INITIAL_STATE])(brick.parameters))

    fprop_x_hidden = theano.function([x], [output_hiddens])
    #fprop_x_sigma = theano.function([x], hiddens + [sigma])
    #mu_linear = Linear(name='mu_linear' + '-', input_dim=hidden_size[0], output_dim=2 * components_size[0])
    # sigma_linear = Linear(name = 'sigma_linear' + '-', input_dim = hidden_size[0], output_dim = components_size[0])
    # initialize([sigma_linear])
    mu = mu_linear.apply(output_hiddens)[-1,:,:]
    mu = mu.reshape((mu.shape[0], 2, components_size[0]))
    # sigma = sigma_linear.apply(output_hiddens)
    # sigma = T.nnet.softplus(sigma)
    # print mu_linear.parameters
    # print mu_linear.parameters[0].get_value()
    # sigma = tensor.nnet.softplus(sigma)
    #fprop_x_sigma = theano.function([x], hiddens + [sigma])

    fprop_hidden_gaussian = theano.function([output_hiddens], [mu])

    # loading the second model
    network_mode = 1

    print('Loading model from {0}...'.format(save_path[1]))

    x2 = tensor.tensor3('features', dtype=theano.config.floatX)
    y2 = tensor.tensor3('targets', dtype=theano.config.floatX)

    x2 = x2.swapaxes(0,1)
    y2 = y2.swapaxes(0,1)

    y_2, cost_2, cells = nn_fprop(x2, y2, in_size[1], out_size[1], hidden_size[1], len(layer_models[1]), layer_models[1], 'SEC_MDN', training=False)
    main_loop = MainLoop(algorithm=None, data_stream=None, model=Model(cost_2),
                         extensions=[saveload.Load(save_path[1])])

    print "y_2 shape is: " + str(y_2.shape)

    for extension in main_loop.extensions:
        extension.main_loop = main_loop
    main_loop._run_extensions('before_training')
    bin_model_two = main_loop.model
    print 'second model loaded. Building prediction function...'
    hiddens_two = []
    initials_two = []

    for i in range(len(layer_models[1])):
        brick = [b for b in bin_model_two.get_top_bricks() if b.name == layer_models[1][i] + str(i) + '-'][0]
        hiddens_two.extend(VariableFilter(theano_name=brick.name + '_apply_states')(bin_model_two.variables))
        hiddens_two.extend(VariableFilter(theano_name=brick.name + '_apply_cells')(cells))
        initials_two.extend(VariableFilter(roles=[roles.INITIAL_STATE])(brick.parameters))

    fprop_hidden_hidden = theano.function([x2], hiddens_two + [y_2])


file = h5py.File(hdf5_file[0], 'r')
input_dataset = file['features']  # input_dataset.shape is (8928, 200, 2)
input_dataset = np.swapaxes(input_dataset,0,1)
nsamples = len(input_dataset[-1, :, -1])
print nsamples

output_mu = np.empty((200, 2, components_size[network_mode]), dtype='float32')
# myfile = open('output.csv', 'wb')
# wr = csv.writer(myfile)
for i in range(200):
    x_curr = input_dataset[:,i:i+1,:]
    #print x_curr
    input_sec_network, newinitials_one = sample(x_curr, fprop_x_hidden)  # the shape of input_sec_network is (200,)
    #test = input_sec_network[-1]

    #print np.mean(my_longitude(test[0]))
    #print np.mean(my_latitude(test[1]))
    ############  make data for the second network ########################
    #print input_helper[i].shape
    final_hiddens, newinitials_two = sample([input_sec_network], fprop_hidden_hidden)

    test, newinitials_three = sample(final_hiddens[-1], fprop_hidden_gaussian)
    test = test[-1]
    test[0] = my_longitude(test[0])
    test[1] = my_latitude(test[1])
    output_mu[i, 0, :] = test[0]
    output_mu[i, 1, :] = test[1]
    # print "mu shape is:"
    # print test.shape
# columns = ['longitude','latitude']
# dtype = [('longitude','float32'), ('latitude','float32')]


##############################################
#############  Plotting  #####################
index = 0
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)

longitude = input_dataset[:, index, 0]
longitude = longitude[longitude > 0.5]
longitude = longitude[longitude < 4]
original_longitude = my_longitude(longitude)

latitude = input_dataset[:, index, 1]
latitude = latitude[latitude > 0.5]
latitude = latitude[latitude < 4]
original_latitude = my_latitude(latitude)
#
#
l1, = plt.plot(original_longitude, original_latitude, 'bo')
#
plt.axis([-74.2, -73.65, 40.55, 41.0])
l2, = plt.plot(output_mu[index, 0, :], output_mu[index, 1, :], 'ro')

plt.title("Prediction Demo")
plt.xlabel("longitude")
plt.ylabel("latitude")



class Index(object):
    index = 0

    def next(self, event):
        self.index += 1
        #plt.plot(output_mu[self.index, 0, :], output_mu[self.index, 1, :], 'ro')

        longitude = input_dataset[:, self.index, 0]
        original_longitude = my_longitude(longitude[longitude > 0.5])

        latitude = input_dataset[:, self.index, 1]
        original_latitude = my_latitude(latitude[latitude > 0.5])

        l1.set_xdata(original_longitude)
        l1.set_ydata(original_latitude)

        l2.set_xdata(output_mu[self.index, 0, :])
        l2.set_ydata(output_mu[self.index, 1, :])
        plt.draw()

callback = Index()
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
plt.show()

# longitude = input_dataset[:, index, 0]
# original_longitude = my_longitude(longitude[longitude > 0])
#
# latitude = input_dataset[:, index, 1]
# original_latitude = my_latitude(latitude[latitude > 0])
#
#
# plt.plot(original_longitude, original_latitude, 'bo')
#
# plt.plot(output_mu[index, 0, :], output_mu[index, 1, :], 'ro')
# plt.axis([-74.06, -73.77, 40.61, 40.91])
# plt.show()
