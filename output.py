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
import time
import geohash
import datetime
from sklearn.metrics import mean_squared_error
locals().update(config)


def sample(x_curr, fprop):

    hiddens = fprop(x_curr)
    probs = hiddens.pop().astype(theano.config.floatX)

    return probs, hiddens

def sMAPE(real, predict):
    abs_error = np.abs(real - predict)
    sums = 0.
    for i in range(real.shape[0]):
        sums = sums + abs_error[i]/(real[i] + predict[i] + 1.)
    error = sums/real.shape[0]

    return error


def naive_predict(targets, week_index, seq_index, fea_index):
    mu = 0.
    num_weeks = 5
    for i in range(num_weeks):
        mu = mu + targets[week_index-i-1, seq_index, fea_index]
    mu = mu /num_weeks
    return mu

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

    x = tensor.tensor3('features', dtype = 'floatX')
    y = tensor.tensor3('targets', dtype = 'floatX')
    x = x.swapaxes(0,1)
    y = y.swapaxes(0,1)
    in_size = num_features
    out_size = num_features
    y_hat, cost, cells = nn_fprop(x, y, in_size, out_size, hidden_size[network_mode], num_layers, layer_models[network_mode][0], 'MDN', training=False)
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

    fprop = theano.function([x], hiddens + [y_hat])

    file = h5py.File('input.hdf5', 'r')
    inputs = file['features']  # input_dataset.shape is (8928, 200, 2) features_x0
    target = file['targets']
    uniqueGeo = np.array(file['uniqueGeo'])
    features = np.zeros((uniqueGeo.shape[0], 2), dtype='float32')
    targets = std * target[:,:,:uniqueGeo.shape[0]]
    targets = np.round(targets)
    for i in range(uniqueGeo.shape[0]):
        features[i] = geohash.decode(uniqueGeo[i])

    dataset_start = datetime.datetime(2014, 01, 01, 0, 0, 0)
    startDateTime = datetime.datetime(2014, 1, 1, 0, 0, 0)
    endDateTime = datetime.datetime(2014, 01, 01, 1, 0, 0)
    sampleDataTime = endDateTime - dataset_start
    endDateTime = datetime.datetime(2014, 4, 1, 0, 0, 0)

    index = int((startDateTime - dataset_start).total_seconds() / (timestamp_length * 60))
    predict_num_timestamp = int((endDateTime - startDateTime).total_seconds() / (timestamp_length * 60))
    print predict_num_timestamp

    saveout1 = []
    saveout2 = []

    event_start = datetime.datetime(2014, 1, 1, 0, 0, 0)
    event_end = datetime.datetime(2014, 5, 1, 0, 0, 0)
    sum_predict = 0.
    sum_real = 0.
    predict_day = np.zeros(24, dtype='float32')
    real_day = np.zeros(24, dtype='float32')
    naive_day = np.zeros(24, dtype='float32')
    for k in range(predict_num_timestamp):
        i = k + index
        currentDataTime = startDateTime + k * sampleDataTime
        # print currentDataTime
        # x_curr = np.zeros((1, 1, num_features), dtype='float32')

        x_curr= [[inputs[i/seq_length[network_mode], i%seq_length[network_mode]]]]
        # x_c = np.repeat(x_curr, num_features, axis=0)
        predict = np.zeros(num_features, dtype='float32')

        mu, newinitials = sample(x_curr, fprop)
        for initial, newinitial in zip(initials, newinitials):
           initial.set_value(newinitial[-1].flatten())

        predict = mu[-1,-1] * std
        predict[predict < 0] = 0
        predict = np.round(predict)

        real = targets[i/seq_length[network_mode], i%seq_length[network_mode]]
        # original = np.power(10, target[i/seq_length[network_mode], i%seq_length[network_mode], :uniqueGeo.shape[0]]) - 1.
        real[real < 0] = 0
        # error = np.sqrt(mean_squared_error(mu_, original))
        # print "The root mean square error in current step is: " + str(error)
        # print real[:10]
        # print predict[:10]
        # print "The root mean square error in current time-step is: " + str(np.sqrt(mean_squared_error(real, predict)))
        if currentDataTime >= event_start and currentDataTime <= event_end:
            sum_predict = sum_predict + predict
            sum_real = sum_real + real
            if (k+1)%24==0:
                error_smape = sMAPE(predict_day, real_day)
                print "The sMAPE error in day is: " + str(error_smape)
                saveout1.append(error_smape)

                # naive_smape = sMAPE(naive_day, real_day)
                # print "The sMAPE error in day is: " + str(naive_smape)
                # saveout2.append(naive_smape)

                predict_day = np.zeros(24, dtype='float32')
                real_day = np.zeros(24, dtype='float32')
                naive_day = np.zeros(24, dtype='float32')


            if k > 0 and (k+1) % (60/timestamp_length) == 0:
                # print sum_predict[:10]
                # print sum_real[:10]
                # naive_day[k%24]=naive_predict(targets, i/seq_length[network_mode], i%seq_length[network_mode], 8)
                predict_day[k%24]=sum_predict[8]
                real_day[k%24]=sum_real[8]
                # saveout2.append(sum_real)
                sum_predict = 0.
                sum_real = 0.

    np.savetxt('./figures/lstm_standard_8.txt', saveout1)
    # np.savetxt('./figures/naive_8.txt', saveout2)
    # np.savetxt('save3.txt', saveout3)
