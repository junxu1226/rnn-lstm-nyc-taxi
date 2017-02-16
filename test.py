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

    x0 = tensor.tensor3('features', dtype = 'floatX')
    y0 = tensor.tensor3('targets', dtype = 'floatX')

    x = x0.swapaxes(0,1)
    y = y0.swapaxes(0,1)

    x_mask = []
    y_mask = []

    in_size = num_features
    out_size = num_features

    y_hat, cost, cells = nn_fprop(x, y, x_mask, y_mask, in_size, out_size, hidden_size[network_mode], num_layers, layer_models[network_mode][0], 'MDN', training=False)
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

    fprop_mu = theano.function([x], hiddens + [y_hat])

    file = h5py.File(hdf5_file[network_mode], 'r')
    input_x = file['features']  # input_dataset.shape is (8928, 200, 2) features_x0
    target = file['targets']
    uniqueGeo = file['uniqueGeo']
    uniqueGeo = np.asarray(uniqueGeo)
    features = np.zeros((uniqueGeo.shape[0], 2), dtype='float32')
    for i in range(uniqueGeo.shape[0]):
        features[i] = geohash.decode(uniqueGeo[i])


    predict_pickups = []
    original_pickups = []

    startDateTime = datetime.datetime(2013, 01, 01, 1, 0, 0)
    endDateTime = datetime.datetime(2013, 01, 01, 2, 0, 0)
    sampleDataTime = endDateTime - startDateTime
    endDateTime = datetime.datetime(2013, 01, 3, 1, 0, 0)
    predict_num_timestamp = int((endDateTime - startDateTime).total_seconds() / (timestamp_length * 60))
    print predict_num_timestamp

    columns = ['datetime','latitude', 'longitude', 'pre_num_pickups', 'ori_num_pickups']
    # df_all = pd.DataFrame(columns=columns)
    # df_all['datetime'] = pd.to_datetime(df_all['datetime'])
    # df_all.to_csv('output.csv', index=False)
    x_curr = np.zeros((1, 1, num_features), dtype='float32')
    x_curr[0, 0, :] = input_x[0, 0]
    output_mu = []
    original_mu = []
    previous_pre = 0
    previous_ori = 0
    for i in range(predict_num_timestamp):

        mu_, newinitials = sample(x_curr, fprop_mu)

        # the shape of mu_ would be (1, 1, num_features)
        # x_curr = mu_
        x_curr = [[target[i/12, i%12]]]
        for initial, newinitial in zip(initials, newinitials):
           initial.set_value(newinitial[-1].flatten())
        mu_[mu_<0] = 0
        helper = features[np.round(std * mu_[-1, -1]) > 1.0]
        output_mu.append(helper)

        helper2 = features[np.round(std * target[i/12, i%12]) >= 1.0]
        original_mu.append(helper2)
        currentDataTime = startDateTime + i * sampleDataTime
        print currentDataTime
        df_eachPiece = pd.DataFrame(columns=columns)
        df_eachPiece['latitude'] = features[:, 0]
        df_eachPiece['longitude'] = features[:, 1]
        df_eachPiece['pre_num_pickups'] = previous_pre + np.round(std * mu_[-1, -1])
        # previous_pre = previous_pre + np.round(std * mu_[-1, -1])
        df_eachPiece['ori_num_pickups'] = previous_ori + np.round(std * target[i/12, i%12])
        # previous_ori = previous_ori + np.round(std * target[i/12, i%12])
        print "The root mean square error is: " + str(np.sqrt(mean_squared_error(np.round(std * mu_[-1, -1]), np.round(std * target[i/12, i%12]))))
        # df_eachPiece['datetime'] = pd.to_datetime(df_eachPiece['datetime'])
        # df_eachPiece['datetime'] = currentDataTime.strftime('%Y-%m-%d %H:%M:%S')
        # df_all.append(df_eachPiece, ignore_index=True)
        # with open('output.csv', 'a') as f:
            # df_eachPiece.to_csv(f, header=False, index=False)
    output_mu = np.asarray(output_mu)
    original_mu = np.asarray(original_mu)


##############################################
#############  Plotting  #####################
    index = 0
    plt.ion()
    fig, ax = plt.subplots()
    plt.xlim([-74.05, -73.78])
    plt.ylim([40.6, 40.9])
    plt.subplots_adjust(bottom=0.2)

    l1, = plt.plot(original_mu[index][:, 1], original_mu[index][:, 0], 'bo', markersize=4)
    l2, = plt.plot(output_mu[index][:, 1], output_mu[index][:, 0], 'ro', markersize=4)
    # plt.text(-74, 40.7,'Current time stamp: %s' %startDateTime.strftime('%Y-%m-%d %H:%M:%S'), fontsize=12)
    plt.title("Prediction Demo")
    plt.xlabel("longitude")
    plt.ylabel("latitude")

    class Index(object):
        index = 0

        def next(self, event):
            self.index += 1
            currentDataTime = startDateTime + self.index * sampleDataTime
            time_in_string = currentDataTime.strftime('%Y-%m-%d %H:%M:%S')
            # plt.text(-74, 40.7,'Current time stamp: %s' %time_in_string, fontsize=12)
            print "Current time stamp is: " + time_in_string
            l1.set_xdata(original_mu[self.index][:, 1])
            l1.set_ydata(original_mu[self.index][:, 0])
            # l1.set_color(original_pickups[self.index])
            # time.sleep(1)
            l2.set_xdata(output_mu[self.index][:, 1])
            l2.set_ydata(output_mu[self.index][:, 0])
            # plt.axes([-74.2, -73.65, 40.55, 41.0])

            plt.draw()

            # ax.collections = []

    callback = Index()
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    plt.ioff()
    plt.show()
