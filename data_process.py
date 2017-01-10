import pandas as pd
import datetime
import numpy as np
import h5py
from fuel.datasets import H5PYDataset
from config import config
import os
import glob
from utils import get_stream, track_best, MainLoop
from theano import tensor

locals().update(config)
network_mode = 0

# timeprocess = lambda x: x.replace(minute = (x.minute/5) * 5, second=0)
# date_parser = pd.tseries.tools.to_datetime

nsamples=10000
# inputs = np.empty((nsamples, 100, 2), dtype='float32')
# outputs = np.zeros((nsamples, 100, 2), dtype='float32')
# inputs_mask = np.ones((nsamples, 100), dtype='float32')
# outputs_mask = np.ones((nsamples, 100), dtype='float32')




inputs_x0 = []
outputs_x0 = []

inputs_x1 = []
outputs_x1 = []
# sizes = np.random.randint(0, 200, size=(nsamples,))
# seq=0
#
#
# inputs = np.random.randint(1, size=(nsamples, size, 2)).astype('float32')


for i in range(nsamples):

    seq = np.random.randint(300) # just one number, seq_length
    eachPiece = np.random.rand(seq) * 2 + 1
    fillEnd = np.ones((1)) * 4
    eachPiece = np.concatenate((eachPiece, fillEnd))
    # eachPiece = eachPiece.swapaxes(0,1)
    # eachPiece = eachPiece.tolist()
    inputs_x0.append(eachPiece)
    # print inputs_x0[i].shape
    eachPiece = np.random.rand(seq) * 2 + 1
    eachPiece = np.concatenate((eachPiece, fillEnd))
    inputs_x1.append(eachPiece)


# print inputs_x0[0].shape
print np.asarray(inputs_x0[0]).shape[0]
print np.asarray(inputs_x0[0]).shape

eachPiece = np.random.rand(300) * 2 + 1
fillEnd = np.ones((1)) * 4
eachPiece = np.concatenate((eachPiece, fillEnd))

inputs_x0[0] = eachPiece

eachPiece = np.random.rand(300) * 2 + 1
fillEnd = np.ones((1)) * 4
eachPiece = np.concatenate((eachPiece, fillEnd))

inputs_x1[0] = eachPiece

print np.asarray(inputs_x0[0]).shape

eachPiece = np.random.rand(300) * 2 + 1
fillEnd = np.ones((1)) * 4
eachPiece = np.concatenate((eachPiece, fillEnd))

inputs_x0[nsamples - 1] = eachPiece

eachPiece = np.random.rand(300) * 2 + 1
fillEnd = np.ones((1)) * 4
eachPiece = np.concatenate((eachPiece, fillEnd))

inputs_x1[nsamples - 1] = eachPiece

# print inputs_x0[0].shape
# print inputs_x0[nsamples - 1].shape
# # print inputs[0]
outputs_x0 = inputs_x0
outputs_x1 = inputs_x1
print len(outputs_x1)



f = h5py.File(hdf5_file[network_mode], mode='w')



dtype = h5py.special_dtype(vlen=np.dtype('float32'))
features_x0 = f.create_dataset('features_x0', (nsamples,), dtype=dtype)
features_x1 = f.create_dataset('features_x1', (nsamples,), dtype=dtype)

targets_x0 = f.create_dataset('targets_x0', (nsamples,), dtype=dtype)
targets_x1 = f.create_dataset('targets_x1', (nsamples,), dtype=dtype)


features_x0[...] = inputs_x0
features_x1[...] = inputs_x1
targets_x0[...] = outputs_x0
targets_x1[...] = outputs_x1

print len(features_x0)

print len(features_x1[0])
print targets_x0[0].shape
#
# print features[0].shape
# print np.asarray(features[0]).shape
# inputs = np.random.rand(100000, seq, 2) * 2 + 1
#
# outputs = inputs

# print 'inputs shape:', inputs.shape
# print 'outputs shape:', outputs.shape
#

# features[...] = inputs
# targets[...] = outputs
features_x0.dims[0].label = 'batch'
# features_x0.dims[1].label = 'sequence'
features_x1.dims[0].label = 'batch'
# features_x1.dims[1].label = 'sequence'
#
#
targets_x0.dims[0].label = 'batch'
# targets_x0.dims[1].label = 'sequence'
targets_x1.dims[0].label = 'batch'
# targets_x1.dims[1].label = 'sequence'
#
nsamples_train = int(nsamples * train_size[network_mode])
split_dict = {
    'train': {'features_x0': (0, nsamples_train), 'features_x1': (0, nsamples_train), 'targets_x0': (0, nsamples_train), 'targets_x1': (0, nsamples_train)},
    'test': {'features_x0': (nsamples_train, nsamples), 'features_x1': (nsamples_train, nsamples), 'targets_x0': (nsamples_train, nsamples), 'targets_x1': (nsamples_train, nsamples)}}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()

# file = h5py.File(hdf5_file[network_mode], 'r')

# train_stream = get_stream(hdf5_file[network_mode], 'train', batch_size[network_mode])
# # # x0 = tensor.tensor3('features_x0', dtype = 'floatX')
# data = train_stream.sources
# print len(data)
# it = train_stream.get_epoch_iterator()
# data = next(it)

# print  data
# print train_stream.sources
