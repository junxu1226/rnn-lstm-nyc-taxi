import pandas as pd
import datetime
import numpy as np
import h5py
from fuel.datasets import H5PYDataset
from config import config
import os
import glob

locals().update(config)
network_mode = 0

# timeprocess = lambda x: x.replace(minute = (x.minute/5) * 5, second=0)
# date_parser = pd.tseries.tools.to_datetime

nsamples=100000
inputs = np.empty((nsamples, 100, 2), dtype='float32')
outputs = np.zeros((nsamples, 100, 2), dtype='float32')

seq=0

for i in range(nsamples):

    seq = np.random.randint(200)

    if seq >= 99:
        eachPiece = np.random.rand(99, 2) * 2 + 1
        fillEnd = np.ones((1, 2)) * 4
        inputs[i] = np.concatenate((eachPiece, fillEnd))
    else:
        fillZeros = np.zeros((99 - seq, 2))
        eachPiece = np.random.rand(seq, 2) * 2 + 1
        fillEnd = np.ones((1, 2)) * 4
        inputs[i] = np.concatenate((np.concatenate((fillZeros, eachPiece)), fillEnd))
    outputs[i] = inputs[i]



# inputs = np.random.rand(100000, seq, 2) * 2 + 1
#
# outputs = inputs

print 'inputs shape:', inputs.shape
print 'outputs shape:', outputs.shape

f = h5py.File(hdf5_file[network_mode], mode='w')
features = f.create_dataset('features', inputs.shape, dtype='float32')
targets = f.create_dataset('targets', outputs.shape, dtype='float32')
features[...] = inputs
targets[...] = outputs
features.dims[0].label = 'batch'
features.dims[1].label = 'sequence'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'sequence'

nsamples_train = int(nsamples * train_size[network_mode])
split_dict = {
    'train': {'features': (0, nsamples_train), 'targets': (0, nsamples_train)},
    'test': {'features': (nsamples_train, nsamples), 'targets': (nsamples_train, nsamples)}}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()
