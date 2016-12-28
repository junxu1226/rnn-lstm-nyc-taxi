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

timeprocess = lambda x: x.replace(minute = (x.minute/5) * 5, second=0)
date_parser = pd.tseries.tools.to_datetime

path ='./data/' # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))
print "total number of datasets: " + str(len(all_files)) + ", in " + str(len(all_files)) +" months"

df_from_each_file = (pd.read_csv(f, index_col = False, parse_dates=['lpep_pickup_datetime'], date_parser=date_parser, sep=",") for f in all_files)
df   = pd.concat(df_from_each_file)

df = df[(df[u'Pickup_longitude'] > -74.2) & (df[u'Pickup_longitude'] < -73.65) &
        (df[u'Pickup_latitude'] < 41.0) & (df[u'Pickup_latitude'] > 40.55)]

df['Pickup_datetime'] = pd.Series(df['lpep_pickup_datetime'].apply(timeprocess), index = df.index)
df['Pickup_longitude'] = df['Pickup_longitude'].apply(lambda x: 2.1 + (x + 73.925) * 4.0) # (-1.1, 1.1) + 2.1 = (1.0, 3.2)
df['Pickup_latitude'] = df['Pickup_latitude'].apply(lambda x: 2.0 + (x - 40.775) * 4.0) # (-0.9, 0.9) + 2.0 = (1.1, 2.9)

data = df[['Pickup_datetime','Pickup_longitude','Pickup_latitude']]
print "total number of GPS traces: " + str(data.shape[0])

startDateTime = datetime.datetime(2016, 05, 01, 0, 0, 0)
endDateTime = datetime.datetime(2016, 05, 01, 0, 5, 0)
sampleDataTime = endDateTime - startDateTime
endDateTime = datetime.datetime(2016, 07, 01, 0, 0, 0)
nsamples = (endDateTime - startDateTime).total_seconds() / 300;
print "total number of time stamps: " + str(nsamples)


in_size = len(input_columns)
out_size = len(output_columns)
inputs = np.empty((nsamples, seq_length[network_mode], in_size), dtype='float32')
outputs = np.zeros((nsamples, seq_length[network_mode], out_size), dtype='float32')

for i in range(int(nsamples)):

    currentDataTime = startDateTime + i * sampleDataTime
    dataEachTime = data[data['Pickup_datetime'] == currentDataTime]
    eachPiece = np.array(dataEachTime[input_columns].as_matrix())
    if len(eachPiece) >= (seq_length[network_mode] - 1):
        eachPiece = eachPiece[0 : seq_length[network_mode] - 1, 0 : seq_length[network_mode] - 1]
        fillEnd = np.ones((1, in_size)) * 4
        inputs[i] = np.concatenate((eachPiece, fillEnd))
    else:
        fillZeros = np.zeros((seq_length[network_mode] - 1 - len(eachPiece), in_size))
        fillEnd = np.ones((1, in_size)) * 4
        inputs[i] = np.concatenate((np.concatenate((fillZeros, eachPiece)), fillEnd))


for j in range(1, int(nsamples)):
    outputs[j-1] = inputs[j]
    #inputs[i] = np.array(data[input_columns].as_matrix()[p:p + seq_length])
    #outputs[i] = np.array(data[output_columns].as_matrix()[p + 1:p + seq_length + 1])

print 'Data time steps: ', len(data)
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
