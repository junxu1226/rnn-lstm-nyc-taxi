import pandas as pd
import datetime
import numpy as np
import h5py
from fuel.datasets import H5PYDataset
from config import config

locals().update(config)
network_mode = 0

timeprocess = lambda x: x.replace(minute = (x.minute/5) * 5, second=0)
date_parser = pd.tseries.tools.to_datetime

df=pd.read_csv('./green_tripdata_2015-12.csv', index_col = False, parse_dates=['lpep_pickup_datetime'], date_parser=date_parser, sep=",")
df.dropna()
df = df[(df[u'Pickup_longitude'] > -74.06) & (df[u'Pickup_longitude'] < -73.76) &
        (df[u'Pickup_latitude'] > 40.61) & (df[u'Pickup_latitude'] < 40.91)]

df['Pickup_datetime'] = pd.Series(df['lpep_pickup_datetime'].apply(timeprocess), index = df.index)
df['Pickup_longitude'] = df['Pickup_longitude'].apply(lambda x: 2 + (x + 73.915) * 7)
df['Pickup_latitude'] = df['Pickup_latitude'].apply(lambda x: 2 + (x - 40.76) * 7)

data = df[['Pickup_datetime','Pickup_longitude','Pickup_latitude']]

startDateTime = datetime.datetime(2015, 12, 01, 0, 0, 0)
endDateTime = datetime.datetime(2015, 12, 01, 0, 5, 0)
sampleDataTime = endDateTime - startDateTime
endDateTime = datetime.datetime(2016, 01, 01, 0, 0, 0)
nsamples = (endDateTime - startDateTime).total_seconds() / 300;


in_size = len(input_columns)
out_size = len(output_columns)
inputs = np.empty((nsamples, seq_length[network_mode], in_size), dtype='float32')
outputs = np.empty((nsamples, seq_length[network_mode], out_size), dtype='float32')

for i in range(int(nsamples)):

    currentDataTime = startDateTime + i * sampleDataTime
    dataEachTime = data[data['Pickup_datetime'] == currentDataTime]
    eachPiece = np.array(dataEachTime[input_columns].as_matrix())
    if len(eachPiece) >= (seq_length[network_mode]):
        eachPiece = eachPiece[0:seq_length[network_mode], 0:seq_length[network_mode]]
        inputs[i] = eachPiece
    else:
        fillZeros = np.zeros((seq_length[network_mode] - len(eachPiece), in_size))
        inputs[i] = np.concatenate((fillZeros, eachPiece))
    outputs[i] = inputs[i]
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
