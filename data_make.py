import pandas as pd
import datetime
import numpy as np
import h5py
from fuel.datasets import H5PYDataset
from config import config
import os
import glob
import geohash

locals().update(config)
network_mode = 0

# timeprocess = lambda x: x.replace(minute = (x.minute/timestamp_length) * timestamp_length, second=0)
date_parser = pd.tseries.tools.to_datetime

path ='./data/' # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))
print "total number of datasets: " + str(len(all_files)) + ", in " + str(len(all_files) / 2) +" months"
# date_parser=date_parser,
df_each = (pd.read_csv(f, index_col = False, header=0, usecols=[1, 5, 6], parse_dates=[0], infer_datetime_format=True, names=['Pickup_datetime', 'Pickup_longitude', 'Pickup_latitude'], memory_map=True) for f in all_files)

df = pd.concat(df_each, ignore_index=True)

df = df[(df['Pickup_longitude'] > -74.06) & (df['Pickup_longitude'] < -73.77) &
        (df['Pickup_latitude'] < 40.91) & (df['Pickup_latitude'] > 40.61)]
print df.shape
df['Pickup_datetime'] = df['Pickup_datetime'].apply(lambda x: x.replace(minute=0, second=0))
print df[:10]
df['Pickup_geohash'] = df[['Pickup_latitude', 'Pickup_longitude']].apply(lambda x: geohash.encode(x[0], x[1], precision=7), axis=1)
print df[:10]
df = df[['Pickup_datetime', 'Pickup_geohash']]
print df.head(10)
print df.shape

print "total number of GPS traces: " + str(df.shape[0])

data_geohash = np.array(df['Pickup_geohash'], dtype="S7")
sorted_unique_geohash, num_pickups = np.unique(data_geohash, return_counts=True)
print "total number of geohashed location: " + str(sorted_unique_geohash.shape)

sorted_unique_geohash = sorted_unique_geohash[num_pickups > 24]
sorted_unique_geohash.sort()
print "after dropping small pickups (number of features): " + str(sorted_unique_geohash.shape)

startDateTime = datetime.datetime(2015, 01, 01, 0, 0, 0)
endDateTime = datetime.datetime(2015, 01, 01, 1, 0, 0)
sampleDataTime = endDateTime - startDateTime
endDateTime = datetime.datetime(2016, 05, 01, 0, 0, 0)
total_timestamp = (endDateTime - startDateTime).total_seconds() / (timestamp_length * 60)
print "total number of timestamps: " + str(total_timestamp)


# data processing
data_all =[]
for i in range(int(total_timestamp)):
    currentDataTime = startDateTime + i * sampleDataTime
    data_curr, df = df[df['Pickup_datetime'] == currentDataTime], df[df['Pickup_datetime'] != currentDataTime]
    geo_curr = np.array(data_curr['Pickup_geohash'], dtype="S7")
    sorted_uniques, count = np.unique(geo_curr, return_counts=True)   # get the sorted unique array and the count

    pickups_curr = np.zeros(sorted_unique_geohash.shape)  # create current pickup features
    pickups_curr[np.in1d(sorted_unique_geohash, sorted_uniques)] = count[np.in1d(sorted_uniques, sorted_unique_geohash)]  # set pickup num
    if i < 2:
        print sorted_uniques[:20]
        print count[:20]
        print pickups_curr[:100]
    data_all.append(np.array(pickups_curr))



nsamples = int((len(data_all) - 1) // seq_length[network_mode])
inputs = np.zeros((nsamples, seq_length[network_mode], sorted_unique_geohash.shape[0]), dtype='float32')
outputs = np.zeros((nsamples, seq_length[network_mode], sorted_unique_geohash.shape[0]), dtype='float32')


index = 0
for i in range(int(nsamples)):
    for j in range(seq_length[network_mode]):
        inputs[i, j] = data_all[index]
        outputs[i, j] = data_all[index + 1]
        index = index + 1


helper_inputs = inputs[inputs > 0]
print 'the max of pickup is: ' + str(np.max(helper_inputs))
pickups_std = np.std(helper_inputs).round()
print 'the std of pickup is: ' + str(pickups_std)

inputs = inputs / pickups_std
outputs = outputs / pickups_std
print inputs[-1,0:4, 0:50]
print outputs[-1,0:4, 0:50]


print sorted_unique_geohash[:10]
uniqueGeoHash = np.array(sorted_unique_geohash, dtype="S7")
print 'inputs shape:', inputs.shape
print 'outputs shape:', outputs.shape
print 'uniqueGeoHash shape:', uniqueGeoHash.shape
#
f = h5py.File(hdf5_file[network_mode], mode='w')
features = f.create_dataset('features', inputs.shape, dtype='float32')
targets = f.create_dataset('targets', outputs.shape, dtype='float32')
uniqueGeo = f.create_dataset('uniqueGeo', uniqueGeoHash.shape, dtype="S7")

features[...] = inputs
targets[...] = outputs
uniqueGeo[...] = uniqueGeoHash
features.dims[0].label = 'batch'
features.dims[1].label = 'sequence'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'sequence'
print uniqueGeo[:10]

nsamples_train = int(nsamples * train_size[network_mode])

split_dict = {
    'train': {'features': (0, nsamples_train), 'targets': (0, nsamples_train)},
    'test': {'features': (nsamples_train, nsamples), 'targets': (nsamples_train, nsamples)}}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()
