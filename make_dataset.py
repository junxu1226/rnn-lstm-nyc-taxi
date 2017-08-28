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
# date_parser = pd.tseries.tools.to_datetime

# path ='./data/' # use your path
# all_files = glob.glob(os.path.join(path, "*.csv"))
# print "total number of datasets: " + str(len(all_files)) + ", in " + str(len(all_files) / 2) +" months"
# date_parser=date_parser,
# df_each = (pd.read_csv(f, index_col = False, header=0, usecols=[1, 5, 6], parse_dates=[0], infer_datetime_format=True, names=['Pickup_datetime', 'Pickup_longitude', 'Pickup_latitude'], memory_map=True) for f in all_files)
#
# df = pd.concat(df_each, ignore_index=True)
# df_2013 = pd.read_csv('./data/2013_5.csv', index_col = False, parse_dates=[0], infer_datetime_format=True)
# df_2014 = pd.read_csv('./data/2014_5.csv', index_col = False, parse_dates=[0], infer_datetime_format=True)
# df_2015 = pd.read_csv('./data/2015_5.csv', index_col = False, parse_dates=[0], infer_datetime_format=True)
df_2016 = pd.read_csv('./data/2016_5.csv', index_col = False, parse_dates=[0], infer_datetime_format=True)


data_geohash = np.array(df_2016['Pickup_geohash'], dtype="S7")
print data_geohash.shape
# data_geohash = np.concatenate((data_geohash, np.array(df_2014['Pickup_geohash'], dtype="S7"), np.array(df_2015['Pickup_geohash'], dtype="S7"), np.array(df_2016['Pickup_geohash'], dtype="S7")))
# data_geohash = np.concatenate((data_geohash, np.array(df_2015['Pickup_geohash'], dtype="S7")))
# print data_geohash.shape


# print "total number of GPS traces: " + str(df.shape[0])

sorted_unique_geohash, num_pickups = np.unique(data_geohash, return_counts=True)
print "total number of geohashed location: " + str(sorted_unique_geohash.shape)

sorted_unique_geohash = sorted_unique_geohash[num_pickups > 200]
sorted_unique_geohash.sort()
print "after dropping small pickups (number of features): " + str(sorted_unique_geohash.shape)

startDateTime = datetime.datetime(2016, 01, 01, 0, 0, 0)
endDateTime = datetime.datetime(2016, 01, 01, 0, 10, 0)
sampleDataTime = endDateTime - startDateTime
endDateTime = datetime.datetime(2016, 07, 01, 0, 0, 0)
total_timestamp = (endDateTime - startDateTime).total_seconds() / (timestamp_length * 60)
print "total number of timestamps: " + str(total_timestamp)


# data processing
data_all =[]
time = []
data = df_2016
for i in range(int(total_timestamp)):
    currentDataTime = startDateTime + i * sampleDataTime
    if i % 6 == 0:
        print currentDataTime
    # if currentDataTime == datetime.datetime(2016, 01, 01, 0, 0, 0):
    #     data = df_2016
    #     print currentDataTime
    #     print data[:10]
    # elif currentDataTime == datetime.datetime(2014, 01, 01, 0, 0, 0):
    #     data = df_2014
    #     print currentDataTime
    #     print data[:10]
    # elif currentDataTime == datetime.datetime(2015, 01, 01, 0, 0, 0):
    #     data = df_2015
    #     print currentDataTime
    #     print data[:10]
    # elif currentDataTime == datetime.datetime(2016, 01, 01, 0, 0, 0):
    #     data = df_2016
    #     print currentDataTime
    #     print data[:10]

    data_curr, data = data[data['Pickup_datetime'] < (currentDataTime + sampleDataTime)], data[data['Pickup_datetime'] >= (currentDataTime + sampleDataTime)]
    geo_curr = np.array(data_curr['Pickup_geohash'], dtype="S7")
    sorted_uniques, count = np.unique(geo_curr, return_counts=True)   # get the sorted unique array and the count

    pickups_curr = np.zeros(sorted_unique_geohash.shape, dtype='float32')  # create current pickup features
    pickups_curr[np.in1d(sorted_unique_geohash, sorted_uniques)] = count[np.in1d(sorted_uniques, sorted_unique_geohash)]  # set pickup num
    time_info_each_day = np.zeros(75, dtype='float32')
    # time_info_each_day[0] = (currentDataTime.year - 2012 - 2.5) / 1.118   # year
    # time_info_each_day[currentDataTime.year - 2013] = 1.0
    time_info_each_day[currentDataTime.month] = 1.0        # month
    time_info_each_day[6 + currentDataTime.day] = 1.0          # day
    time_info_each_day[37 + currentDataTime.hour + 1] = 1.0        # hour
    time_info_each_day[61 + int(currentDataTime.minute/10) + 1] = 1.0        # hour
    day_of_week = ((i/6)/24 + 4) % 7 + 1    #from 1 - 7, 2013/01/01 is Tuesday, which is 1
    time_info_each_day[67 + day_of_week] = 1.0           # day of that week, the 2013/01/01 is Tuesday
    if i < 2:
        print data_curr.shape
        print data.shape
        print data_curr[:20]
        print sorted_uniques[:20]
        print count[:20]

    data_all.append(pickups_curr)
    time.append(time_info_each_day)


nsamples = int((len(data_all) - 1) // seq_length[network_mode])
inputs = np.zeros((nsamples, seq_length[network_mode], sorted_unique_geohash.shape[0] + 75), dtype='float32')
outputs = np.zeros((nsamples, seq_length[network_mode], sorted_unique_geohash.shape[0] + 75), dtype='float32')


index = 0
for i in range(int(nsamples)):
    for j in range(seq_length[network_mode]):
        inputs[i, j,:sorted_unique_geohash.shape[0]] = data_all[index]
        inputs[i, j,sorted_unique_geohash.shape[0]:] = time[index]
        outputs[i, j,:sorted_unique_geohash.shape[0]] = data_all[index + 1]
        outputs[i, j,sorted_unique_geohash.shape[0]:] = time[index + 1]
        index = index + 1


helper = inputs[:, :, :sorted_unique_geohash.shape[0]]
helper = helper[helper > 0]
print 'the max of pickup is: ' + str(np.max(helper))
pickups_std = np.std(helper).round()
print 'the std of pickup is: ' + str(pickups_std)

inputs[:,:,:sorted_unique_geohash.shape[0]] = inputs[:,:,:sorted_unique_geohash.shape[0]] / pickups_std
outputs[:,:,:sorted_unique_geohash.shape[0]] = outputs[:,:,:sorted_unique_geohash.shape[0]] / pickups_std
print inputs[-1,0:3, 0:100]
print outputs[-1,0:3, 0:100]


print sorted_unique_geohash[:10]
uniqueGeoHash = np.array(sorted_unique_geohash, dtype="S7")
print 'inputs shape:', inputs.shape
print 'outputs shape:', outputs.shape
print 'uniqueGeoHash shape:', uniqueGeoHash.shape

###################################################################################################
############################## for raw num pickups ##############################
# f_raw_data = h5py.File('pickups_raw.hdf5', mode='w')
# rawdata = f_raw_data.create_dataset('rawdata', helper_inputs.shape, dtype='float32')
# geohash_rawdata = f_raw_data.create_dataset('geohash_rawdata', uniqueGeoHash.shape, dtype="S7")
# rawdata[...] = helper_inputs
# geohash_rawdata[...] = uniqueGeoHash
# print geohash_rawdata[:10]
# f_raw_data.flush()
# f_raw_data.close()
# #####################################################################################################


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
