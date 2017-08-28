import numpy as np
import h5py
import theano
import datetime
from config import config
from fuel.datasets import H5PYDataset

locals().update(config)
network_mode = 0

file = h5py.File('input_double_20.hdf5', 'r')
input = np.array(file['features'], dtype='float32')
target = np.array(file['targets'], dtype='float32')
uniqueGeoHash= np.array(file['uniqueGeo'], dtype='S7')

wea_file = h5py.File('weather.hdf5', 'r')
weather = np.array(wea_file['wea'], dtype='float32')
print weather[:10]

in_size = 2*uniqueGeoHash.shape[0]

input = input[:, :, :in_size]
target = target[:, :, :uniqueGeoHash.shape[0]]
input = input.round()
target = target.round()

print input[0,1,:10]
print target[0,0,:10]


# inputs = input
N1 = input.shape[0]
N2 = input.shape[1]
N3 = input.shape[2]

inputs = np.ones((N1, N2, N3+79), dtype='float32')
inputs[:,:,:N3] = input
# inputs = inputs*0.001

# index = 0
# for i in xrange(0,2*uniqueGeoHash.shape[0],2):
#     inputs[:,:,i] = input[:,:,index]
#     inputs[:,:,i+1] = input[:,:,index+uniqueGeoHash.shape[0]]
#     index = index + 1
# inputs = np.zeros((N1, N2, 71), dtype='float32')
#
startDateTime = datetime.datetime(2014, 01, 01, 0, 0, 0)
endDateTime = datetime.datetime(2014, 01, 01, 0, 20, 0)
sampleDataTime = endDateTime - startDateTime
# #
index = 0
for i in range(int(N1)):
    for j in range(N2):
        currentDataTime = startDateTime + index * sampleDataTime
        index = index + 1
        time_info_each_step = np.zeros(67, dtype='float32')
        time_info_each_step[currentDataTime.month] = 1.0        # month
        time_info_each_step[currentDataTime.day] = 1.0          # day
        time_info_each_step[31 + currentDataTime.hour + 1] = 1.0        # hour
        time_info_each_step[55 + int(currentDataTime.minute/20) + 1] = 1.0        # min
        time_info_each_step[(index/(3*24)+2)%7+1+59] = 1.
        # if index < 10:
        #     print time_info_each_step
        # inputs[i, j, :N3] = input[i, j]
        inputs[i, j, N3:N3+67] = time_info_each_step
        inputs[i, j, N3+67:] = weather[index/(3*24)]
        # previous = input[i, j, :]
        # target[i,j] = target[i,j] - previous
        # time_info_each_day = np.zeros(32, dtype='float32')
        # time_info_each_day[j%24 + 1] = 1.0        # hour
        # day_of_week = (j/24 + 2) % 7 + 1    #from 1 - 7, 2013/01/01 is Tuesday, which is 1
        # time_info_each_day[24 + day_of_week] = 1.0
        # inputs[i, j, N3:N3+32] = time_info_each_day
        # inputs[i, j, N3:] = wea[:,index,:].flatten()
        # if (j+1)%24 == 0:
        #     index = index + 1

############################ JUST TIME ########################



helper = np.array(inputs)
print 'the std of pickup is: ' + str(np.std(helper))
helper = helper[helper>0]
pickups_std = np.ceil(2*np.std(helper))
print 'the max of pickup is: ' + str(np.max(helper))
print 'the std of pickup is: ' + str(pickups_std)

print uniqueGeoHash[:10]

# pickups_std = 13.0
inputs[:, :, :in_size] = inputs[:, :, :in_size] / 10.
target[:, :, :uniqueGeoHash.shape[0]] = target[:, :, :uniqueGeoHash.shape[0]] / 10.
print inputs[0,1,:10]
print target[0,0,:10]
# inputs = inputs.round()
# target = target.round()

# inputs[:, :, :uniqueGeoHash.shape[0]] = np.log10(inputs[:, :, :uniqueGeoHash.shape[0]] + 1.)
# target[:, :, :uniqueGeoHash.shape[0]] = np.log10(target[:, :, :uniqueGeoHash.shape[0]] + 1.)

print 'inputs shape:', inputs.shape
print 'outputs shape:', target.shape
print 'uniqueGeoHash shape:', uniqueGeoHash.shape

f = h5py.File(hdf5_file[network_mode], mode='w')
features = f.create_dataset('features', inputs.shape, dtype='float32')
targets = f.create_dataset('targets', target.shape, dtype='float32')
uniqueGeo = f.create_dataset('uniqueGeo', uniqueGeoHash.shape, dtype="S7")

features[...] = inputs
targets[...] = target
uniqueGeo[...] = uniqueGeoHash
features.dims[0].label = 'batch'
features.dims[1].label = 'sequence'
targets.dims[0].label = 'batch'
targets.dims[1].label = 'sequence'
print uniqueGeo[:10]
print features[-1,-1,:10]
print targets[-1,-2,:10]
# print features[-1,-1,-50:-1]
nsamples = inputs.shape[0]
print nsamples
nsamples_train = int(nsamples * train_size[network_mode])

split_dict = {
    'train': {'features': (0, nsamples_train), 'targets': (0, nsamples_train)},
    'test': {'features': (nsamples_train, nsamples), 'targets': (nsamples_train, nsamples)}}

f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f.flush()
f.close()
