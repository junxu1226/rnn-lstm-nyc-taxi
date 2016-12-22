import pandas as pd
import datetime
import numpy as np
import h5py
from fuel.datasets import H5PYDataset
from config import config
import matplotlib.pyplot as plt
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

# data_long = np.array(df['Pickup_longitude'].as_matrix())

data = df[['Pickup_datetime','Pickup_longitude','Pickup_latitude']]
print "total number of GPS traces: " + str(data.shape[0])

startDateTime = datetime.datetime(2016, 03, 01, 0, 0, 0)
endDateTime = datetime.datetime(2016, 03, 01, 0, 5, 0)
sampleDataTime = endDateTime - startDateTime
endDateTime = datetime.datetime(2016, 05, 01, 0, 0, 0)
nsamples = (endDateTime - startDateTime).total_seconds() / 300;
print "total number of time stamps: " + str(nsamples)


currentDataTime = startDateTime
numberofPoints = []
for i in range(int(nsamples)):
    dataEachTime = data[data['Pickup_datetime'] == currentDataTime]
    numberofPoints.append(len(dataEachTime))
    currentDataTime = currentDataTime + sampleDataTime

#print(inputs)
#print numberofPoints
print(np.mean(numberofPoints))
print(np.min(numberofPoints))
print(np.max(numberofPoints))


# print len(data_long)
# print(np.mean(data_long))
# print(np.min(data_long))
# print(np.max(data_long))
#
#
#
# print len(data_lati)
# print(np.mean(data_lati))
# print(np.min(data_lati))
# print(np.max(data_lati))

# data_lati = data_lati[data_lati > 41.]
# print len(data_lati)

# plt.plot(data_long, data_lati, 'bo')
# plt.show()
