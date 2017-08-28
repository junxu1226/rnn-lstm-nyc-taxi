import pandas as pd
import datetime
import numpy as np
import h5py
import geohash

def convert(array):
    wea = np.zeros(4, dtype='float32')
    if array[0] > 0. or array[5] > 0.:#rain || rain and snow
        wea[0] = 1.
    if array[1] > 0 or array[2] > 0:#snow
        wea[1] = 1.
    if array[3] > 0 or array[4] > 0:#fog
        wea[2] = 1.
    if array[6] > 0:#thunder
        wea[3] = 1.
    return wea

def dist(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

def wea_location(sta_loca, Geohash):
    wea_index = np.zeros((Geohash.shape), dtype='int8')
    print wea_index.shape
    print Geohash.shape
    for i in range(Geohash.shape[0]):
        l = np.array(geohash.decode(Geohash[i]))
        print l
        print sta_loca[0]
        array = np.array([dist(l, sta_loca[0]), dist(l, sta_loca[1]), dist(l, sta_loca[2])])
        print array
        wea_index[i] = np.argmin(array)
    return wea_index


# file = h5py.File('input_double_2015.hdf5', 'r')
# input = np.array(file['features'], dtype='float32')
# target = np.array(file['targets'], dtype='float32')
# uniqueGeoHash= file['uniqueGeo']


date_parser = pd.tseries.tools.to_datetime

df = pd.read_csv('./data/weather/974574.csv', header=0,
usecols=[1, 3, 4, 5, 6, 8, 12, 13, 15, 16, 18],
parse_dates=[3],
names=['STATION_NAME', 'LAT', 'LONG', 'DATE', 'RAIN',
'SNOW', 'LSNOW', 'FFOG', 'HFOG', 'RaS', 'THUN'])

# STATION_NAME, LATITUDE, LONGITUDE, DATE, PRCP, SNOW,
# WT09, WT01, WT02, WT04, WT03

names = np.array(df['STATION_NAME'])
m, n = np.unique(names, return_counts=True)

df_0 = df[df['STATION_NAME']==m[0]]
df_1 = df[df['STATION_NAME']==m[1]]
df_2 = df[df['STATION_NAME']==m[2]]

sta_loca = np.zeros((3, 2), dtype='float32')
sta_loca[0] = np.array([df_0['LAT'].iloc[0], df_0['LONG'].iloc[0]])
sta_loca[1] = np.array([df_1['LAT'].iloc[0], df_1['LONG'].iloc[0]])
sta_loca[2] = np.array([df_2['LAT'].iloc[0], df_2['LONG'].iloc[0]])

df_0.drop(df_0.columns[[0,1,2,3]], inplace=True, axis=1)
dat0 = np.array(df_0)
df_1.drop(df_1.columns[[0,1,2,3]], inplace=True, axis=1)
dat1 = np.array(df_1)
df_2.drop(df_2.columns[[0,1,2,3]], inplace=True, axis=1)
dat2 = np.array(df_2)


# weather = np.zeros((3, 730, 4), dtype=np.float32)
weather = np.zeros((730, 3*4), dtype=np.float32)
# weather_index = wea_location(sta_loca, uniqueGeoHash)
# weather[rain, snow, fog, thunder]

for i in range(730):
    weather[i, 0:4] = convert(dat0[i])
    weather[i, 4:8] = convert(dat1[i])
    weather[i, 8:12] = convert(dat2[i])


f = h5py.File('weather.hdf5', mode='w')
wea = f.create_dataset('wea', weather.shape, dtype='float32')
# wea_index = f.create_dataset('wea_index', weather_index.shape, dtype='int8')

wea[...] = weather
print wea[0:20]
# wea_index[...] = weather_index

f.flush()
f.close()
