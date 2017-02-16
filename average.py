import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


timeprocess = lambda x: x.replace(minute = (x.minute/5) * 5, second=0)
date_parser = pd.tseries.tools.to_datetime

df = pd.read_csv('output.csv', index_col = False, parse_dates=['datetime'], date_parser=date_parser, sep=",")

df['Pickup_datetime'] = pd.Series(df['datetime'].apply(timeprocess), index = df.index)

data = df[['Pickup_datetime', 'ori_num_pickups', 'pre_num_pickups']]

print data.shape
startDateTime = datetime.datetime(2015, 07, 01, 0, 30, 0)
endDateTime = datetime.datetime(2015, 07, 02, 0, 30, 0)
sampleDataTime = (endDateTime - startDateTime)

timestamp_length = datetime.datetime(2015, 07, 01, 1, 00, 0) - startDateTime
print timestamp_length
numdays = 7
# start = data[data['datetime'] == startDateTime]

all_average = []

numtimestampes = 48
for k in range(numtimestampes):
    average = np.zeros((8999))
    startDateTime = startDateTime + timestamp_length
    for i in range(numdays):
        currentDataTime = startDateTime + i * sampleDataTime
        print currentDataTime
        data_eachday = data[data['Pickup_datetime'] == currentDataTime]
        print data_eachday.shape
        average = average + np.asarray(data_eachday['ori_num_pickups'])
    average = average / (numdays + 1)
    all_average.append(average)

all_real_pickups = np.array(data['ori_num_pickups'])
all_pred_pickups = np.array(data['pre_num_pickups'])
print all_real_pickups.shape
y_pred = []
y_aver = []
for i in range(48):
    real_pickups = all_real_pickups[i*8999:(i+1)*8999]
    pred_pickups = all_pred_pickups[i*8999:(i+1)*8999]
    y_pred.append(np.sqrt(mean_squared_error(real_pickups, pred_pickups)))
    y_aver.append(np.sqrt(mean_squared_error(real_pickups, all_average[i])))


t = np.arange(0., 48., 1)
plt.plot(y_aver, 'bs', y_pred, 'r--')
plt.show()
