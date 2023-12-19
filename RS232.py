import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#==================================
# data was taken from the oscilloscope
# Model,MSO2014
# Firmware Version,1.35
# Point Format,Y,
# Horizontal Units,S,
# Horizontal Scale,0.04,
# Sample Interval,3.2e-06,
# Filter Frequency,4.2e+07,
# Record Length,125000,
# Gating,0.0% to 100.0%,
# Probe Attenuation,1,
# Vertical Units,V,
# Vertical Offset,0,
# Vertical Scale,5,
# Label,,
# TIME,CH1,CH1 Peak Detect
# Ublock GPS RS232 waveform decode
# Sampath Bandara
# ==================================
baudrate = 9600  # in hz
baudrate_second = 1 / baudrate
Sample_Interval = 1.6e-06  # in second
# Sample_Interval = 3.2e-06
numsample = int(baudrate_second / Sample_Interval)
# ===================================

file_Name = "T0001CH1.CSV"
# file_Name = 'T0000CH1.CSV'
df = pd.read_csv(file_Name, skiprows=15)

df.iloc[df.iloc[:, 1] > 0, 1] = 0
df.iloc[df.iloc[:, 1] < 0, 1] = 1
n0 = df.loc[:, 'CH1'].idxmin()
n1 = np.where(df.loc[:, 'CH1'] == 0)
n1 = n1[0][-1]

x = df.loc[n0:n1 + 1, 'TIME'].to_numpy()
y = df.loc[n0:n1 + 1, 'CH1'].to_numpy(dtype=int)

peaks = np.diff(y)

rdge = np.where(peaks == 1)[0] + 1
fdge = np.where(peaks == -1)[0]

uedge = np.sort(np.append([rdge], fdge))
uedge = uedge[:-1]
ledge = np.sort(np.append([np.append([0], rdge - 1)], fdge + 1))
fedge = np.sort(np.append([uedge], ledge))
edge_array = fedge.reshape(-1, 2)
# print(edge_array.shape[0])

my_array = []
for i in range(edge_array.shape[0]):
    # print(edge_array[i][0])
    edrange = np.arange(edge_array[i][0], edge_array[i][1], 1)
    L = len(edrange)
    extra = L % (numsample - 1)
    if extra:
        alist = y[edrange[0:-extra]]
    else:
        alist = y[edrange]
    alist = np.array(alist, dtype =int).reshape(-1, (numsample - 1))
    aLogic = np.sum(alist, axis=1, dtype=int) / int((numsample - 1))
    # print(aLogic)
    my_array.append(aLogic)

my_array = np.concatenate(my_array).astype(int)

ascii_array  = []
for j in range(0,len(my_array), 10):
    # there are 10 bits for each patten witch include (start bit + 8 bits data + stop bit)
    str_array = ''.join([str(i) for i in my_array[j:j+10]])
    # remove the stop bit and the start bit
    str_array_2 = str_array[1:9]
    # take the reverse order of  the string
    str_array_3 = str_array_2[::-1]
    # first convert to the base 10 value and then convert to the ascii
    ascii_Letter = chr(int(str_array_3, 2))
    ascii_array.append(ascii_Letter)

ascii_array = [''.join(i for i in ascii_array)]
print('Ascii Message is : ----> \n{}'.format(*ascii_array))
ascii_val = re.findall("[+|-]?[0-9]+[.|*]?[0-9]+|[A-Z]+[*]?[0-9]?[A-Z]?",ascii_array[0])

def getTime(ascii_val):
    time_ = float(ascii_val[1])
    lat_ = float(ascii_val[3])
    long_ = float(ascii_val[5])
    hours = int(time_/10000)
    minutes = int((time_ - hours*10000)/100)
    second = time_ - (hours*10000 + minutes*100)
    lat = np.round(lat_/100,7)
    lon = np.round(long_/100,7)

    print('Time in UTC : {}:{}:{}\nLattitude: {}N\nLongitude: {}W'.format(hours,
                                                                          minutes, second, lat, lon))
    return hours, minutes, second,lat,lon

info = getTime(ascii_val)


# plt.figure()
# plt.plot(df.loc[:, 'TIME'], df.loc[:, 'CH1'], '-')
# plt.plot(x[fedge], y[fedge], 'o', markersize=10)
# plt.plot(df.loc[n0:n1, 'TIME'], df.loc[n0:n1, 'CH1'], 'o-')
# plt.show()
