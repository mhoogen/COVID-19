import numpy as np
import math
import datetime as dt


patient_id = 1000
stay_id = 1
patients = 100
time_points_mu = 100
time_points_sigma = 20
start_date = dt.datetime(2020, 5, 17)
granularity_minutes = 15 # minutes

f = open("../../data/fake_data.csv", "w")
f.write("patient_id,stay_id,time,attr_1,attr_2,attr_3,attr_4,label\n")

for i in range(0, patients):
    number_time_points = int(np.random.normal(time_points_mu, time_points_sigma, 1)[0])

    for j in range(0, number_time_points):
        timestamp = start_date + j * dt.timedelta(minutes = granularity_minutes)
        f.write(str(patient_id + i) + ', ')
        f.write(str(stay_id) + ', ')
        f.write(timestamp.strftime("%Y-%m-%d %H:%M:%S") + ',')
        f.write(str(np.random.normal(10, 2, 1)[0]) + ',')
        f.write(str(np.random.normal(2, 0.5, 1)[0]) + ',')
        f.write(str(np.random.normal(20, 2, 1)[0]) + ',')
        f.write(str(np.random.normal(5, 1, 1)[0]) + ',')
        f.write(str(i%2) + '\n')

f.close()
