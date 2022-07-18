import serial
import time
import csv
from kalman_filter import KalmanFilter
import numpy as np

# booker's elbow to sensor bottom: 6.7 
# booker's elbow to sensor top: 4.15

# configuration of Serial communication
serialPort = serial.Serial( port = "COM7", 
                            baudrate=115200,
                            bytesize=8, # number of data bits
                            timeout=2, # just wait 2 seconds
                            stopbits=serial.STOPBITS_ONE)

# initilize kalman filter
kalman_pos = KalmanFilter(x_est0=np.array([0.,0.]),n_obs=1, deltaT=0.01, sigmaR=1e-3, sigmaQ=1e-2)
counter=0


"""
# create .csv and store data
file = open('data.csv', 'w', newline='')
csv_writer = csv.writer(file)
# add headers
header = ['data_'+str(i) for i in range(1,3)]
header += ['time']
csv_writer.writerow(header)
print(f"creating file ...")
"""

try:
    while counter<100000:    
        # recieve data
        t1 = time.time()
        data = serialPort.readline()
        data = data.decode().strip()#.replace(b'\n',b'').replace(b'\r',b'')
        data = int(data)
        #data = serialPort.read(10)
        # time
        t2 = time.time()
        # save data
        #csv_writer.writerow(data)
        dt = 1000*(t2-t1)
        if dt > 0 and counter>5 and counter%100==0:
            #print(f"dt: {dt}, data: {data}")
            pos, vel = kalman_pos.run_kalman_filter(q=data, new_deltaT=dt)
            print(f"dt: {dt:.2f}, data: {data}, pos: {pos:.2f}, vel: {vel:.2f} ")
        counter +=1

    serialPort.close()
    print(f"closing ports ...")
except:
    serialPort.close()
    print(f"closing ports ...")
