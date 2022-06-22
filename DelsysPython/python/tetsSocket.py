import socket
import struct
import csv
import time
import tetsSocketCMD as cmd

import numpy as np

# https://delsys.com/downloads/USERSGUIDE/trigno/sdk.pdf
HOST = "127.0.0.1" 
PORT_EMG = 50041 
PORT_ACC = 50042
print('Iniciado')
f =  open('data.csv', 'w',newline='')
csv_writer = csv.writer(f)

t1 = [0,0]
t2 = [0,0]
t3 = [0,0]

try:
    
    s_emg = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_acc =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)    
    s_emg.connect((HOST, PORT_EMG))
    s_acc.connect((HOST, PORT_ACC)  )

    print('Guardando datos...')
    
    for i in range(1000):
        
           

        t1[0] += time.time()
        #data_emg = s_emg.recv(16*4)
        data_acc = s_acc.recv(16*4*3)
        t1[1] += time.time()
    
        t2[0] += time.time()
        #data = (struct.unpack("<16f",data_emg))
        data = (struct.unpack("<%df"%(16*3),data_acc))
        t2[1] += time.time()

         

        t3[0] += time.time()
        csv_writer.writerow(data)
        t3[1] += time.time()
    
           
                #print(f"Received {data!r}")
except:
    f.close()
    print('Error')

print('Finalizado')

s_emg.close()
s_acc.close()

cmd.sock.close()
f.close()

print("El tiempo 1 es %.5f"%(t1[1]-t1[0]))
print("El tiempo 2 es %.5f"%(t2[1]-t2[0]))
print("El tiempo 3 es %.5f"%(t3[1]-t3[0]))