import socket
import struct 
import csv
import time
import numpy as np
# API
_START_MESSAGE='START\r\n\r\n'

class DelsysSocket():
    def __init__(self, TRIGNO_HOST=50040, PORT_EMG=50041, PORT_ACC=50042, save_data=False, sensors={'EMG':False,'ACC':True}):
        # recieve parameters
        self.save_data = save_data
        self.sensors = sensors


        # establish connection between cpu (client) and delsys (server)
        self.delsys_socket = socket.create_connection(('localhost', TRIGNO_HOST))
        
        # AF_INET = set the internet family ipv4
        # SOCK_STREAM = set TCP protocol
        if self.sensors['EMG']:           
            # create a server to recieve delsys data
            self.emg_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # establish connection between cpu (sever) and delsys (client)
            self.emg_socket.connect(('localhost', PORT_EMG))
        if self.sensors['ACC']:
            # create a server to recieve delsys data
            self.acc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # establish connection between cpu (sever) and delsys (client)
            self.acc_socket.connect(('localhost', PORT_ACC))

        # confirmation message: connection established
        self.recieve_confirmation_message()
        # start communication
        self.delsys_socket.sendall(bytes(_START_MESSAGE, 'utf-8'))
        # confirmation message: start communication
        self.recieve_confirmation_message()

        # create .csv and store data
        if self.save_data:
            self.file = open('data.csv', 'w', newline='')
            self.csv_writer = csv.writer(self.file)
            # add headers
            header=[]
            if self.sensors['EMG']:  
                header += ['emg_'+str(i) for i in range(1,17)]
            if self.sensors['ACC']:                
                header += ['acc_'+str(i)+'_'+axis for i in range(1,17) for axis in ['x','y','z']]
            self.csv_writer.writerow(header)
            print(f"creating file")

    def recieve_confirmation_message(self):
        data = self.delsys_socket.recv(1024)
        print(f"message recieved: {data}")

    def recieve_data(self, is_calibrating=False):
        if self.sensors['EMG']:
            emg_raw_data = self.emg_socket.recv(16*4)
            #print(f"emg: {emg_raw_data}")   
        if self.sensors['ACC']:
            acc_raw_data = self.acc_socket.recv(16*4*3)
            #print(f"acc: {acc_raw_data}")
        data = []
        if self.sensors['EMG']:
            data += list(struct.unpack("<16f", emg_raw_data))
        if self.sensors['ACC']:
            data += list(struct.unpack("<48f", acc_raw_data))

        if self.save_data and not is_calibrating:
            self.csv_writer.writerow(data)
        
        return data

    def init_accel_calibration(self, n_samples=10, device=16):
        print(f"calibratring: ...")
        
        dt_vector = np.zeros((n_samples, 1))
        data_matrix = np.zeros((n_samples, 16, 3))
        
        #for k in range(n_samples):
        k=0
        while k<n_samples:
            t0 = time.time()
            accel_data = self.recieve_data(is_calibrating=True)
            accel_data = np.array(accel_data).reshape((16, 3))
            dt = time.time() - t0
            if dt>1e-3 and all(accel_data[device-1,:])!=0:
                dt_vector[k,] = dt
                data_matrix[k,:, :] = accel_data

                #print(f"/niter: {k}")
                #print(f"(i) sampling time: {dt_vector[k]}")
                #print(f"(ii) initial accel value: {data_matrix[k,device-1,:]}")                
                k +=1

        print(f"mean values")
        print(f"(i) sampling time: {dt_vector.mean()}")
        print(f"(ii) initial accel value: {data_matrix.mean(axis=0)[device-1,:]}")


        return dt_vector.mean(), data_matrix.mean(axis=0)

    def end_communication(self):
        self.delsys_socket.close()
        if self.sensors['EMG']:
            self.emg_socket.close()
        if self.sensors['ACC']:
            self.acc_socket.close()            
        if self.save_data:
            self.file.close()