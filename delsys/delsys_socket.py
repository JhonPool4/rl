import socket
import struct 
import csv

# API
_START_MESSAGE='START\r\n\r\n'

class DelsysSocket():
    def __init__(self, TRIGNO_HOST=50040, PORT_EMG=50041, PORT_ACC=50042, save_data=False):
        
        # establish connection between cpu (client) and delsys (server)
        self.delsys_socket = socket.create_connection(('localhost', TRIGNO_HOST))
        # create a server to recieve delsys data
        # AF_INET = set the internet family ipv4
        # SOCK_STREAM = set TCP protocol          
        self.emg_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.acc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # establish connection between cpu (sever) and delsys (client)
        self.emg_socket.connect(('localhost', PORT_EMG))
        self.acc_socket.connect(('localhost', PORT_ACC))

        # confirmation message: connection established
        self.recieve_confirmation_message()
        # start communication
        self.delsys_socket.sendall(bytes(_START_MESSAGE, 'utf-8'))
        # confirmation message: start communication
        self.recieve_confirmation_message()

        # create .csv and store data
        self.save_data = save_data
        if self.save_data:
            self.file = open('data.csv', 'w', newline='')
            self.csv_writer = csv.writer(self.file)
            # add headers
            header = ['emg_'+str(i) for i in range(1,17)]
            header += ['acc_'+str(i)+'_'+axis for i in range(1,17) for axis in ['x','y','z']]
            self.csv_writer.writerow(header)

    def recieve_confirmation_message(self):
        data = self.delsys_socket.recv(1024)
        print(f"message recieved: {data}")

    def recieve_emg_data(self):
        emg_raw_data = self.emg_socket.recv(16*4)
        acc_raw_data = self.acc_socket.recv(16*4*3)
        data = struct.unpack("<16f", emg_raw_data)
        data += struct.unpack("<48f", acc_raw_data)
        if self.save_data:
            self.csv_writer.writerow(data)


    def end_communication(self):
        self.delsys_socket.close()
        self.emg_socket.close()
        if self.save_data:
            self.file.close()


emg_handle = DelsysSocket(save_data=True)

try:
    while True:
        emg_handle.recieve_emg_data()

except:
    emg_handle.end_communication()
    print(f"closing sockets ...")