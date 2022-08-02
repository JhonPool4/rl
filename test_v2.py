import serial
import time
import csv
from test_v4 import accurate_delay
from utils import KalmanFilter
from FES import Rehamove
import numpy as np
from gym import spaces
from rl_utils import GaussianPolicyNetwork
from rl_utils import print_info
from utils import accurate_delay
import torch
import os

_GOALS ={'state':True, 'upper':0.6, 'lower':0.1}

_MAX_BICEP = 30
_MIN_BICEP = 6

_MAX_TRICEP = 25
_MIN_TRICEP = 6

_ARM_RADIUS =0.3
_MASS = 5

_PULSE_WIDTH = 300


class FesRL:
    def __init__(self, 
                    portSensor='COM7', 
                    portRehamove='COM5', 
                    baudrate=1000000,
                    save_path='./tests',
                    file_name='exp_1'):
        print(f"============================================")
        print(f"\tInitializing test")
        print(f"============================================")                        
        # create serial communication
        self.serialPort = serial.Serial(port=portSensor, 
                                        baudrate=baudrate,
                                        bytesize=8, # number of data bits
                                        timeout=2, # just wait 2 seconds
                                        stopbits=serial.STOPBITS_ONE)
        # time to initilize arduino code
        time.sleep(2)
        print_info(f"connected to sensor in port {portSensor}")

        # initilize encoder
        init_dt = self.init_encoder(samples=10)
        print_info(f"mean dt: {init_dt:.2f}")    

        # create kalman filter
        self.kalman_pos = KalmanFilter(x_est0=np.array([0.,0.]),n_obs=1, deltaT=init_dt, sigmaR=1e-3, sigmaQ=1e-3)

        # create rehamove object
        self.r = Rehamove(port_name=portRehamove)            # Open USB port (on Windows)]

        # create rl action space
        self.action_space = spaces.Box(low=np.zeros((2,), dtype=np.float32),
                                    high=np.ones((2,), dtype=np.float32),
                                    shape=(2,))           
        # create rl agent 
        self.agent = GaussianPolicyNetwork(8, 2, self.action_space, hidden_layer=(64,32,16),lr=1e-4)
        
        # load agent parameters
      
        net_path = './trained_models/Arm1DEnv_FES_brach_freq_1_nn_64_32_16_v3/sac/agent_parameters/7450'
        self.agent.load_state_dict(torch.load(os.path.join(net_path,'pi_net_sac')))        
        print_info(f"loaded agent parameters")

        # create directory to save experimental results
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print_info(f"creating test directory: {save_path}")       

        self.data_path = os.path.join(save_path, file_name)
        self.column_names =  ['pos', 'vel', 'radius', 'mass', 'torque', 'tri', 'bi', 'des_pos']
        with open(self.data_path, 'w',newline='') as f:
            # create the csv writer
            csv_writer = csv.writer(f)
            # write header
            csv_writer.writerow(self.column_names)
        print_info(f"creating new test file")  

        with open(os.path.join(save_path, 'info'), 'w',newline='') as f:
            # create the csv writer
            csv_writer = csv.writer(f)
            # write header
            csv_writer.writerow(['min_bi: '+ str(_MIN_BICEP)])
            csv_writer.writerow(['max_bi: '+ str(_MAX_BICEP)])
            csv_writer.writerow(['min_tri: '+ str(_MIN_TRICEP)])
            csv_writer.writerow(['max_tri: '+ str(_MAX_TRICEP)])
            csv_writer.writerow(['arm_radius: '+ str(_ARM_RADIUS)])
            csv_writer.writerow(['mass: '+ str(_MASS)])
            csv_writer.writerow(['pulse_width '+ str(_PULSE_WIDTH)])

        print_info(f"creating new info file")           

        # writer handle
        self.open_csv_handle()             


    def open_csv_handle(self):
        # file and csv handle
        self.f = open(self.data_path, 'a', newline='')
        self.csv_writer = csv.writer(self.f)

    def close_csv_handle(self):
        # close file
        self.f.close()        

    def save_data(self, obs):
        # save training data
        self.csv_writer.writerow(obs)                 

    def apply_fes(self, current=0, channel="blue", duration=100):
        """
        @inputs:
        --------
        - channel: could be read or blue
        - current: intensity in milliAmperes (mA)
        - duration: length of the pulse in microSeconds (uS) 
        """
        self.r.pulse(channel,current, duration)     # units: r.pulse("COLOR OF STIM OUTPUT", CURRENT IN mA, PULSE DURATION in microseconds)
    
    def convert_nn_to_current(self, muscle_min, muscle_max, nn_muscle_activation_output):
        '''
        muscle_min = int value for min possible current to apply to muscle
        muscle_max = int value for max possible current to apply to muscle
        nn_muscle_activation_output = value between 0 and 1

        '''
        diff_between_max_and_min = abs(muscle_max-muscle_min)
        percentage_of_diff = nn_muscle_activation_output * diff_between_max_and_min 
        return muscle_min + percentage_of_diff


    def request_data(self, is_negative=False):
        """
        @outputs:
        ---------
        - dt: sampling time in seconds (s)
        - data: angular position (rad)
        """
        # reference time
        t1 = time.time()
        # recieve data and represent in radians
        self.serialPort.write('r'.encode('utf-8'));
        data = self.serialPort.readline()
        data = data.decode().strip()
        data = int(data)*np.pi/1024
        # sampling time
        t2 = time.time()
        dt = t2-t1
        if is_negative:
            return dt, -data
        return dt, data

   


    def init_encoder(self, samples=10):
        dt_vector = np.zeros((samples,1))
        k = 0
        while k<samples:
            dt, _ = self.request_data()
            if dt>0 and dt<10:
                dt_vector[k,] = dt
                k +=1

        return dt_vector.mean()

    def end_communication(self):
        self.serialPort.close()
        print(f"ports closed safety")
        self.close_csv_handle()
        print(f"data file closed")




# main looop
fes_rl = FesRL(portSensor='COM7', portRehamove='COM9',file_name='exp_1')
act = torch.tensor([0,0], dtype=torch.float)
cycles = 100
tt = 0.5
time_list = [i*tt for i in range(1,cycles+1)]
max_sim_time = tt*cycles
new_counter = 0
old_counter = 0
goal = _GOALS['upper']
sim_time = 0
counter= 0 
pulse_freq = 50

ttt0 = time.time()
dt_fes = 0
try:
    while sim_time < max_sim_time:
        dt, angle = fes_rl.request_data(is_negative=True)

        if dt > 0: # to avoid false positive
            # reference timer
            t0 = time.time()


            # update simulation time
            sim_time += dt

            # just to print
            counter +=1

            # get goal
            if sim_time>time_list[new_counter]:
                new_counter += 1
            
            if new_counter>old_counter:
                old_counter = new_counter
                _GOALS['state']= not _GOALS['state']
                goal = _GOALS['upper'] if _GOALS['state'] else _GOALS['lower']
            
                #print_info(f"nuevo cambio!")
                #print_info(f'goal= {goal}, time = {round(sim_time,2):.2f}')                


            
            # estimate angular pos and vel
            pos, vel = fes_rl.kalman_pos.update(q=angle, new_deltaT=dt)
       
            # torque
            torque = _ARM_RADIUS*np.sin(pos)*_MASS*9.81
                        
            # normalization
            norm_pos = (pos)/np.deg2rad(150)
            norm_vel = (vel+np.pi)/(2*np.pi)
            norm_torque = torque/51.5


            # create obs vector
            # obs = [pos, vel, radius, mass, torque, tri, bi, des_pos]
            obs = torch.tensor([norm_pos, norm_vel, 0.5, 0.5, norm_torque, act[0], act[1], goal], dtype=torch.float)
            # save observations
            fes_rl.save_data([norm_pos, norm_vel, 0.5, 0.5, norm_torque, act[0].item(), act[1].item(), goal])


            # predict action
            act, _ = fes_rl.agent.predict_action(obs)

            # apply electric stimulation                               
            act_tricep = fes_rl.convert_nn_to_current(muscle_min=_MIN_TRICEP, 
                                                        muscle_max=_MAX_TRICEP, 
                                                        nn_muscle_activation_output= act[0].item()) 
            act_bicep = fes_rl.convert_nn_to_current(muscle_min=_MIN_BICEP, 
                                                        muscle_max=_MAX_BICEP, 
                                                        nn_muscle_activation_output= act[1].item())  
            # second timer: update sim_time
            loop_time = time.time() - t0  
            sim_time += loop_time

            
            #fes_rl.apply_fes(current=0, channel="red", duration=40*1000000)
            accurate_delay(20 - 1000*(loop_time+ dt+ dt_fes))
            tt0 = time.time()

            if counter%50==0:
                #print(f"dt sleep: {1000*(time.time()-tt0)} ms")
                #print(f"sim_time: {1000*sim_time:.2f}, pos: {np.rad2deg(angle):.2f}, kalman: {np.rad2deg(pos):.2f}")                 
                print(f"loop_time: {1000*loop_time:.2f} ms")
                print(f"sensor_time: {1000*dt:.2f} ms")
                print(f"fes_time: {1000*dt_fes:.2f} ms")
                print(f"freq: {1/(time.time() - t0 + dt + dt_fes)} Hz")
                
                print(f" ")

            act_bicep = 25 # mA
            
            fes_rl.apply_fes(current=act_bicep, channel="red", duration=_PULSE_WIDTH)

            
            dt_fes = time.time() - tt0
            sim_time += dt_fes
            #fes_rl.apply_fes(current=act_tricep, channel="blue", duration=_PULSE_WIDTH)

except:
    print(f"total time: {time.time()-ttt0}")
    print(f"counter: {counter}")
    fes_rl.end_communication()    