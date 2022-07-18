from numpy import diff
from rehamove import *          # Import our library
import time
from random import uniform
import serial
import numpy as np
from gym import spaces
from rl_utils import rl_networks
import torch
import os

r = Rehamove("COM5")            # Open USB port (on Windows)]
def apply_fes(current=0, channel="blue", duration=100):

    freq = 30000
    for i in range(0, duration):
        r.pulse(channel,current, freq)     # units: r.pulse("COLOR OF STIM OUTPUT", CURRENT IN mA, PULSE DURATION in microseconds)
def convert_nn_to_current(muscle_min,muscle_max,nn_muscle_activation_output):
    '''
    muscle_min = int value for min possible current to apply to muscle
    muscle_max = int value for max possible current to apply to muscle
    nn_muscle_activation_output = value between 0 and 1

    '''
    diff_between_max_and_min = abs(muscle_max-muscle_min)
    percentage_of_diff = nn_muscle_activation_output * diff_between_max_and_min 
    return muscle_min + percentage_of_diff


action_space = spaces.Box(low=np.zeros((2,), dtype=np.float32), \
                                        high=np.ones((2,), dtype=np.float32), \
                                        shape=(2,))    
agent = rl_networks.GaussianPolicyNetwork(8, 2,action_space, hidden_layer=(64,32,16),lr=1e-4)
agent.load_state_dict(torch.load(os.path.join('pi_net_sac')))

_GOALS ={'state':True, 'upper':1, 'lower':0}

max_bicep = 15#15
min_bicep = 5#5

max_tricep = 15
min_tricep = 5
arm_radius =0.3
mass = 5




serialPort = serial.Serial(port = "COM11", baudrate=20000, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
cycles = 4
time_list = [i*3 for i in range(1,cycles+1)]
max_time = 3*cycles
new_counter = 0
current_time = 1
old_counter = 0
serialString = ""  
obs = torch.tensor([0,0,0,0.5,0.5,0,0,0],dtype=torch.float32)
goal = _GOALS['upper']
previous_angle = 10
loop_time = .01
current_angle = 0
old_time = 0
start = time.time()
current = time.time()
print(f'goal= {goal}, time = {round(current_time,2):.2f}')
while current-start < max_time:
    loop_start_time = time.time()
    current_time =current-start 
    actions,_ = agent.predict_action(obs) 
    bicep_nn_val = actions[1].item()
    tricep_nn_val =actions[0].item()
    converted_signal_bicep = convert_nn_to_current(min_bicep,max_bicep,bicep_nn_val)
    converted_signal_tricep = convert_nn_to_current(min_bicep,max_bicep,tricep_nn_val)
    #print(f"bicep = {bicep_nn_val}, tricep = {tricep_nn_val}")
    #print(f"converted_signal_bicep = {converted_signal_bicep}, converted_signal_tricep = {converted_signal_tricep}")
    # Wait until there is data waiting in the serial buffer
    serialString = serialPort.readline()
    #print('bytes',serialString)
    serialString=serialString.decode().strip()
    #print('raw',serialString)
    try:
        previous_angle = current_angle
        current_angle =float(serialString)
        current_vel = (current_angle-previous_angle)/(loop_time)
        normalized_vel = (current_vel+180)/360
        #print(current_angle,previous_angle,current_vel,normalized_vel)
        normalized_angle_input = (current_angle-10)/(150-10)
        #print(f"bicep = {bicep_nn_val}, tricep = {tricep_nn_val},vel = {current_vel}",current_angle,previous_angle)

    except:
        print('current serial string: ', serialString)
        print("Serial output cant be converted to float")
        normalized_angle_input = 0
        normalized_vel = 0
    current_torque = arm_radius*np.sin(np.deg2rad(current_angle))*mass*9.81
    normalized_torque = current_torque/51.5
    #print(current_torque,normalized_torque)  normalized_angle_input  normalized_vel
    new_obs = torch.tensor([normalized_angle_input,normalized_vel,0.5,0.5,normalized_torque,tricep_nn_val,bicep_nn_val,goal],dtype=torch.float32)
    obs = new_obs
    print(f'obs = {obs}')
    #print(f"biceps: {converted_signal_bicep:.2f}, triceps: {converted_signal_tricep:.2f}")
    #apply action to bicep
    #apply_fes(current=converted_signal_bicep, channel="red", duration=1)
    #apply action to tricep
    #apply_fes(current=converted_signal_tricep,  channel="blue", duration=1)
    


    if current_time>= time_list[new_counter]:
        new_counter += 1
            

    if new_counter>old_counter:
        old_counter = new_counter
        _GOALS['state']= not _GOALS['state']
        if _GOALS['state']:
            goal = _GOALS['upper']
        else:
            goal = _GOALS['lower']
        print(f"nuevo cambio!")
        print(f'goal= {goal}, time = {round(current_time,2):.2f}')
        
    
    old_time = current_time
    current = time.time()
    #print(current-loop_start_time)
   




# Pruebas brazo derecho bicep(Booker)
# 17 mA de 30000
# 12 mA de 30000
# 100 pulsos
