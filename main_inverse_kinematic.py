from turtle import pos
import torch as T
import torch.nn as nn
import numpy as np
import random
import os
from rl_utils import get_min_max_cartesian
from rl_utils import get_rand_obs
from rl_utils import test
from rl_utils import get_rand_obs_vel
from torch.utils.tensorboard import SummaryWriter

train_position = False
load_pos = False
load_vel = False
""" 
if train_position:
    net_model_pos = nn.Sequential(nn.Linear(in_features=4, out_features=64), nn.ReLU(),
                            nn.Linear(in_features=64, out_features=16), nn.ReLU(),
                            nn.Linear(in_features=16, out_features=2))
    learning_rate = 1e-3
    net_optim_pos = T.optim.Adam(params=net_model_pos.parameters(), lr=learning_rate)

    last_epoch=0
    if load_pos:
        last_epoch = 7300000
        load_path_pos=os.path.join(os.getcwd(), 'trained_models', 'inverse_kinematics/position/inverse_kin_pos_model_'+str(last_epoch))
        net_model_pos.load_state_dict(T.load(load_path_pos))
        print(f'Just the position model loaded succesfully at {last_epoch} epochs!')

        # 2D robot model
    l1 = 330 # m
    l2 = 240 # 
    #loss_list = []
    _MAX_LIST = {'q1':{'pos':np.deg2rad(180), 'vel': np.deg2rad(180)},
                    'q2':{'pos':np.deg2rad(130), 'vel': np.deg2rad(180)}}

    _MIN_LIST = {'q1':{'pos':np.deg2rad(-90), 'vel': np.deg2rad(-180)},
                'q2':{'pos':np.deg2rad(0), 'vel': np.deg2rad(-180)}}
    x_range,y_range,dx_range,dy_range = get_min_max_cartesian(l1,l2,_MIN_LIST,_MAX_LIST) #for normalization
    writer = SummaryWriter()
    n_epochs = 20000000
    losses = []
    observation_pos,expected_output_pos = get_rand_obs(x_range,y_range,dx_range,dy_range,min_batch=375,max_batch=475)
    for epoch in range(1,n_epochs+1):
        epoch += last_epoch
        prediction_pos = net_model_pos(observation_pos)
        loss_pos = 0.5*(expected_output_pos-prediction_pos)**2
        net_optim_pos.zero_grad()
        loss_pos.mean().backward()
        net_optim_pos.step()      
        loss_pos_mean = loss_pos.mean().item()
        losses.append(loss_pos_mean)
        if epoch%10 ==0:
            observation_pos,expected_output_pos = get_rand_obs(x_range,y_range,dx_range,dy_range,min_batch=375,max_batch=475)
        if epoch%5000==0:
            #print(f"ep: {epoch}, position loss: {round(loss_pos_mean,2)}, velocity loss: {round(loss_vel_mean,2)}, vel_batches= {num_batches_vel}")
            print(f"ep: {epoch}, position loss avg: {round(sum(losses[-4999:-1])/len(losses[-4999:-1]),6)}")
        #if epoch%50000 == 0:
            #learning_rate -= learning_rate/2
            #net_optim_pos = T.optim.Adam(params=net_model_pos.parameters(), lr=learning_rate)
            #print(f'new learning rate = {learning_rate}')
        if epoch%50000 ==0:
            print(f"Saving Model......")
            #Verbose test
            loss_mean,loss_max,loss_min,below_thresh=test(net_model_pos,x_range,y_range,dx_range,dy_range,position = True,min_batch=9,verbose=True)
            # out of a hundred test
            loss_mean,loss_max,loss_min,below_thresh=test(net_model_pos,x_range,y_range,dx_range,dy_range,position = True,threshold = 2,min_batch=100)
            save_path_pos=os.path.join(os.getcwd(), 'trained_models', 'inverse_kinematics/position/inverse_kin_pos_model_'+str(epoch))
            T.save(net_model_pos.state_dict(),save_path_pos)
            print('Model saved!')
            print()
            writer.add_scalar(f'PositionTestLoss', loss_mean, epoch)
            print()
        writer.add_scalar(f'PositionLoss', loss_pos_mean, epoch)

else:
    net_model_pos = nn.Sequential(nn.Linear(in_features=4, out_features=32), nn.ReLU(),
                          nn.Linear(in_features=32, out_features=16), nn.ReLU(),
                          nn.Linear(in_features=16, out_features=2))
    last_epoch_pos = 27300000
    load_path_pos=os.path.join(os.getcwd(), 'trained_models', 'inverse_kinematics/position/inverse_kin_pos_model_'+str(last_epoch_pos))
    net_model_pos.load_state_dict(T.load(load_path_pos))

    net_model_vel = nn.Sequential(nn.Linear(in_features=4, out_features=128), nn.GELU(),
                            nn.Linear(in_features=128, out_features=64), nn.GELU(),
                            nn.Linear(in_features=64, out_features=32), nn.GELU(),
                            nn.Linear(in_features=32, out_features=2))
    net_optim_vel = T.optim.Adam(params=net_model_vel.parameters(), lr=1e-5)

    last_epoch=0
    if load_vel:
        last_epoch = 2800000
        load_path_vel=os.path.join(os.getcwd(), 'trained_models', 'inverse_kinematics/velocity/inverse_kin_vel_model_'+str(last_epoch))
        net_model_vel.load_state_dict(T.load(load_path_vel))
    
    # 2D robot model
    l1 = 330 # m
    l2 = 240 # 
    #loss_list = []
    _MAX_LIST = {'q1':{'pos':np.deg2rad(180), 'vel': np.deg2rad(180)},
                    'q2':{'pos':np.deg2rad(130), 'vel': np.deg2rad(180)}}

    _MIN_LIST = {'q1':{'pos':np.deg2rad(-90), 'vel': np.deg2rad(-180)},
                'q2':{'pos':np.deg2rad(0), 'vel': np.deg2rad(-180)}}
    x_range,y_range,dx_range,dy_range = get_min_max_cartesian(l1,l2,_MIN_LIST,_MAX_LIST) #for normalization
    q1_ordered = T.linspace(_MIN_LIST['q1']['pos'],_MAX_LIST['q1']['pos'],1000000).reshape(1000000,1)
    q2_ordered = T.linspace(_MIN_LIST['q2']['pos'],_MAX_LIST['q2']['pos'],1000000).reshape(1000000,1)
    dq1_ordered = T.linspace(_MIN_LIST['q1']['vel'],_MAX_LIST['q1']['vel'],1000000).reshape(1000000,1)
    dq2_ordered = T.linspace(_MIN_LIST['q2']['vel'],_MAX_LIST['q2']['vel'],1000000).reshape(1000000,1)
    x = l1*T.sin(q1_ordered) + l2*T.sin(q1_ordered+q2_ordered)
    y = -l1*T.cos(q1_ordered) - l2*T.cos(q1_ordered+q2_ordered)
    dx = l1*T.cos(q1_ordered)*dq1_ordered + l2*T.cos(q1_ordered+q2_ordered)*(dq1_ordered+dq2_ordered)
    dy = l1*T.sin(q1_ordered)*dq1_ordered + l2*T.sin(q1_ordered+q2_ordered)*(dq1_ordered+dq2_ordered)

    writer = SummaryWriter()
    n_epochs = 20000000000000
    num_batches_pos = 0
    num_batches_vel = 0
    observation_vel,expected_output_vel = get_rand_obs_vel(dq1_ordered,dq2_ordered,x_range,y_range,dx_range,dy_range,q1_ordered,q2_ordered,dq1_ordered,dq2_ordered,batch_size = 500,pos_nn = net_model_pos)
    for epoch in range(1,n_epochs+1):
        epoch += last_epoch
        prediction_vel = net_model_vel(observation_vel)
        loss_vel = 0.5*(expected_output_vel-prediction_vel)**2
        net_optim_vel.zero_grad()
        loss_vel.mean().backward()
        net_optim_vel.step()      
        loss_vel_mean = loss_vel.mean().item()
        num_batches_vel +=1
        observation_vel,expected_output_vel = get_rand_obs_vel(dq1_ordered,dq2_ordered,x_range,y_range,dx_range,dy_range,q1_ordered,q2_ordered,dq1_ordered,dq2_ordered,batch_size = 500,pos_nn = net_model_pos)
        if epoch%5000==0:
            print(f"ep: {epoch}, velocity loss: {round(loss_vel_mean,2)}")
        if epoch%50000 ==0:
            print(f"Saving Model......")
            #vel_test_mean = test(net_model_vel,x_range,y_range,dx_range,dy_range,position=False)
            save_path_vel=os.path.join(os.getcwd(), 'trained_models', 'inverse_kinematics/velocity/inverse_kin_vel_model_'+str(epoch))
            T.save(net_model_vel.state_dict(),save_path_vel)
            print('Models saved!')
            print()
            #writer.add_scalar(f'VelocityTestLoss', vel_test_mean, epoch)
        writer.add_scalar(f'VelocityLoss', loss_vel_mean, epoch)

 """

l1 = 330 # m
l2 = 240 # 
_MAX_LIST = {'q1':{'pos':np.deg2rad(180), 'vel': np.deg2rad(180)},
    'q2':{'pos':np.deg2rad(130), 'vel': np.deg2rad(180)}}

_MIN_LIST = {'q1':{'pos':np.deg2rad(-90), 'vel': np.deg2rad(-180)},
'q2':{'pos':np.deg2rad(0), 'vel': np.deg2rad(-180)}}
x_range,y_range,dx_range,dy_range = get_min_max_cartesian(l1,l2,_MIN_LIST,_MAX_LIST) #for normalization
net_model_pos = nn.Sequential(nn.Linear(in_features=4, out_features=64), nn.ReLU(),
                          nn.Linear(in_features=64, out_features=16), nn.ReLU(),
                          nn.Linear(in_features=16, out_features=2))
last_epoch = 27300000
load_path_pos=os.path.join(os.getcwd(), 'trained_models', 'inverse_kinematics/position/inverse_kin_pos_model_'+str(last_epoch))
net_model_pos.load_state_dict(T.load(load_path_pos,map_location=T.device('cpu')))
print(f'Just the position model loaded succesfully at {last_epoch} epochs!')

loss_mean,loss_max,loss_min,below_thresh=test(net_model_pos,x_range,y_range,dx_range,dy_range,position = True,verbose=True,min_batch=100)
print(f'loss_mean: {loss_mean}, loss_max= {loss_max}, loss_min = {loss_min}, # below 2 = {below_thresh}')
print(below_thresh)