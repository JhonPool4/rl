
import torch as T
import numpy as np
import random


def shuffle_tensor(tensor,batch_size):
    '''shuffle a tensor (so we can have full range instead of random'''
    tens_list = tensor.tolist()
    random.shuffle(tens_list)
    return T.tensor(tens_list,dtype=T.float).reshape(batch_size,1)
def get_min_max_cartesian(l1,l2,_MIN_LIST,_MAX_LIST):
    with T.no_grad():
        batch_size =100000
        q1 = T.linspace(_MIN_LIST['q1']['pos'],_MAX_LIST['q1']['pos'],batch_size).reshape(batch_size,1)
        q2 = T.linspace(_MIN_LIST['q2']['pos'],_MAX_LIST['q2']['pos'],batch_size).reshape(batch_size,1)
        dq1 = T.linspace(_MIN_LIST['q1']['vel'],_MAX_LIST['q1']['vel'],batch_size).reshape(batch_size,1)
        dq2 = T.linspace(_MIN_LIST['q2']['vel'],_MAX_LIST['q2']['vel'],batch_size).reshape(batch_size,1)
    
        x = l1*T.sin(q1) + l2*T.sin(q1+q2)
        y = -l1*T.cos(q1) - l2*T.cos(q1+q2)
        dx = l1*T.cos(q1)*dq1 + l2*T.cos(q1+q2)*(dq1+dq2)
        dy = l1*T.sin(q1)*dq1 + l2*T.sin(q1+q2)*(dq1+dq2)
        x_range = (min(x),max(x))
        y_range = (min(y),max(y))
        dx_range = (min(dx),max(dx))
        dy_range = (min(dy),max(dy))
    return x_range,y_range,dx_range,dy_range

def normalize_inputs(x,y,dx,dy,x_range,y_range,dx_range,dy_range):
    with T.no_grad():

        x_diff = T.abs(x_range[1]-x_range[0])
        x_min = x_range[0]
        x_norm = (x-x_min)/x_diff

        y_diff = T.abs(y_range[1]-y_range[0])
        y_min = y_range[0]
        y_norm = (y-y_min)/y_diff

        dx_diff = T.abs(dx_range[1]-dx_range[0])
        dx_min = dx_range[0]
        dx_norm = (dx-dx_min)/dx_diff

        dy_diff = T.abs(dy_range[1]-dy_range[0])
        dy_min = dy_range[0]
        dy_norm = (dy-dy_min)/dy_diff
    return x_norm,y_norm,dx_norm,dy_norm

def test(nn,x_range,y_range,dx_range,dy_range,threshold = 2,position = True,min_batch=100,verbose=False):
    max_batch= min_batch+1
    with T.no_grad():
        if position:
            print()
            print('POSITION TEST:')
            observation,expected_output = get_rand_obs(x_range,y_range,dx_range,dy_range,min_batch=min_batch,max_batch=max_batch)
            observation=observation
            expected_output=expected_output
        
        prediction = nn(observation)
        loss = 0.5*(expected_output-prediction)**2
        if verbose:
            print('Observation = ',observation)
            print('Prediction: ',prediction)
            print("Expected: ", expected_output)
            print("Loss matrix" ,loss)
        below_thresh = 0
        for i,l in enumerate(loss):
          if l.mean().item() <threshold:
            below_thresh +=1
        loss_max = T.max(loss).item()
        loss_min = T.min(loss).item()
        loss_mean = loss.mean().item()
        #print('Prediction:',prediction)
        #print('Expected:',expected_output)
        #print('Loss:',loss)
        if not verbose:
            print(f'loss_mean: {round(loss_mean,3)}, loss_max= {round(loss_max,3)}, loss_min = {round(loss_min,3)}, #_below_2 = {below_thresh}')
        else:
            print(f'loss_mean: {round(loss_mean,3)}')

        return loss_mean,loss_max,loss_min,below_thresh


def get_rand_obs(x_range,y_range,dx_range,dy_range,min_batch=250,max_batch=750,position = True):
    _MAX_LIST = {'q1':{'pos':np.deg2rad(180), 'vel': np.deg2rad(180)},
                'q2':{'pos':np.deg2rad(130), 'vel': np.deg2rad(180)}}

    _MIN_LIST = {'q1':{'pos':np.deg2rad(-90), 'vel': np.deg2rad(-180)},
                'q2':{'pos':np.deg2rad(0), 'vel': np.deg2rad(-180)}}

    batch_size = random.randint(min_batch,max_batch)
    #print(f'NEW BATCH SIZE:{batch_size} ')
    with T.no_grad(): 
        # training data (output): joint data
        q1_ordered = T.linspace(_MIN_LIST['q1']['pos'],_MAX_LIST['q1']['pos'],batch_size).reshape(batch_size,1)
        q2_ordered = T.linspace(_MIN_LIST['q2']['pos'],_MAX_LIST['q2']['pos'],batch_size).reshape(batch_size,1)
        dq1_ordered = T.linspace(_MIN_LIST['q1']['vel'],_MAX_LIST['q1']['vel'],batch_size).reshape(batch_size,1)
        dq2_ordered = T.linspace(_MIN_LIST['q2']['vel'],_MAX_LIST['q2']['vel'],batch_size).reshape(batch_size,1)

        # all in radians
        q1 = shuffle_tensor(q1_ordered,batch_size)
        q2 = shuffle_tensor(q2_ordered,batch_size)
        dq1 = shuffle_tensor(dq1_ordered,batch_size)
        dq2 = shuffle_tensor(dq2_ordered,batch_size)
        l1 = 330 # mm
        l2 = 240 # 
        x = l1*T.sin(q1) + l2*T.sin(q1+q2)
        y = -l1*T.cos(q1) - l2*T.cos(q1+q2)
        dx = l1*T.cos(q1)*dq1 + l2*T.cos(q1+q2)*(dq1+dq2)
        dy = l1*T.sin(q1)*dq1 + l2*T.sin(q1+q2)*(dq1+dq2)
        x_norm,y_norm,dx_norm,dy_norm=normalize_inputs(x,y,dx,dy,x_range,y_range,dx_range,dy_range)
 
        expected_output = T.rad2deg(T.cat((q1, q2), dim=1))
        observation = T.cat((x_norm, y_norm,dx_norm,dy_norm), dim=1)
            
        return observation,expected_output

def get_rand_obs_vel(dq1_ordered,dq2_ordered,x_range,y_range,dx_range,dy_range,x,y,dx,dy,batch_size = 500,pos_nn = None):
    _MAX_LIST = {'q1':{'pos':np.deg2rad(180), 'vel': np.deg2rad(180)},
                'q2':{'pos':np.deg2rad(130), 'vel': np.deg2rad(180)}}

    _MIN_LIST = {'q1':{'pos':np.deg2rad(-90), 'vel': np.deg2rad(-180)},
                'q2':{'pos':np.deg2rad(0), 'vel': np.deg2rad(-180)}}

    with T.no_grad(): 
        # all in radians
        batch_index = random.sample(len(x_batch),batch_size)
        dq1_expected = dq1_ordered[batch_index]
        dq2_expected = dq2_ordered[batch_index]
        x_batch = x[batch_index]
        y_batch = y[batch_index]
        dx_batch = dx[batch_index]
        dy_batch = dy[batch_index]
       
        x_norm,y_norm,dx_norm,dy_norm=normalize_inputs(x_batch,y_batch,dx_batch,dy_batch,x_range,y_range,dx_range,dy_range)
        pos_observation = T.cat((x_norm, y_norm,dx_norm,dy_norm), dim=1)

        pos_nn_output = pos_nn(pos_observation)
        q1_estim = pos_nn_output[:,0].reshape(len(pos_nn_output[:,1]),1)
        q2_estim = pos_nn_output[:,1].reshape(len(pos_nn_output[:,1]),1)

        q1_min = T.tensor(_MIN_LIST['q1']['pos'],dtype=T.float)
        q1_max = T.tensor(_MAX_LIST['q1']['pos'],dtype=T.float)
        q1_diff = T.abs(q1_max-q1_min)
        q1_norm = (q1_estim-q1_min)/q1_diff

        q2_min = T.tensor(_MIN_LIST['q2']['pos'],dtype=T.float)
        q2_max = T.tensor(_MAX_LIST['q2']['pos'],dtype=T.float)
        q2_diff = T.abs(q2_max-q2_min)
        q2_norm = (q2_estim-q2_min)/q2_diff

        expected_output = T.rad2deg(T.cat((dq1_expected, dq2_expected), dim=1))
        observation = T.cat((q1_norm, q2_norm), dim=1)
            
        return observation,expected_output