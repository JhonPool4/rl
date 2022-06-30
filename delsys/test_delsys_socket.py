from delsys_socket import DelsysSocket
from kalman_filter import MultipleKalmanDerivator
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

_DEVICE = 14

delsys_handle = DelsysSocket(save_data=True, sensors={'EMG':False, 'ACC':True})
calibration_dt, calibration_accel = delsys_handle.init_accel_calibration(n_samples=20, device=_DEVICE)


#accel_data = delsys_handle.recieve_data()
#accel_data = np.array(accel_data).reshape((16,3))
#calibration_accel_device = calibration_accel[_DEVICE-1,:]

n_dof = 3
# kalman filter: cartesian data
cartesian_kalman = MultipleKalmanDerivator(deltaT=calibration_dt, 
                                            x0=np.zeros(n_dof), 
                                            dx0=np.zeros(n_dof), 
                                            ddx0=np.zeros(n_dof), 
                                            n_obs=1, 
                                            sigmaR = 1e-2, 
                                            sigmaQ = 1e-2)

test_samples=500 
pos_est = np.zeros((3,test_samples))
vel_est = np.zeros((3,test_samples))
acc_est = np.zeros((3,test_samples))


pos_med = np.zeros((3,test_samples))
vel_med = np.zeros((3,test_samples))
acc_med = np.zeros((3,test_samples))

try:


    for k in range(test_samples):
        # raw data
        accel_data = delsys_handle.recieve_data()
        accel_data = np.array(accel_data).reshape((16,3))
        
        accel_device = accel_data[_DEVICE-1,:]-calibration_accel[_DEVICE-1,:]
        #print(f"raw: {accel_data[-1,:]}")
        #print(f"init: {init_accel[-1,:]}")

        #if k%100==0:
        #    print(f"\n")
        #    print(f"sample: {k}")
        #    print(f"accel_device: {9.81*accel_device}")        

        #acc_med[0,k] = copy(accel_device[0])
        #acc_med[1,k] = copy(accel_device[1])
        #acc_med[2,k] = copy(accel_device[2]) #copy(accel_device.reshape((3,1)))

        # process accel data
        pos_est[:,k], vel_est[:,k], acc_est[:,k] = cartesian_kalman.update(accel_device)  



        #print(f"med: {acc_med[:,:]}")        
        #print(f"est: {acc_est[:,k]}")        

    delsys_handle.end_communication()
    
    print(f"closing sockets ...")

    plt.figure(figsize=(20, 7))
    plt.subplot(1,3,1)
    plt.plot(pos_est[0,:], '--r')
    plt.plot(pos_est[1,:], '--g')
    plt.plot(pos_est[2,:], '--b')        
    plt.title('position')
    
    plt.subplot(1,3,2)
    plt.plot(vel_est[0,:], '--r')
    plt.plot(vel_est[1,:], '--g')
    plt.plot(vel_est[2,:], '--b')        
    plt.title('velocity')

    #plt.subplot(1,1,2)
    #plt.plot(acc_med[0,:], '--r')
    #plt.plot(acc_med[1,:], '--g')
    #plt.plot(acc_med[2,:], '--b')        
    #plt.title('acceleration')

    plt.subplot(1,3,3)
    plt.plot(acc_est[0,:], '--r')
    plt.plot(acc_est[1,:], '--g')
    plt.plot(acc_est[2,:], '--b')        
    plt.title('acceleration')
    plt.show()                 


    


except:
    delsys_handle.end_communication()
    
    print(f"closing sockets ...")
    
