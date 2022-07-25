import numpy as np


def fifth_order_pos_trajectory(a,t):
    return np.dot(np.array([np.power(t,5), np.power(t,4), np.power(t,3), np.power(t,2), t, 1]), a).item()

def fifth_order_vel_trajectory(a,t):
    return np.dot(np.array([5*np.power(t,4), 4*np.power(t,3), 3*np.power(t,2), 2*t, 1, 0]), a)    

def fifth_order_accel_trajectory(a,t):
    return np.dot(np.array([20*np.power(t,3), 12*np.power(t,2), 6*t, 2, 0, 0]), a)    


def fifth_order_trajectory_generator(x_1, x_2, sim_time):
    """
    @info: generate trajectory for two position points with zero velocity amd accel.
    @output:
    --------
        - p:    position points
    """
    # time matrix 
    T = np.array([  [0, 0, 0, 0, 0, 1], 
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 2, 0, 0],
                    [np.power(sim_time,5), np.power(sim_time,4), np.power(sim_time,3), np.power(sim_time,2), sim_time, 1], 
                    [5*np.power(sim_time,4), 4*np.power(sim_time,3), 3*np.power(sim_time,2), 2*sim_time, 1, 0],
                    [20*np.power(sim_time,3), 12*np.power(sim_time,2), 6*sim_time, 2, 1, 0]])

    # position and velocity of the two points
    x = np.array([x_1, 0, 0, x_2, 0, 0]).reshape((-1,1))            

    # trajectory coefficients
    a = np.dot(np.linalg.inv(T),x)

    # generate points
    #pos = [fifth_order_pos_trajectory(a, sim_time*t/n_points) for t in range(0,n_points+1)]
    #vel = [fifth_order_vel_trajectory(a, sim_time*t/n_points) for t in range(0,n_points+1)]
    #accel = [fifth_order_accel_trajectory(a, sim_time*t/n_points) for t in range(0,n_points+1)]

    #if return_time:
    #    return pos, vel, accel, [sim_time*t/n_points for t in range(n_points+1)]

    return a