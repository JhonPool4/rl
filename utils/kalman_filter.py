from copy import copy
import numpy as np
from numpy.linalg import inv
from numpy import matmul as mx
from numpy import transpose as tr


class KalmanFilter:
    """
    @info the kalman filter algorithm of the book "Probabilistic Robotics (Thrun 2000, pg. 36)" 
    @inputs:
    -------
        - x_est0: initial states
        - n_obs: number of observable states
        - deltaT: sampling time
        - sigmaR: covariance matrix that indicates model uncertainty / motion noise
        - sigmaQ: covariance matrix that indicates measurement noise  
    """
    def __init__(self, x_est0, n_obs, deltaT, sigmaR = 1e-3, sigmaQ = 1e-1):
        # useful parameters
        self.deltaT = deltaT # samping time
        self.n_input = len(x_est0) # input states
        self.n_obs = n_obs # output states

        # prediction stage: initial values
        self.F = np.array([[1., self.deltaT],[0., 1.]]) # model
        self.x_hat = np.zeros((self.n_input,1)) # [x, dx, ddx]
        self.P_hat =  np.zeros((self.n_input, self.n_input))

        # observation-correction stage: initial values
        self.H = np.array([1,0]).reshape((self.n_obs, self.n_input)) # to scale measurements
        self.x_est = copy(x_est0).reshape((self.n_input,1)) # initial states [x, dx ,ddx]
        self.P_est = np.zeros((self.n_input, self.n_input))
        
        # covariance matrices
        self.R = sigmaR*np.eye(self.n_input)  # model uncertainty
        self.Q = sigmaQ*np.eye(self.n_obs) # measurement noise

        # kalman gain: initial value
        self.K = np.zeros((self.n_input,self.n_obs))         
        self.I = np.eye(self.n_input)

        
    def update(self, q, new_deltaT=None):
        if new_deltaT is not None:
            self.deltaT=new_deltaT
            self.F = np.array([[1., self.deltaT],[0., 1.]]) # model
        # measurements
        self.z = np.array([q])

        #print(f"F: {self.F}")
        #print(f"q: {self.z}")
        # prediction stage
        self.x_hat = mx(self.F,self.x_est)
        self.P_hat = mx(self.F, mx(self.P_est, tr(self.F))) +  self.R

        # kalman gain
        self.K = mx(mx(self.P_hat, tr(self.H)), inv(mx(self.H, mx(self.P_hat, tr(self.H))) + self.Q))   

        # observation-correction stage      
        self.x_est = self.x_hat + mx(self.K, (self.z - mx(self.H, self.x_hat)))
        self.P_est = mx(self.I - mx(self.K,self.H), self.P_hat)
        
        # return position, velocity and acceleration
        return self.x_est[0][0], self.x_est[1][0]
