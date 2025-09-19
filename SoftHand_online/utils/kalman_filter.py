from filterpy.kalman import predict, KalmanFilter
from filterpy.common import Q_discrete_white_noise, Q_continuous_white_noise, Saver
from filterpy.stats import mahalanobis
import numpy as np


class KalmanFilterModule():
    def __init__(self):
        # Kalman Filter pos and vel
        dt = 1
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([0., 0.]) # Initial state - x,  vel
        self.kf.P = np.diag([10., 10.]) # Initial cov matrix
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=.1) # Uncertanty of the motion predicition - set it between 1/2 aceleration and aceletation (between 2 samples)
        self.kf.H = np.array([[1., 0.]]) # Converts mesurement to kalman variable space
        self.kf.R = np.array([[20.]]) # Mesurement Noise
        self.kf.F = np.array([[1, dt], [0, 1]]) # equation for the state variables to evolve based on 

    def Next_step(self, z):
        self.kf.predict()
        # print(self.kf.x[0])
        self.kf.update(z)
        # print(self.kf.x[0])
        return self.kf.x[0]