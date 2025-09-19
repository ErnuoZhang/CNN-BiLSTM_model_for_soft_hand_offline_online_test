from multiprocessing.dummy import Process
import Processing.Preprocessing
import numpy as np
import time
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from numpy.lib.stride_tricks import sliding_window_view
import multiprocessing as mp
import os
from scipy import signal
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn import preprocessing
import ctypes as ct
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter, sosfiltfilt
import joblib
from utils.kalman_filter import KalmanFilterModule
from utils.Dinamic_Array import DinamicArray, find_nearest_idx
import Processing.Regression
import Processing.USCruncher
from Processing.Training import TrainingManager
import Network_funcs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

def R2(y_pred, y_true):
    return r2_score(y_true, y_pred, multioutput="raw_values")

class DataPipeline(Processing.USCruncher.USCruncher, TrainingManager):
    def __init__(self):
        # "Shared" variables of current sample
        self.US = []
        self.Timestamp = 0
        self.Features = []
        self.target = [0,0,0]
        TrainingManager.__init__(self) # start training manager

    def Setup_MLManager(self, nshots, nchs, samples_per_line, offset, Standart_feature_window, Bracelet_groups,  Downsampling_factor):
        self.autolabeling = False
        self.folder = 0
        self.predictions_ac = []
        self.leap_ac = []
        self.nshots, self.nchs, self.samples_per_line, self.offset = nshots, nchs, samples_per_line, offset
        self.Downsampling_factor = Downsampling_factor
        # Setup Dynamic arrays and alocate memory
        self._Setup_Arrays()
        self._Setup_USCruncher(self.nchs, self.samples_per_line)
        # Setups ML Model
        self._Setup_Model()

    def _Setup_Model(self):
        # Sets model up
        model_path = r"C:\Users\Be.Neuro\Desktop\Ernuo\sweep_3_best_mnodel\best_model_ancient-sweep-2.pth"
        self.model = Network_funcs.CNNBiLSTMRegressor(16, 0, lstm_hidden_dim=128, conv1_filters=64, conv2_filters=128)
        self.model.load_state_dict(torch.load(model_path))
        self.model.cuda()
        self.model.eval()


    def Load_Network(self):
        n = 6
        self.X = np.load("Data/"+ str(n)+ "/_feature_US.npy").astype("float32")
        self.Y = np.load("Data/"+ str(n)+ "/_leap_angles.npy")[:,[1,3]].astype("float32")
        print("Fine tuning Data:", self.X.shape, self.Y.shape)
        self.Fine_tune()

    def _Setup_Arrays(self):
        self.US_data = DinamicArray(init_capacity = 20000, shape=(0, self.nshots, self.nchs, self.samples_per_line), capacity_multiplier=2, dtype= "int16")
        self.Feature_data = DinamicArray(init_capacity = 20000, shape= (0, 16, 2776), capacity_multiplier=2, dtype= "float32")
        self.Angle_data = DinamicArray(init_capacity = 20000, shape=(0, 11), capacity_multiplier=2, dtype= "float32")
        self.TS_data = DinamicArray(init_capacity = 20000, shape=(0, 1), capacity_multiplier=2, dtype= "int64")

    def Acumulate_Feature(self):
        self.Feature_data.update(self.Features)
        self.Angle_data.update(self.angles)
        self.TS_data.update(self.Timestamp)
        # self.US_data.update(self.US)

    def Acumulate_Raw(self):
        self.US_data.update(self.US)
        self.Angle_data.update(self.angles)
        self.TS_data.update(self.Timestamp)

    def Create_labels_data_Pairs(self):

        self.X = self.Feature_data.finalize()
        self.Y = self.Angle_data.finalize()[:,[1,3]]
        print("Fine tuning Data:", self.X.shape, self.Y.shape)

    def Train_Model(self):
        self.Create_labels_data_Pairs()
        self.Save_data()
        self.Fine_tune()

    def Fine_tune(self):
        self.setup_kalman()
        self.Y = (self.Y - np.amin(self.Y)) / (np.amax(self.Y)  - np.amin(self.Y))
        # Create DataLoader
        half_point = int(self.X.shape[0]/2)
        self.loader = DataLoader(Network_funcs.UltrasoundDataset(self.X[:half_point, ...], self.Y[:half_point, ...]), batch_size=16, shuffle=True)
        self.loader_test = DataLoader(Network_funcs.UltrasoundDataset(self.X[half_point:, ...], self.Y[half_point:, ...]), batch_size=16, shuffle=False)
        # ----------Fine-tune
        self.model.train()
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()

        self.model.eval()
        true_values, predicted_values = [], []
        with torch.no_grad():
            for train_data, train_target in self.loader_test:
                train_data = train_data.to("cuda")
                train_target = train_target.to("cuda")
                predictions = self.model(train_data)
                true_values.append(train_target.cpu().numpy())
                predicted_values.append(predictions.cpu().numpy())
        predicted_values = np.concatenate(predicted_values, axis=0)
        true_values = np.concatenate(true_values, axis=0)
        print("Orginal r2. ", np.round(R2(predicted_values, true_values),2))

        # loop over epochs
        for epoch in range(10):
            # Training 
            self.model.train()
            train_loss = 0.0
            true_values, predicted_values = [], []
            for train_data, train_target in self.loader:
                train_data = train_data.to("cuda")
                train_target = train_target.to("cuda")
                # print(train_data[0,0,500:520])
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(train_data)
                loss = self.loss_fn(outputs, train_target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * train_data.size(0)
            train_loss /= self.X.shape[0]
            print("Epoch:", epoch, " - loss: ", train_loss)
            # testing every epoch
            self.model.eval()
            with torch.no_grad():
                for train_data, train_target in self.loader_test:
                    train_data = train_data.to("cuda")
                    train_target = train_target.to("cuda")
                    predictions = self.model(train_data)
                    true_values.append(train_target.cpu().numpy())
                    predicted_values.append(predictions.cpu().numpy())

            predicted_values = np.concatenate(predicted_values, axis=0)
            true_values = np.concatenate(true_values, axis=0)
            print("r2. ", np.round(R2(predicted_values, true_values),2))

        np.save("pred_test.npy", predicted_values)
        np.save("true_test.npy", true_values)
        self.model.eval()
        print("Network Fine-tuned - Ready for Online")

    def setup_kalman(self):
        self.kf_0 = KalmanFilterModule()
        self.kf_1 = KalmanFilterModule()

    def kalman_predict(self, prediction):
        prediction[0] = self.kf_0.Next_step(prediction[0])
        prediction[1] = self.kf_1.Next_step(prediction[1])                   
        return prediction

    def Predict(self):
        predictions = self.model(torch.tensor(self.Features, dtype=torch.float32).to("cuda"))
        preds = predictions.cpu().detach().numpy()
        preds = self.kalman_predict(preds[0])
        preds[0] = preds[0]* 100
        preds[1] = preds[1]* 100
        self.predictions_ac.append(preds)
        temp = [self.angles[1], self.angles[3]]
        self.leap_ac.append(temp)
        return preds

    def Save_Real_time(self):
        if not self.shared_data.online:
            if len(self.predictions_ac) != 0:
                print("Saving Prediciton Data")
                while os.path.exists("Data/Online_" + str(self.folder)+"/"):
                    self.folder += 1
                os.makedirs("Data/Online_" + str(self.folder)+"/")
                self.predictions_ac = np.concatenate(self.predictions_ac, axis=0)
                self.leap_ac = np.concatenate(self.leap_ac, axis=0)
                np.save("Data/Online_" + str(self.folder)+"/preds.npy", self.predictions_ac)
                np.save("Data/Online_" + str(self.folder)+"/leap.npy", self.leap_ac)
                self.predictions_ac = []
                self.leap_ac = []

    def Save_data(self):
        print("Saving Data - Wait before recording again")
        while os.path.exists("Data/" + str(self.folder)+"/"):
            self.folder += 1
        os.makedirs("Data/" + str(self.folder)+"/")
        if self.US_data.get_size() > 1:
            np.save("Data/" + str(self.folder)+"/_raw_US", self.US_data.finalize())
        if self.Feature_data.get_size() > 1:
            np.save("Data/" + str(self.folder)+"/_feature_US", self.Feature_data.finalize())
        np.save("Data/" + str(self.folder)+"/_leap_angles", self.Angle_data.finalize()) 
        np.save("Data/" + str(self.folder)+"/_timesamples", self.TS_data.finalize())
        del self.US_data
        del self.Feature_data
        del self.Angle_data
        del self.TS_data
        self.folder += 1
        self._Setup_Arrays()
        print("Data Saved - Ready for next one")
        self.shared_data.ready = True
