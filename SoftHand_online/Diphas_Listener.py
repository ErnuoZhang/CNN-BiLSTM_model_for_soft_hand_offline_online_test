import ctypes as ct
import numpy as np
from threading import Thread
import msvcrt
import time
import multiprocessing as mp
import os
#

from Processing.DataPipeline import DataPipeline

# Defines C structures and functions
class USdataStruct(ct.Structure):
    _fields_ = [('rfDataChannelData', ct.POINTER(ct.c_short)),
    ('channelDataChannelsPerDataset',ct.c_int),
    ('channelDataSamplesPerChannel',ct.c_int),
    ('channelDataTotalDatasets',ct.c_int),
    ('rfDataArrayBeamformed', ct.POINTER(ct.c_short)),
    ('beamformedLinesX',ct.c_int),
    ('beamformedLinesY',ct.c_int),
    ('beamformedSamples',ct.c_int),
    ('beamformedTotalDatasets',ct.c_int),
    ('imageData',ct.POINTER(ct.c_ubyte)),
    ('imageWidth',ct.c_int),
    ('imageHeight',ct.c_int),
    ('imageDepth',ct.c_int),
    ('imageBytesPerPixel',ct.c_int),
    ('imageSetsTotal',ct.c_int),
    ('timeStamp',ct.c_double)]

callback_Statusmessages = ct.CFUNCTYPE(None, ct.c_char_p)
callback_Exceptionmessages = ct.CFUNCTYPE(None, ct.c_char_p)
callback_USdata = ct.CFUNCTYPE(None, USdataStruct)

# Diphas Listener
class DiphasDataListener(DataPipeline):
    def __init__(self, offset, angles, n_shots, n_channels, Downsampling_factor, visualization_downsampling, samples_per_line, Standart_feature_window, 
                    Bracelet_groups, Target_Queue, Comand_Queue, Data_plot, shared_data):
        """
        """
        self.offset = offset
        self.nshts = n_shots
        self.nchs = n_channels
        self.visualization_downsampling = visualization_downsampling
        self.last = 0
        self.ml_delay = 0
        self.angles = angles

        self.Target_Queue = Target_Queue
        self.Comand_Queue = Comand_Queue
        self.Data_plot = Data_plot
        self.shared_data = shared_data

        #Setup ML_manager
        self.Setup_MLManager(self.nshts, self.nchs, samples_per_line, offset, Standart_feature_window, Bracelet_groups, Downsampling_factor)


    def Load_dll(self, dll_path):
        """
        Loads dll and other dinamic dependencies and registers callbacks

        Parameters
        ----------
        dll_path : str
            Path to the dll. Must externaly expose "PredefinedScanModesTest", "Setcallbacks" and "controlExE" functions
        """
        self.API = ct.CDLL(dll_path)
        # Define args and res of functions
        self.API.Setcallbacks.argtypes = (callback_Statusmessages, callback_Exceptionmessages, callback_USdata)
        self.API.Setcallbacks.restype = None
        self.API.controlExE.argtypes = [ct.c_char]
        self.API.controlExE.restype = None
        self.API.PredefinedScanModesTest.argtypes = None
        self.API.PredefinedScanModesTest.restype = ct.c_int
        # Bind callbacks
        self.cb_st = self.initStatusmessages()
        self.cb_exc = self.initExceptionmessages()
        self.cb_data = self.initUSdata()
        self.API.Setcallbacks(self.cb_st, self.cb_exc, self.cb_data)

    def Start(self):
        """
        Starts Diphas listener and control thread
        """
        self.Dhiphas_thread = Thread(target= self.API.PredefinedScanModesTest)
        self.Dhiphas_thread.start()
        print("Phyton SETUP: DIPHAS starting on ", self.Dhiphas_thread.name, "...")
        self.Comand_thread = Thread(target= self._Comand_input, daemon=True)
        self.Comand_thread.start()
        print("Phyton SETUP: Comand input starting on ", self.Comand_thread.name, "...")
        while not self.shared_data.end:
            time.sleep(.5)
            if self.shared_data.train:
                self.Train_thread = Thread(target=self.Train_Model)
                self.Train_thread.start()
                self.shared_data.train = False
            if self.shared_data.load:
                self.Load_thread = Thread(target=self.Load_Network)
                self.Load_thread.start()
                self.shared_data.load = False
            if self.shared_data.save_real_time:
                self.Load_thread = Thread(target=self.Save_Real_time)
                self.Load_thread.start()
                self.shared_data.save_real_time = False

        self.API.controlExE("q".encode())
        time.sleep(1)


    def initStatusmessages(self):
        """
        Diphas Status message callback
        """
        @callback_Statusmessages
        def Statusmessages(message):
            print("===================================================================================================")
            print("DIPHAS Message: ", message.decode())
            print("===================================================================================================")
            time.sleep(.1) # just to slow it down.
        return Statusmessages

    def initExceptionmessages(self):
        """
        Diphas Exception message callback
        """
        @callback_Exceptionmessages
        def Exceptionmessages(message):
            print("===================================================================================================")
            print("DIPHAS Exception: ", message.decode())
            print("===================================================================================================")
            time.sleep(.1) # just to slow it down.
        return Exceptionmessages

    def initUSdata(self):
        """
        Diphas US data callback
        """
        @callback_USdata
        def USdata(data):

            self.US = np.ctypeslib.as_array(data.rfDataChannelData, 
                            shape=(self.nshts, data.channelDataChannelsPerDataset, data.channelDataSamplesPerChannel))[:,16:32,self.offset::self.Downsampling_factor]
            self.Timestamp = time.time_ns()//1_000_000

            # print("Time since last: ", time.time_ns()//1_000_000 - self.last)
            self.last = time.time_ns()//1_000_000

            #Process Data
            if self.shared_data.process:
                self.Process_US()

            # acumulates processed data
            if self.shared_data.save_processed:
                self.Acumulate_Feature()

            # Acumulates Raw data
            if not self.shared_data.online and not self.shared_data.save_processed:
                if self.shared_data.save_raw:
                    self.Acumulate_Raw()

            # Predicts targets
            if self.shared_data.online:
                target = self.Predict()
                self.Target_Queue.put(target)

            # Copy US to shared buffer if we are visualizing it
            if self.shared_data.show_us:
                np.copyto(np.frombuffer(self.Data_plot.get_obj()), self.US[:,:,::self.visualization_downsampling].reshape(-1))

        return USdata

    def _Comand_input(self):
        """
        Runs on thread to continously listen to char comands
        """
        c = " "
        while not self.shared_data.end:
            c = msvcrt.getwch()
            self.API.controlExE(c.encode()) # Sends the comands to DIPHAs
            self.Comand_Queue.put(c) # Puts command in Queue for Window Manager
            time.sleep(.5) # Slows this down as there is no need for it to run a lot and takes processing from the data reciving
        