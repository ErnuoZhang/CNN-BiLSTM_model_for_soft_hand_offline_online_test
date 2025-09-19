import multiprocessing as mp
import ctypes as ct
import time
import msvcrt
from threading import Thread
import utils.Kinematics
import numpy as np
from utils.Dinamic_Array import DinamicArray
import os

class LeapdataStruct(ct.Structure):
    _fields_ = [('Handid', ct.c_int),
    ('framerate',ct.c_float),
    ('timestamp',ct.c_int),
    ('RS',ct.POINTER(ct.c_float)),
    ('M1',ct.POINTER(ct.c_float)),
    ('M2',ct.POINTER(ct.c_float)),
    ('M3',ct.POINTER(ct.c_float)),
    ('M4',ct.POINTER(ct.c_float)),
    ('M5',ct.POINTER(ct.c_float)),
    ('P1',ct.POINTER(ct.c_float)),
    ('P2',ct.POINTER(ct.c_float)),
    ('P3',ct.POINTER(ct.c_float)),
    ('P4',ct.POINTER(ct.c_float)),
    ('P5',ct.POINTER(ct.c_float)),
    ('D1',ct.POINTER(ct.c_float)),
    ('D2',ct.POINTER(ct.c_float)),
    ('D3',ct.POINTER(ct.c_float)),
    ('D4',ct.POINTER(ct.c_float)),
    ('D5',ct.POINTER(ct.c_float))]


# C type callbacks
callback_Leapdata = ct.CFUNCTYPE(None, LeapdataStruct)


class Leap_Listener(mp.Process):

    def __init__(self, path, angles, Comand_Queue, leap_framerate, shared_data):
        """ Creates multiprocess as a class """
        mp.Process.__init__(self, daemon=True)
        self.shared_data = shared_data
        self.path = path
        self.angles = angles
        self.Comand_Queue = Comand_Queue
        self.leap_framerate = leap_framerate
        self.n_save = 0 
    
    def Load_dll(self, path):
        """ Loads dll, prep functions and sends callback"""
        self.dll_loaded = True

        #Loads the Dll
        self.API = ct.CDLL(path)

        # Sets args and returns types
        self.API.Setcallbacks.argtypes = [callback_Leapdata]
        self.API.Setcallbacks.restype = None
        self.API.main.argtypes = None
        self.API.main.restype = ct.c_int
        self.API.controlExE.argtypes = [ct.c_char]
        self.API.controlExE.restype = None

        # Sends callback
        self.python_calback = self.leapdata()
        self.API.Setcallbacks(self.python_calback)
    

    def run(self):
        """ Overide base run() with your code """
        self.Load_dll(self.path)
        if self.dll_loaded:
            # self.Setup_Arrays()

            # Calls comand listener on a thread
            main_thread = Thread(target= self.Comand_input, daemon=True)
            main_thread.start()

            # Calls dll loop on a thread
            main_thread = Thread(target= self.API.main)
            main_thread.start()

    # def Setup_Arrays(self):
    #     # Setup some np.arrays that can grow if needed 
    #     # not really recomended for them to grow as it is really slow, but it won't crash anything most likely
    #     self.Angle_data = DinamicArray(init_capacity = 30000, shape=(0, 11), capacity_multiplier=2, dtype= "int16")
    #     self.AngleTS_data = DinamicArray(init_capacity = 30000, shape=(0, 1), capacity_multiplier=2, dtype= "int64")

    # def Acumulate_Angles(self):
    #     self.Angle_data.update(self.angles)
    #     self.AngleTS_data.update(time.time_ns()//1_000_000)
    #     # if self.AngleTS_data.get_size() % 1000 == 0:
    #     #     print("Acumulated ", self.AngleTS_data.get_size(), " Leap samples")
    
    # def Save_angles(self):
    #     time.sleep(1)
    #     self.shared_data.save = False
    #     print("-- Starting to save Leap Data --")
    #     if not os.path.exists("Data/" + str(self.n_save)+"/"):
    #         os.makedirs("Data/" + str(self.n_save)+"/")
    #     np.save("Data/"+ str(self.n_save) +"/Raw_angles", self.Angle_data.finalize(), allow_pickle=True)
    #     np.save("Data/"+ str(self.n_save) +"/Raw_Angle_TS", self.AngleTS_data.finalize(), allow_pickle=True)
    #     self.n_save = self.n_save + 1
    #     print("-- Leap Data Saved --")
    #     del self.Angle_data
    #     del self.AngleTS_data
    #     self.Setup_Arrays()
    #     print("-- New Leap ready --")


    def leapdata(self):
            """
            Leap Motion data callback
            """
            @callback_Leapdata
            def Leapdata(data):
                if not self.shared_data.online:
                    self.angles[0:5], self.angles[5:10], self.angles[10] = utils.Kinematics.Calculate_angles(data)

                # print((time.time_ns()//1_000_000) - self.leap_framerate.value)
                frame = (time.time_ns()//1_000_000)
                self.leap_framerate.value = frame
                # print(self.leap_framerate.value, time.time_ns()//1_000_000)

                # if self.shared_data.save_processed:
                #     self.Acumulate_Angles()

                # if self.shared_data.save:
                #     self.Save_angles()
                
        
            return Leapdata
    
    def Comand_input(self):
        """
        Runs on thread to continously listen to char comands
        """
        c = " "
        while not self.shared_data.end:
            c = msvcrt.getwch()
            self.API.controlExE(c.encode()) # Sends the comands to Leap
            self.Comand_Queue.put(c) # Puts command in Queue for Window Manager
            time.sleep(.5) # No need to do this super fast
        