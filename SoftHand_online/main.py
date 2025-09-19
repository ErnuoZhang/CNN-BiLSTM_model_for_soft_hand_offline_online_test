import numpy as np
import os
import multiprocessing as mp
import ctypes as ct
import psutil
from pathlib import Path
import sys

import Leap_Listener
import Diphas_Listener
import Window


if __name__ == "__main__": # only runs once on main

    with mp.Manager() as data_manager: #Shared Data Manager
        shared_data = data_manager.Namespace()

        # Setup of the US comming - This has to agree with the acquisition parameters on the C API   
        n_channels, n_shots = 16, 16
        samples_per_line = 2996
        offset = 0

        # Training
        shared_data.traininng_path = "Training/path.txt"
        
        # visualization
        visualization_downsampling = 2

        #Setup Processing and Machine learning
        Standart_feature_window = 20
        Downsampling_factor = 1
        Bracelet_groups = [np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])]
        # Bracelet_groups = [np.array([0,1,2,3,4,5,6,7,8,9,10,11])]

        #Create Queues
        Comand_Queue, Target_Queue = mp.Queue(), mp.Queue()


        # Control flags
        shared_data.end = False #Flags program close
        shared_data.save_processed = False 
        shared_data.save_raw = False
        shared_data.centre_online = False
        shared_data.train = False
        shared_data.online = False
        shared_data.process = False
        shared_data.ready = False
        shared_data.save = False
        shared_data.show_us = False
        shared_data.load = False
        shared_data.save_real_time = False
        shared_data.soft_hand = True

        # Shared ctype variables
        Data_plot = mp.Array(ct.c_double, int(n_shots * n_channels* ((samples_per_line - offset)/visualization_downsampling)))# Plotting array
        leap_framerate = mp.Value('d', 1) # Time of last frame
        angles = mp.Array(ct.c_double, 11) 


        # Set process to real time priority
        process = psutil.Process(os.getpid())
        process.nice(psutil.REALTIME_PRIORITY_CLASS)

        # Deals with the Leap Motion Data Flow
        Leap = Leap_Listener.Leap_Listener("Leap_dlls/PollingSample.dll", angles, Comand_Queue, leap_framerate, shared_data)
        Leap.start()

        # Window
        window = Window.Window_Manager(Target_Queue, angles, leap_framerate, Data_plot, visualization_downsampling, offset, shared_data)
        window.start()

        #Start Listener 
        Diphas = Diphas_Listener.DiphasDataListener(offset, angles, n_shots, n_channels, Downsampling_factor, visualization_downsampling, samples_per_line, Standart_feature_window, 
                                                        Bracelet_groups, Target_Queue, Comand_Queue, Data_plot, shared_data)
        path = os.path.join(Path().resolve(), "DiPhaS_DLLs")
        os.add_dll_directory(path)
        path = os.path.join(path, "Application.CWrapperBinaryTest.dll")
        print(path)
        # Diphas.Load_dll(path)
        Diphas.Load_dll('C:\\Users\\Be.Neuro\\Desktop\\Bruno\\SoftHandControl\\DiPhaS_DLLs\\Application.CWrapperBinaryTest.dll')
        Diphas.Start()

        # Clear all queue data
        print("Clearing Queues")
        for queue in [Comand_Queue, Target_Queue]:
            while not queue.empty():
                queue.get() 

        # Joins All Children Processes
        print("Joining Procesees")
        for p in mp.active_children():
            p.join(2)
            print("Killing: ", p.name)
            p.terminate() 

        print("Shuting down - bye bye")
        sys.exit(0)
