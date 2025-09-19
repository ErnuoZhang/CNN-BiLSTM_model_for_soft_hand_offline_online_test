import multiprocessing as mp
from matplotlib import pyplot as plt
import matplotx
import matplotlib
import numpy as np
from utils.Blit_Manager import Blit_Manager
import utils.CustomWidgets as cw
import time
import os
#from pySerialTransfer import pySerialTransfer as txfer
import Processing.Preprocessing
from Processing.Training import TrainingManager
from scipy.signal import butter, lfilter, sosfiltfilt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
import Softhand.controlFuncs


class Window_Manager(mp.Process, TrainingManager):
    def __init__(self, Target_Queue, angles, leap_framerate, Data_plot, visualization_downsampling, offset, shared_data):
        mp.Process.__init__(self, daemon=True) 

        self.vertical_offset = 200
        self.visualization_downsampling = visualization_downsampling
        self.offset = offset
        self.training = False
        self.Data_plot = Data_plot
        self.angles = angles
        self.leap_framerate = leap_framerate
        self.Target_Queue = Target_Queue
        self.shared_data = shared_data
        self.plot_angles = np.zeros((1498, 11))
        self.rolling_buff = 0

        self.last_values = [0, 0]

    def run(self):
        """Overide base run() with our Draw loop."""
        TrainingManager.__init__(self)
        self.Setup_Draw()
        self.Draw_loop()

    def Setup_Draw(self):
        # Creates and setups all GUI elements
        plt.style.use(matplotx.styles.onedark)
        matplotlib.rcParams['toolbar'] = 'None' 
        self.fig, self.axes = plt.subplot_mosaic("AAAAHC;AAAAID;AAAAJE;AAAAKF;AAAALG;AAAABZ")
        self.fig.set_figheight(8), self.fig.set_figwidth(12)
        self.fig.canvas.manager.window.move(200, 0) # 1screen 500 --- 2screens 2300
        self.fig.canvas.mpl_connect('close_event', self.BT_Shutdow)
        self.fig.set_tight_layout(True)

        #Creates the intial main screen
        self.Main_screen(screen="Joystick", first_time = True, samples= int((2996 - self.offset)/self.visualization_downsampling))
        self.channel = 0 
        # axep = self.fig.add_axes(self.axes["A"].get_position().bounds, polar=True, frameon=False)
        # axep.set_rlim(0,100)

        self.bt_end = cw.Custom_Buttom(self.axes["Z"], 'Close', self.BT_Shutdow)
        self.bt_load = cw.Custom_Buttom(self.axes["C"], 'Load', self.BT_Load)
        self.bt_tra = cw.Custom_Buttom(self.axes["D"], 'Record', self.BT_training)
        self.bt_onl = cw.Custom_Buttom(self.axes["F"], 'Start Online', self.BT_online)
        self.bt_showUS = cw.Custom_Buttom(self.axes["I"], 'Show US', self.BT_ShowUS)
        self.bt_showLeap = cw.Custom_Buttom(self.axes["J"], 'Show Leap', self.BT_ShowLeap)
        self.bt_showControl = cw.Custom_Buttom(self.axes["K"], 'Show Control', self.BT_ShowControl)
        self.bt_channel = cw.Custom_Slider(self.axes["L"], 'Channel', self.BT_Channel)

        self.busy = self.axes["H"].annotate(" ", (0.7, 0.85), xycoords='figure fraction', c="#1f9c1f", fontsize= "large", animated=True)

        self.axes["B"].axis("off")
        self.axes["G"].axis("off")
        self.axes["E"].axis("off")
        self.axes["H"].axis("off")

        # Sends all artists to Blitt Manager
        self.Bmg = Blit_Manager(self.fig.canvas, [*self.main_screen_axes, self.busy,
                                    self.bt_end.ax, self.bt_load.ax, self.bt_tra.ax, self.bt_onl.ax, 
                                    self.bt_showUS.ax, self.bt_showLeap.ax, self.bt_showControl.ax, self.bt_channel.ax])

        #Displays window and wait for it to show before blitter takes control
        plt.show(block=False)
        plt.pause(.1)


    def Main_screen(self, screen= "Joystick", first_time = False, samples = 470):

        if first_time:
            self.main_screen_axes  = []
            #Joystick
            self.US_streaming = False   
            self.Leap_streaming = False    
            (self.ln1,) = self.axes["A"].plot([0,0],[0,0], marker = 'o', animated=True)
            self.main_screen_axes.append(self.ln1)
            self.ln1.set(alpha = 0)
            (self.ln2,) = self.axes["A"].plot([0,0],[0,0], marker = 'o', color="r", animated=True)
            self.main_screen_axes.append(self.ln2)
            self.ln2.set(alpha = 0)
            #US plots
            for ch in range(16):
                (self.ln,) = self.axes["A"].plot(np.arange(0, samples,1), np.zeros(samples) - (self.vertical_offset*ch), animated=True)
                self.ln.set(alpha = 0)
                self.main_screen_axes.append(self.ln)

        if screen == "Joystick":
            self.US_streaming = False
            self.axes["A"].set(xbound = (0,100))
            self.axes["A"].set(ybound = (0,100))
            self.axes["A"].grid("on", linestyle = "--", alpha=0.7)
            self.ln1.set(alpha = 1)
            self.ln2.set(alpha = 1)
            for x in range(16):
                self.main_screen_axes[x+2].set(alpha = 0)

        elif screen == "US":
            self.US_streaming = True
            self.axes["A"].set_xlim(0, int((2996 - self.offset)/self.visualization_downsampling))
            # self.axes["A"].set_yticks(np.full(16, 800) - (self.vertical_offset*36))
            # self.axes["A"].set_xlim(0, 470)
            self.axes["A"].set_ylim(-self.vertical_offset*16.5, self.vertical_offset * 0.5)
            self.ln1.set(alpha = 0)
            self.ln2.set(alpha = 0)
            for x in range(16):
                self.main_screen_axes[x+2].set(alpha = 1)

        elif screen == "Leap":
            self.Leap_streaming = True
            self.axes["A"].set_xlim(0, 500)
            self.axes["A"].set_ylim(200, -1000)
            self.ln1.set(alpha = 0)
            self.ln2.set(alpha = 0)
            for x in range(8):
                self.main_screen_axes[x+2].set(alpha = 1)

    def Draw_loop(self):
        """Draw loop"""
        while not self.shared_data.end: # Main Draw Loop
            

            if self.training:
                # targets = self.update_training()
                if (time.time_ns()//1_000_000 - self.start_time) > 60000:
                        self.shared_data.save_raw = False
                        self.shared_data.save_processed = False
                        self.shared_data.process = False
                        self.training = False
                        self.shared_data.train = True
                        print("Finished Recording")
                    
                

            if not self.US_streaming and not self.Leap_streaming: # we are on the control case
                if self.shared_data.online:
                    if not self.Target_Queue.empty():
                        targets = self.Target_Queue.get()
                        if self.shared_data.soft_hand:
                            self.Move_Soft_Hand(targets[0], targets[1])
                        
                        self.ln2.set_data([targets[0], 0], [targets[1], 0])
            else:
                if self.US_streaming:
                    data = np.frombuffer(self.Data_plot.get_obj()).reshape(16, 16, int((2996 - self.offset)/self.visualization_downsampling))

                    for ch in range(16):
                        self.main_screen_axes[ch+2].set_ydata(data[self.channel, ch,:] - (self.vertical_offset * ch))

                if self.Leap_streaming:
                    angles = np.frombuffer(self.angles.get_obj())
                    # print(angles.shape)
                    self.plot_angles[self.rolling_buff,:] = angles
                    self.rolling_buff += 1
                    if self.rolling_buff == 500: 
                        self.rolling_buff = 0
                        self.plot_angles = np.zeros((1498, 11))
                        
                    for ch in range(5):
                        self.main_screen_axes[ch+2].set_ydata(self.plot_angles[:,ch] - (self.vertical_offset * ch))


            self.Bmg.update() # Blits window!

    def BT_Shutdow(self, event):
        print("Closing")
        if self.shared_data.soft_hand:
            try:
                Softhand.controlFuncs.set_motor_inputs(self.softhand_serial, self.ID, 0, 0)
                Softhand.controlFuncs.activate_device(self.softhand_serial, self.ID, False)
                self.softhand_serial.close()
            except:
                pass

        self.shared_data.end = True
    
    def BT_Load(self, event):
        self.shared_data.load = True

    def BT_online(self, event):

        if self.shared_data.online:
            self.shared_data.save_real_time = True
            self.shared_data.online = False
            self.shared_data.process = False
            self.shared_data.save_processed = False 
            self.shared_data.save_raw = False
            self.busy.set_text(" Offline ")
        else:
            if self.shared_data.soft_hand:
                # Start and Connect to softhand
                self.ID = 1
                self.limit1 = -14500
                self.limit2 =  20000
                self.softhand_serial = Softhand.controlFuncs.init_device("COM5")
                Softhand.controlFuncs.activate_device(self.softhand_serial, self.ID, True)
                Softhand.controlFuncs.set_motor_inputs(self.softhand_serial, self.ID, 0, 0)

            
            self.shared_data.save_real_time = True
            self.shared_data.online = True
            self.shared_data.process = True
            self.shared_data.save_processed = False 
            self.shared_data.save_raw = False
            self.busy.set_text(" Online ")

    def BT_training(self, event):
        print("Start recording")
        self.shared_data.save_raw = True
        self.shared_data.save_processed = True
        self.shared_data.process = True
        self.training = True
        self.start_time = time.time_ns()//1_000_000
   
    def BT_ShowControl(self, event):
        if self.Leap_streaming or self.Leap_streaming:
            self.shared_data.show_us = False
            self.Main_screen(screen= "Joystick", first_time = False)

    def BT_ShowLeap(self, event):
        if not self.Leap_streaming:
            self.Main_screen(screen= "Leap", first_time = False)


    def BT_ShowUS(self, event):
        if not self.US_streaming:
            self.shared_data.show_us = True
            self.Main_screen(screen= "US", first_time = False)

    def BT_Channel(self, val):
        self.channel = int(val - 1)
    
    def Move_Soft_Hand(self, val1, val2):

        if abs(val1 - self.last_values[0]) > 20:
            self.last_values[0] = val1

        if abs(val2 - self.last_values[1]) > 20:
            self.last_values[1] = val2



        scaled_val1 = abs((max(min(self.last_values[0],100), 0) / 100) - 1) * self.limit1
        scaled_val2 = abs((max(min(self.last_values[1],100), 0) / 100) - 1) * self.limit2

        # scaled_val1 = (max(min(val1,100), 0) / 100) * self.limit1
        # scaled_val2 = (max(min(val2,100), 0) / 100) * self.limit2

        # print(scaled_val1,scaled_val2)

        Softhand.controlFuncs.set_motor_inputs(self.softhand_serial, self.ID, int(scaled_val1), int(scaled_val2))
