import numpy as np
import time


class TrainingManager():
    def __init__(self):
        self.current = 0
        self.counter = 0
        self.position = [0, 0, 0, 0]
        self.status = "new"
        self.stop_flag = False
        self.start_time = time.time_ns()//1_000_000
        file = np.loadtxt(self.shared_data.traininng_path, delimiter=",", comments="#")
        self.N_trainings = file.shape[0]
        self.start_point = np.zeros((self.N_trainings,2))
        self.end_point = np.zeros((self.N_trainings,2))
        self.training_times = np.zeros(1000)

        self.start_point[:,0] = file[:,0].astype(int)
        self.end_point[:,0] = file[:,1].astype(int)
        self.start_point[:,1] = file[:,2].astype(int)
        self.end_point[:,1] = file[:,3].astype(int)
        self.delay = (file[:,4]*1000).astype(int)
        self.lenght = (file[:,5]*1000)
        self.stop = file[:,6].astype(int)
        self.fake = file[:,7].astype(int)
    
    def generate_start(self):
        self.start_time = time.time_ns()//1_000_000
        self.startmov_time = time.time_ns()//1_000_000
        if self.delay[self.current] != 0:
            self.ln2.set_data([self.start_point[self.current, 0], 0], [self.start_point[self.current, 1], 0])
            self.busy.set_text("Prep. - " + str((self.delay[self.current]/1000)) + " s")

    def update_training(self):
        self.time_elapsed = time.time_ns()//1_000_000 - self.start_time
        if self.time_elapsed > self.delay[self.current]:
            if self.time_elapsed > self.delay[self.current] + self.lenght[self.current]:
                if self.status == "update":
                    self.training_times[int(self.counter)] = self.startmov_time
                    self.training_times[int(self.counter+1)] = time.time_ns()//1_000_000
                    self.counter += 2
                    self.status = self.generate_new()

                if self.status == "new":
                    self.generate_start()
                    self.time_elapsed = time.time_ns()//1_000_000 - self.start_time
                    self.status = "update"

                elif self.status == "stop":
                    if self.stop_flag == False:
                        self.shared_data.process = True
                        self.shared_data.save_img = True
                        self.current += 1
                        self.status = "new"
                    self.busy.set_text("Waiting for input")
                        
                elif self.status == "end":
                    self.training = False
                    self.shared_data.train = True
                    self.shared_data.save_raw = False
                    self.shared_data.save_img = False
                    self.position = [0, 0]
                    self.shared_data.training_times = self.training_times
                    self.busy.set_text("")
                    self.ln2.set_data([self.position[0], 0], [self.position[1], 0])
                    self.status = "update"
            else:
                self.generate_next_step()
                self.busy.set_text("Follow the Line")
                self.ln2.set_data([self.position[0], 0], [self.position[1], 0])
        else:
            self.startmov_time = time.time_ns()//1_000_000 # update start time if still on the delay period
        return self.position
    
    def generate_next_step(self):
        w = (self.time_elapsed - self.delay[self.current]) / self.lenght[self.current]
        self.position =  self.start_point[self.current, :] *  (1 - w) + self.end_point[self.current, :] * w

    def generate_new(self):
        if (self.current + 1) == self.N_trainings:
            self.current = 0
            return "end"
        else:
            if self.stop[self.current + 1] == 1:
                self.stop_flag = True
                self.shared_data.process = False
                self.shared_data.save_img = False
                # self.change_text("Resting - Await further guidance", "#DFDF60")
                return "stop"
            self.current += 1
            return "new"

    def generate_training_array(self, tr_number, ts_array):
        self.current = tr_number
        tr_array = []
        ts_array = ts_array + self.delay[self.current] # add delay
        for ts in ts_array:
            self.time_elapsed = ts
            self.generate_next_step()
            tr_array.append(self.position) 
        self.current = 0
        return tr_array, self.fake[tr_number]