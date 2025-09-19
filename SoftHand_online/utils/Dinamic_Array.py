import numpy as np

class DinamicArray:
    def __init__(self, init_capacity=100, shape=(0,), capacity_multiplier=4, dtype=float):
        """First item of shape is ingnored, the rest defines the shape"""
        self.shape = shape
        self.data = np.zeros((init_capacity,*shape[1:]),dtype=dtype)
        self.capacity = init_capacity
        self.capacity_multiplier = capacity_multiplier
        self.size = 0
    
    def get_size(self):
        return self.size

    def update(self, x):
        if self.size == self.capacity:
            self.capacity *= self.capacity_multiplier
            newdata = np.zeros((self.capacity,*self.data.shape[1:]))
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size] = x
        self.size += 1

    def finalize(self):
        return self.data[:self.size]
    
    def save(self, path):
        np.save(path + "_1", self.data[:int(self.size/2)])
        np.save(path + "_2", self.data[int(self.size/2):int(self.size)])



def find_nearest_idx(array, value):
    array = np.ravel(array)
    idx = (np.abs(array - value)).argmin()
    return idx