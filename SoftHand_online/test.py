import matplotlib.pyplot as plt
import numpy as np
import Processing.USCruncher 


# data = np.load("Data/25/_leap_angles.npy")[:,[1,3]]

# data = (data - np.amin(data)) / (np.amax(data)  - np.amin(data))

# plt.plot(data)
# plt.show()



# data = np.load("Data/25/_raw_US.npy")[101,...]
# print(data.shape)



# offset = 1
# us_diagonal = np.zeros((1, 16, 2996))

# for c in range(16):
#     us_diagonal[0, c, :] = data[c, c, :]

# us_diagonal = Processing.USCruncher.time_gain_compensation(us_diagonal)

# us_diagonal = Processing.USCruncher.gaussian_smooth(us_diagonal)

# us_diagonal = Processing.USCruncher.apply_hilbert(us_diagonal, pad_len=100)

# us_diagonal = Processing.USCruncher.apply_log_compression(us_diagonal)

# us_diagonal = us_diagonal[:, :, 200:-20]

# # us_diagonal = us_diagonal.astype("float32")

# data = np.load("Data/25/_feature_US.npy")
# print(data.shape)
# print(data[1000,0,1000:1005])


# # plt.subplot(1,2,1)
# # plt.imshow(us_diagonal[0,:,:],aspect=50)

# # plt.subplot(1,2,2)
# plt.imshow(data[1001,:,:],aspect=50)
# plt.show()



# plt.figure()
# for x in range(16):
#     plt.plot(us_diagonal[0,x,:] - (offset*x))
# plt.show()




# # print(self.Features.shape)

# plt.imshow(data[500,:,:],aspect=50)

# # for x in range(16):
# #     plt.plot(data[10,x,:])
# plt.show()






# data = np.load("Data/20/_feature_US.npy")
# print(data.shape)

# plt.imshow(data[500,:,:],aspect=50)

# # for x in range(16):
# #     plt.plot(data[10,x,:])
# plt.show()




pred = np.load("pred_test.npy")
true = np.load("true_test.npy")

plt.figure()
plt.subplot(2,1,1)
plt.plot(true[:,0])
plt.plot(pred[:,0], linestyle="--")
plt.subplot(2,1,2)
plt.plot(true[:,1])
plt.plot(pred[:,1], linestyle="--")
plt.show()









# # data = np.load("Data/Pilot2_Ernuo/Trial_1/2/_leap_angles.npy")
# data = np.load("Data/Pilot2_Ernuo/Trial_1/2/_Raw_US.npy")

# print(data.shape)

# data2 = np.empty((data.shape[0],data.shape[1],data.shape[3]))
# for x in range(16):
#     data2[:,x,:] = data[:,x,x,:] 
    
# np.savez_compressed("Data/Pilot2_Ernuo/Trial_1/2/_USzip.npy", data2)


# # for x in [1,4]:
# #     plt.plot(data[10,x,:]) 
# # plt.show()

# data = np.load("Data/Pilot2_Ernuo/Trial_1/0/_USzip.npy.npz")["arr_0"]

# # print(data["arr_0"].shape)

# # for x in [1,4]:
# #     plt.plot(data["arr_0"][10,x,:]) 
# # plt.show()


