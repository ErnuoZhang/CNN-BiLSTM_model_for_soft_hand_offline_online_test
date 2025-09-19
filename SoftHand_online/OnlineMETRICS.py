import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

pred = np.load(r"C:\Users\11952\Desktop\softhand_results\online test\xian\preds.npy")
true = np.load(r"C:\Users\11952\Desktop\softhand_results\online test\xian\leap.npy")

pred_joint1 = pred[::2]
pred_joint2 = pred[1::2]


true_joint1 = ((true[::2]-112)/(193-112))*100
true_joint2 = ((true[1::2]-90)/(200-90))*100

data_to_save = np.column_stack((pred_joint1, true_joint1, pred_joint2, true_joint2))
np.savetxt("online_metrics.csv", data_to_save, delimiter=",", header="pred1,true1,pred2,true2", comments="")

plt.figure()
plt.subplot(2,1,1)
plt.plot(pred_joint1)
plt.plot(true_joint1)
plt.subplot(2,1,2)
plt.plot(pred_joint2)
plt.plot(true_joint2)
plt.show()