import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



folder = '0.5mm'

step = 0.5 
limx = [0,12.5]
limy = [0,3]
numx = int(np.round(limx[1]/step) )+1
numy = int(np.round(limy[1]/step))+1

print(f"({numx}, {numy})")

xx = np.linspace(limx[0],limx[1], numx)
yy = np.linspace(limy[0],limy[1], numy) 

print(xx)
zz = np.array([12.5])

shape = [xx.shape[0], yy.shape[0], zz.shape[0], 3]

def get_avgs(xx, yy, zz, i, j, k, M):
    x, y, z = xx[i], yy[j], zz[k] 
    data = pd.read_csv(f"{folder}/{x}-{y}-{z}.data")
    mag = data.loc[:,'Magnetometerx':'Magnetometerz'].set_axis(['Mx','My','Mz'],axis=1)
    mag_avg = mag.mean()
    return mag_avg[M]

data = np.zeros(shape)

for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            for m in range(shape[3]):
                data[i, j, k, m] = get_avgs(xx, yy, zz, i, j, k, m)

print(data.shape)
z =  int(input("z: ") or "0")

fig, axs = plt.subplots(2, 3)
magnetic_labels = ["Mx", "My", "Mz"]
position_labels = ["x", "y"]

pos = [xx, yy]


for i in range(2):
    for j in range(3):
        data_xx_yy = [data[:,:,z, j], data[:, :, z, j].T]
        axs[i, j].plot(pos[i], data_xx_yy[i])
        axs[i, j].set(xlabel=position_labels[i])
        axs[i, j].set_title(magnetic_labels[j])
        axs[i, j].legend(labels=pos[i])
plt.show()
