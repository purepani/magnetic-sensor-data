import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



folder = '0.5mm'

step = 0.5 
limx = [0,12.5]
limy = [0,3.0]

xx = np.arange(limx[0],limx[1], step)
yy = np.arange(limy[0],limy[1], step) 
zz = np.array([12.5])

shape = [xx.shape[0], yy.shape[0], zz.shape[0], 3]

def get_avgs(xx, yy, zz, i, j, k, M):
    x, y, z = xx[i], yy[j], zz[k] 
    data = pd.read_csv(f"{folder}/{x}-{y}-{z}.data")
    mag = data.loc[:,'Magnetometerx':'Magnetometerz'].set_axis(['Mx','My','Mz'],axis=1)
    print(mag.size)
    mag_avg = mag.mean()
    return mag_avg[M]

data = np.zeros(shape)

for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            for m in range(shape[3]):
                data[i, j, k, m] = get_avgs(xx, yy, zz, i, j, k, m)

x, y, z = int(input("x: ") or "0"), int(input("y: ") or "0"), int(input("z: ") or "0")
point = (x, y, z)
plt.plot(xx, data[:, point[1], point[2], 0])
plt.plot(yy, data[point[0], :, point[2], 0])
plt.title("Mx")
plt.show()

plt.plot(xx, data[:, point[1], point[2], 1])
plt.plot(yy, data[point[0], :, point[2], 1])
plt.title("My")
plt.show()

plt.plot(xx, data[:, point[1], point[2], 2])
plt.plot(yy, data[point[0], :, point[2], 2])
plt.title("Mz")
plt.show()
print(data)



#print(f"({x},{y}) - ({mag_avg['Mx']}, {mag_avg['My']}, {mag_avg['Mz']})"ata


