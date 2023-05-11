import pandas as pd

from time import sleep
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as animation
from functools import partial
import scipy.spatial.qhull


from scipy.integrate import quad
from scipy.spatial.transform import Rotation 
import scipy.optimize

from scipy.interpolate import RBFInterpolator
from scipy.interpolate import LinearNDInterpolator 

import DipoleModel
#import einops as eo

import dill

from data.Sensors import PIMSensor as Sensor

#sensor = Sensor()

sensors = [Sensor(0x1d, i2c_dev=1), Sensor(0x1e, i2c_dev=5), Sensor(0x1e, i2c_dev=1), Sensor(0x1d, i2c_dev=5)]
sensor_positions = np.array([[0, 0, 0], [35, 0, 0], [0, 35, 0], [35, 35, 0]])-np.array([17.5, 17.5, 0])
print(sensor_positions.shape)

fig=plt.figure()
ax=plt.axes(projection="3d")

lim=3000
ax.set_xlim3d([-lim,lim])
ax.set_ylim3d([-lim,lim])
ax.set_zlim3d([-lim,lim])
 
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
point=[0,0,0]


scat = ax.scatter(point[0], point[1], point[2], s = 5) 
linepnts = [((0, -lim), (0, 0), (0,0)), ((0,0), (0, lim), (0,0)),((0,0), (0,0), (0, -lim))]
lines = [ax.plot(line[0], line[1], line[2], "--", color="orange")[0] for line in linepnts]





def animate(model, guess=np.array([0,0,0,0,0, 1])):
    B = np.array([sensor.get_magnetometer() for sensor in sensors])
    X = model(B, sensor_positions, guess=guess)
    
    x, y, z, x_axis, y_axis, z_axis = (round(pos, 3) for pos in X) 
    print(f"{x} | {y} | {z} || {x_axis} | {y_axis} | {z_axis}")
    print(np.round(B, 4))
    return X




if __name__=="__main__":
    #model = Pipeline([('poly', PolynomialFeatures(degree=6)), ('linear', LinearRegression(fit_intercept=False))])
    #data = pd.read_csv("/media/pi/58ba4525-1a76-44b4-9f48-8c15e728a138/0.2mm-2/DataAvg.txt")
    #print(data)
    #with open("position_predictors/PIM_03022023-interpolator.pkl", 'rb') as calibration:
    #    model = dill.load(calibration)

    model = DipoleModel.Dipole(1210, 25.4*3/16, 25.4*2/16)
    #ani = animation.FuncAnimation(fig, partial(animate, model), interval=50)
    guess=np.array([1,0,0,0,0,1])
    while True:
        X=animate(model, guess)
        guess=X
        sleep(0.1)
    
    #plt.show()


#while True:
#    tlv493d.update_data()
#    x = tlv493d.get_x()
#    y = tlv493d.get_y()
#    z = tlv493d.get_z()

#    print("x: ", x, "y: ", y, "z: ",z)
#    sleep(0.5)




