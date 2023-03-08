import pandas as pd

from time import sleep
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as animation
from functools import partial
import scipy.spatial.qhull

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from scipy.integrate import quad
from scipy.spatial.transform import Rotation 
import scipy.optimize

from scipy.interpolate import RBFInterpolator
from scipy.interpolate import LinearNDInterpolator 
#import einops as eo

import dill

from data.Sensors import PIMSensor as Sensor

sensor = Sensor()

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

def poly_fit(file): 
    df = pd.read_csv(file)
    print(df)
    X = np.array(df.loc[:, "x":"z"]) # unpacks to x, y, z
    M = np.array(df.loc[:, "Mx":"Mz"])

    print(X.shape, M.shape)
    max = 5
    samples = 20
    xx, yy, zz = np.linspace(-max, max, samples), np.linspace(-max, max, samples), np.linspace(-max, max, samples)
    Xfit = np.meshgrid(xx, yy, zz)

    Xfit = np.array([eo.rearrange(x, "x y z -> (x y z)") for x in Xfit]).T

    print(Xfit.shape)

    model = Pipeline([('poly', PolynomialFeatures(degree=6)), ('linear', LinearRegression(fit_intercept=False))])
    model = model.fit(M, X)
    return model



def animate(model, i):
    x, y, z = sensor.get_magnetometer()
    print(x, y, z)
    step = 20
    point = np.array([x,y,z])
    #xbool = (data["Mx"]  < x+step) & (data["Mx"]>x-step)
    #ybool = (data["My"]  < y+step) & (data["My"]>y-step)
    #zbool = (data["Mz"]  < z+step) & (data["Mz"]>z-step)
    #print(data[xbool & ybool & zbool].loc[:, "x":"z"])    
    #X = model.predict(point[np.newaxis, :]) 
    X,  = model(point)
    x, y, z = (round(pos, 3) for pos in X) 
    print(x, y, z)
    #scat._offsets3d = point 
    #linepnts = np.array([[[x, -lim], [y, y], [z,z]], [[x,x], [y, lim], [z,z]],[[x,x], [y,y], [z, -lim]]])
    #for i, line in enumerate(lines):
    #    line.set_data(linepnts[i][0:2])
    #    line.set_3d_properties(linepnts[i][2])

if __name__=="__main__":
    #model = Pipeline([('poly', PolynomialFeatures(degree=6)), ('linear', LinearRegression(fit_intercept=False))])
    #data = pd.read_csv("/media/pi/58ba4525-1a76-44b4-9f48-8c15e728a138/0.2mm-2/DataAvg.txt")
    #print(data)
    with open("position_predictors/PIM_03022023-interpolator.pkl", 'rb') as calibration:
        model = dill.load(calibration)
    ani = animation.FuncAnimation(fig, partial(animate, model), interval=50)
    plt.show()
#while True:
#    tlv493d.update_data()
#    x = tlv493d.get_x()
#    y = tlv493d.get_y()
#    z = tlv493d.get_z()

#    print("x: ", x, "y: ", y, "z: ",z)
#    sleep(0.5)




