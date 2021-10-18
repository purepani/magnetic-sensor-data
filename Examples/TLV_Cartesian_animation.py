import TLV

from time import sleep
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as animation

tlv493d = TLV.TLV493D()

fig=plt.figure()
ax=plt.axes(projection="3d")

lim=100
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


def animate(i):
    tlv493d.update_data()
    x = tlv493d.get_x()
    y = tlv493d.get_y()
    z = tlv493d.get_z()
    point = [[x],[y],[z]]
    print(point)
    scat._offsets3d = point 
    linepnts = np.array([[[x, -lim], [y, y], [z,z]], [[x,x], [y, lim], [z,z]],[[x,x], [y,y], [z, -lim]]])
    for i, line in enumerate(lines):
        line.set_data(linepnts[i][0:2])
        line.set_3d_properties(linepnts[i][2])

if __name__=="__main__":
    
    ani = animation.FuncAnimation(fig, animate)
    plt.show()
#while True:
#    tlv493d.update_data()
#    x = tlv493d.get_x()
#    y = tlv493d.get_y()
#    z = tlv493d.get_z()

#    print("x: ", x, "y: ", y, "z: ",z)
#    sleep(0.5)




