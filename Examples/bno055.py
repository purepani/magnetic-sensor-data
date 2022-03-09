# SPDX-FilehjCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT 
from printrun.printcore import printcore
import time
import board
import adafruit_bno055
import numpy as np
import pandas as pd
import os
import sys


def getNumSamples(lims, step):
    return int(np.round((lims[1]-lims[0])/step)) + 1


def temperature():
    global last_val  # pylint: disable=global-statement
    result = sensor.temperature
    if abs(result - last_val) == 128:
        result = sensor.temperature
        if abs(result - last_val) == 128:
            return 0b00111111 & result
    last_val = result
    return result

def move_printer(printer, x, y, z):
     command = f"G01 X{x} Y{y} Z{z}"
    print(command)
    printer.send(command)

i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c)
printer = printcore("/dev/serial/by-id/usb-UltiMachine__ultimachine.com__RAMBo_7403333303735140D070-if00", 250000)


print("Waiting for Printer Connection")
while not printer.online:
    pass
print("Connected")


# If you are going to use UART uncomment these lines
# uart = board.UART()
# sensor = adafruit_bno055.BNO055_UART(uart)

last_val = 0xFFFF



folder_name=input("Enter Folder name:" )
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

samples_per_measurement = 100
#note, time could be longer based on the functions getting data from the board
min_time_between_samples = 0.01

column_categories = ['Accelerometer', 'Magnetometer', 'Gyroscope', 'EulerAng', 'Quaternion', 'LinearAcceleration', 'Gravity']

columns = ['SampleID'] + [x+f'{coordinate}' for x in column_categories for coordinate in ['x','y','z']] 

for x in columns:
    print(x)


step = float(input("Enter step: "))
xlims = [int(input("Enter xmin: ")), int(input("Enter xmax: "))]
ylims = [int(input("Enter ymin: ")), int(input("Enter ymax: "))]
zlims = [int(input("Enter zmin: ")), int(input("Enter zmax: "))]

lims = xlims + ylims + zlims + [step]
str_lims = [str(x) for x in lims]
lims_to_string = " ".join(str_lims)
with open(f"{folder}/info.txt}"):
    f.writeline("xlim_min xlims_max ylim_min ylim_max zlim_min zlim_max step")
    f.writeline(lims_to_string)


try:
    if zlims[0]<0 or zlims[1]<0:
        raise ValueError("z lims can't be less than 0") 
    print("hi")
except ValueError:
    raise
    sys.exit()

move_printer(printer, xlims[0], ylims[0], zlims[0])


xx = np.linspace(xlims[0], xlims[1], getNumSamples(xlims, step))
yy = np.linspace(ylims[0], ylims[1], getNumSamples(ylims, step))
zz = np.linspace(zlims[0], zlims[1], getNumSamples(zlims, step))

#coords = np.array(np.meshgrid(xx, yy, zz)).T.reshape(-1, 3)


input("Press enter when printer is done moving")
for z in zz:
    for x in xx:
        for y in yy:
            coord = [x, y, z]
            x_str, y_str, z_str = (str(np.round(i, 1)) for i in coord)
            
            move_printer(printer, x_str, y_str, z_str)
            time.sleep(1)
            #print("Accelerometer (m/s^2): {}".format(sensor.acceleration))
            #print("Magnetometer (microteslas): {}".format(sensor.magnetic))
            #print("Gyroscope (rad/sec): {}".format(sensor.gyro))
            #print("Euler angle: {}".format(sensor.euler))
            #print("Quaternion: {}".format(sensor.quaternion))
            #print("Linear acceleration (m/s^2): {}".format(sensor.linear_acceleration))
            #print("Gravity (m/s^2): {}".format(sensor.gravity))
            #print(type(sensor.gravity))
            #time.sleep(1)
            #file_name= input("Give new name for file:") + ".data"
            file_name = f"{x_str}_{y_str}_{z_str}.data"
            row_data=[]
            for i in range(samples_per_measurement):
                try:
                    row=sensor.acceleration + sensor.magnetic + sensor.gyro + sensor.euler + sensor.quaternion+sensor.linear_acceleration+sensor.gravity
                    row_data.append(list(row)) 
                    time.sleep(min_time_between_samples)
                except:
                    i2c = board.I2C()
                    sensor = adafruit_bno055.BNO055_I2C(i2c)

            data=pd.DataFrame(row_data,columns=columns)
            data.to_csv(f"{folder_name}/{file_name}")
