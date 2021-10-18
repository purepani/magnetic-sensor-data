# SPDX-FilehjCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT 

import time
import board
import adafruit_bno055
import numpy as np
import pandas as pd
import os


i2c = board.I2C()
sensor = adafruit_bno055.BNO055_I2C(i2c)

# If you are going to use UART uncomment these lines
# uart = board.UART()
# sensor = adafruit_bno055.BNO055_UART(uart)

last_val = 0xFFFF


def temperature():
    global last_val  # pylint: disable=global-statement
    result = sensor.temperature
    if abs(result - last_val) == 128:
        result = sensor.temperature
        if abs(result - last_val) == 128:
            return 0b00111111 & result
    last_val = result
    return result

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


while True:
    #print("Accelerometer (m/s^2): {}".format(sensor.acceleration))
    #print("Magnetometer (microteslas): {}".format(sensor.magnetic))
    #print("Gyroscope (rad/sec): {}".format(sensor.gyro))
    #print("Euler angle: {}".format(sensor.euler))
    #print("Quaternion: {}".format(sensor.quaternion))
    #print("Linear acceleration (m/s^2): {}".format(sensor.linear_acceleration))
    #print("Gravity (m/s^2): {}".format(sensor.gravity))
    #print(type(sensor.gravity))
    #time.sleep(1)
    file_name= input("Give new name for file:")
    row_data=[]
    for i in range(samples_per_measurement):
        row=sensor.acceleration + sensor.magnetic + sensor.gyro + sensor.euler + sensor.quaternion+sensor.linear_acceleration+sensor.gravity
        row_data.append(list(row)) 
        time.sleep(min_time_between_samples)
    data=pd.DataFrame(row_data,columns=columns)
    data.to_csv(f"{folder_name}/{file_name}")
