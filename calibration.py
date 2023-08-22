# SPDX-FilehjCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT 
from printrun.printcore import printcore
import time
from time import sleep
import numpy as np
import pandas as pd
import os
import sys
import signal
import einops as eo
import tqdm

from Sensors import PIMSensor as Sensor

import git
from github import Github

ACCESS_TOKEN = input("Enter Temporary Access token: ")

class mean_tracker:
    def __init__(self, mean, num):
        self.mean = mean
        self.num = num
    def update_mean(self, samp):
        self.num = self.num +1 
        self.mean = (self.mean*(self.num-1) + samp)/self.num

def print_time(t, axis):
    print(f"Estimate time remaining based on {axis} samples: {round(t/(60*60), 2)} hours, or {round(t/(60*60*24), 2)} days.")
def getNumSamples(lims, step):
    return int(np.round((lims[1]-lims[0])/step)) + 1


def move_printer(printer,  x, y, z):
    command = f"G01 X{x} Y{y} Z{z}"
    print(command)
    printer.send(command)

def git_push(repo, message, foldername):
    repo.git.add(all=True)
    repo.index.commit(message)
    origin = repo.remote(name='origin')
    origin.push()
    return None 

sensor = Sensor(0x1e)
sensors = [Sensor(0x1d, i2c_dev=1), Sensor(0x1e, i2c_dev=5), Sensor(0x1e, i2c_dev=1), Sensor(0x1d, i2c_dev=5)]
sensor_positions = np.array([[0., 0., 0.], [35., 0., 0.], [0., 35., 0.], [35., 35., 0.]])

printer = printcore("/dev/serial/by-id/usb-Prusa_Research__prusa3d.com__Original_Prusa_i3_MK2_CZPX1017X003XC14071-if00", 115200)


print("Waiting for Printer Connection")
while not printer.online:
    pass
print("Connected")



repo = git.Repo("./.git")
folder_name=input("Enter Folder name:" )

if folder_name not in repo.branches:
    repo.create_head(folder_name)
repo.heads[folder_name].checkout()



branch_ref = f"refs/remotes/origin/{folder_name}"
if branch_ref in repo.git.ls_remote("--heads", "origin").splitlines():
    # push the new branch and set upstream
    repo.remote().pull(refspec=f"refs/heads/{folder_name}")


last_val = 0xFFFF



if not os.path.exists(folder_name):
    os.makedirs(folder_name)

samples_per_measurement = 100
#note, time could be longer based on the functions getting data from the board
min_time_between_samples = 0.01

#column_categories = ['Accelerometer', 'Magnetometer', 'Gyroscope', 'EulerAng', 'Quaternion', 'LinearAcceleration', 'Gravity']

column_categories = ['Magnetometer']
#columns = ['SampleID'] + [x+f'{coordinate}' for x in column_categories for coordinate in ['x','y','z']] 

columns = ["Sensor Position"]+[x+f'{coordinate}' for x in column_categories for coordinate in ['x','y','z']] 



for x in columns:
    print(x)

settings_file = "info.txt"
doManualEntry = False
if os.path.exists(f"{folder_name}/{settings_file}"):
    answered_prompt = False
    while not answered_prompt:
        answered_prompt = True
        try:
            settings = pd.read_csv(f"{folder_name}/{settings_file}") 
        except:
            print(f"Reading settings file failed. Do manual entry.")
            doManualEntry =True
            break
        print(f"These are the settings in {settings_file}.")
        with pd.option_context('display.max_columns', None):
            print(settings)
        answer = input(f"Would you like to use settings from {settings_file}? ('y/n): ")
        if answer.casefold() == "y":
            try:
                x_step = settings.iloc[0]["x_step"] 
                y_step = settings.iloc[0]["y_step"] 
                z_step = settings.iloc[0]["z_step"] 
                xmin = settings.iloc[0]["xlim_min"]
                print(xmin)
                xmax = settings.iloc[0]["xlim_max"]
                ymin = settings.iloc[0]["ylim_min"]
                print(ymin)
                ymax = settings.iloc[0]["ylim_max"]
                zmin = settings.iloc[0]["zlim_min"]
                print(zmin)
                zmax = settings.iloc[0]["zlim_max"]
                print("yay")
                xlims = [xmin, xmax]
                ylims = [ymin, ymax]
                zlims = [zmin, zmax]
            except:
                print(f"Parsing settings file failed. Do manual entry.")
                doManualEntry =True
        elif answer.casefold() == "n":
            doManualEntry =True
        else:
            answered_prompt = False
else:
    doManualEntry = True

if doManualEntry:
    x_step = float(input("Enter x_step: "))
    y_step = float(input("Enter y_step: "))
    z_step = float(input("Enter z-step: "))
    xlims = [int(input("Enter xmin: ")), int(input("Enter xmax: "))]
    ylims = [int(input("Enter ymin: ")), int(input("Enter ymax: "))]
    zlims = [int(input("Enter zmin: ")), int(input("Enter zmax: "))]

    lims = xlims + ylims + zlims + [x_step, y_step, z_step] 
    str_lims = [str(x) for x in lims]
    lims_to_string = " ".join(str_lims)

    info_file = f'{folder_name}/info.txt'
    settings_data = pd.DataFrame({
        "xlim_min":xlims[0],
        "xlim_max":xlims[1],
        "ylim_min":ylims[0],
        "ylim_max":ylims[1],
        "zlim_min":zlims[0],
        "zlim_max":zlims[1],
        "x_step": x_step,
        "y_step": y_step,
        "z_step": z_step
        }, index = [0])
    settings_data.to_csv(info_file)

SkipData = True if input("Skip data?(y/n): ")=="y" else False





    # check if the branch exists in the remote repository
branch_ref = f"refs/remotes/origin/{folder_name}"
if branch_ref not in repo.git.ls_remote("--heads", "origin").splitlines():
    # push the new branch and set upstream
    repo.remote().push(refspec=f"refs/heads/{folder_name}", set_upstream=True)
else:
    # push the new branch without setting upstream
    repo.remote().push(refspec=f"refs/heads/{folder_name}")





try:
    if zlims[0]<0 or zlims[1]<0:
        raise ValueError("z lims can't be less than 0") 
    print("hi")
except ValueError:
    raise
    sys.exit()
try:

    
    move_printer(printer, xlims[0], ylims[0], zlims[0])


    xx = np.linspace(xlims[0], xlims[1], getNumSamples(xlims, x_step))
    yy = np.linspace(ylims[0], ylims[1], getNumSamples(ylims, y_step))
    zz = np.linspace(zlims[0], zlims[1], getNumSamples(zlims, z_step))
    total_y_samples = xx.size*yy.size*zz.size
    total_x_samples = xx.size*zz.size
    total_z_samples = zz.size

    #coords = np.array(np.meshgrid(xx, yy, zz)).T.reshape(-1, 3)
    xprev, yprev, zprev = None, None, None
    pos_labels = ["x", "y", "z"]
    mean_times = {pos_labels[i]: mean_tracker(0, 0) for i in range(3)}


    input("Press enter when printer is done moving")
    for z in zz:
        add_z_sleep = True
        start_z = time.time()
        for x in xx:
            add_x_sleep = True
            start_x = time.time()
            for y in yy:
                start_y=time.time()
                coord = [x, y, z]
                x_str, y_str, z_str = (str(np.round(i, 1)) for i in coord)
                file_name = f"{x_str}_{y_str}_{z_str}.data"
                if os.path.exists(f"{folder_name}/{file_name}") and SkipData:
                    continue
                
                move_printer(printer, x_str, y_str, z_str)

                movedUpLayer = not z==zprev and not zprev==None
                hitxEdge = x==xx[0] and xprev==xx[-1] and not xprev == None
                hityEdge = y==yy[0] and yprev==yy[-1] and not yprev == None 
                #sleep_time = 0.5 + 0.5*movedUpLayer + 1.5*hityEdge+0.75*hitxEdge
                sleep_time = 0.5 + 0.5*add_x_sleep + 1.5*add_z_sleep
                add_x_sleep, add_z_sleep = False, False
                print(f"Pausing for {sleep_time} seconds.")
                time.sleep(sleep_time)

                row_data=np.zeros((samples_per_measurement, len(sensors), 3))


                for i in range(samples_per_measurement):
                    try:
                        B = np.array([sensor.get_magnetometer() for sensor in sensors])
                        #tlv493d.update_data()
                        #Bx = tlv493d.get_x()
                        #By = tlv493d.get_y()
                        #Bz = tlv493d.get_z()
                        row_data[i]=B
                        time.sleep(min_time_between_samples)
                    except:
                        print("Measurement Failed")
                        sensor = Sensor()
                        print("Reconnected")
                idx = pd.IndexSlice
    
                cols=pd.MultiIndex.from_product([np.array(["position", "Magnetometer"]), np.array(['x', 'y', 'z'])], names=["values", "axis"])
                ind=pd.MultiIndex.from_product([range(len(sensors)), range(samples_per_measurement)], names=["Sensor", "Sample"])
                data=pd.DataFrame(columns=cols, index=ind)


                #row_data = eo.rearrange(row_data, "samples sensors dim -> dim (sensors samples)")
                for i in range(len(sensors)):
                    data.loc[idx[i, :] , ("Magnetometer", 'x')]=row_data[:, i, 0]
                    data.loc[idx[i, :], ("Magnetometer", 'y')]=row_data[:, i, 1]
                    data.loc[idx[i, :], ("Magnetometer", 'z')]=row_data[:, i, 2]
                    data.loc[idx[i, :] , ("position", 'x')] = sensor_positions[i, 0]
                    data.loc[idx[i, :] , ("position", 'y')] = sensor_positions[i, 1]
                    data.loc[idx[i, :] , ("position", 'z')] = sensor_positions[i, 2]
                data = data.apply(pd.to_numeric)
                print("Done")
                data.to_csv(f"{folder_name}/{file_name}")
                yprev  = y
                end_y = time.time()
                y_time = end_y - start_y
                mean_times["y"].update_mean(y_time)
                estimated_y = mean_times["y"].mean*(total_y_samples-mean_times["y"].num)
                print(f"Num samples = {mean_times['y'].num}") 
                print(f"Mean time of sample = {round(mean_times['y'].mean,2)}") 
                print(f"Most Recent Sample time = {y_time}") 
                print_time(estimated_y, "y")
                del row_data
            xprev = x 
            end_x = time.time()
            mean_times["x"].update_mean(end_x-start_x)
            estimated_x= mean_times["x"].mean*(total_x_samples-mean_times["x"].num) 

            print_time(estimated_x, "x")
        zprev = z
        end_z = time.time()
        mean_times["z"].update_mean(end_z-start_z)
        estimated_z = mean_times["z"].mean*(total_z_samples-mean_times["z"].num)
        print_time(estimated_z, "z")
        try:
            git_push(repo, f"Added the {z} slice to the {folder_name} test", folder_name)
        except:
            print(f"Failed to push the {z} slice to the {folder_name} test")

except:
    move_printer(printer, "0", "0", "0")
    raise

move_printer(printer, "0", "0", "0")
repo.git.checkout("main")
g = Github(ACCESS_TOKEN)
repo_name = f"{repo.remotes.origin.url.split('/')[-2]}/{repo.remotes.origin.url.split('/')[-1].split('.')[0]}".split(":")[1]

print(repo_name)
github_repo = g.get_repo(repo_name)

branch_name = folder_name
input_payload = {f"folder_name": folder_name}

github_repo.get_workflows()[0].create_dispatch(f"refs/heads/{branch_name}", inputs=input_payload)

print("Done")
