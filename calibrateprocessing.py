from genericpath import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import axes3d
import multiprocessing as mp
import os
from functools import partial

def get_stats(data):
    mag = data.loc[:,'Magnetometerx':'Magnetometerz'].set_axis(['Mx','My','Mz'],axis=1)
    mag_avg = mag.mean() 
    mag_std = mag.std()
    return mag_avg, mag_std


folder = input("Enter folder with data: ").strip()
files = os.listdir(folder)
files.remove("info.txt")
file_name = "DataAvg.txt"
print(f"There are {len(files)} files to parse.")


data_dict = {"x": [], "y":[] ,"z": [], "Mx":[], "My":[], "Mz":[], "Mx_std": [], "My_std":[], "Mz_std":[]}

magnetic_labels = ["Mx", "My", "Mz"]
position_labels = ["x", "y", "z"]
def get_stats_from_file(folder, file):
    position_labels = ["x", "y", "z"]
    #print(f"{folder}/{file}")
    try:
        data = pd.read_parquet(f"{folder}/{file}")
    except:
        print(f"{folder}/{file} failed to read.") 
    file = file.strip(".data")
    coord_str = file.split("_")
    coord = {k:float(v) for k, v in zip(position_labels, coord_str)} 

    avg, std = get_stats(data)
    return coord, avg, std
    #for pos in position_labels:
    #    data_dict[pos].append(coord[pos])
    #    data_dict[f"M{pos}"].append(avg[f"M{pos}"])
    #    data_dict[f"M{pos}_std"].append(std[f"M{pos}"])

def read_data(folder, files, position_labels, name = "DataAvg.txt"):
    i = 0
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for coord, avg, std in pool.imap_unordered(partial(get_stats_from_file, folder), files):
            for pos in position_labels:
                data_dict[pos].append(coord[pos])
                data_dict[f"M{pos}"].append(avg[f"M{pos}"])
                data_dict[f"M{pos}_std"].append(std[f"M{pos}"])
            i = i + 1
            if i%100 == 0:
                print(i)
            #del data

    df = pd.DataFrame(data_dict)
    df.to_csv(f"{folder}/{name}")
    return df

if os.path.exists(f"{folder}/{file_name}"): 
    df = pd.read_csv(f"{folder}/{file_name}") 
else:
    df = read_data(folder, files, position_labels, file_name)

print(df)



print(f"The valid values of the axes are: {position_labels}")
axis = input("Enter axis: ")
while(not axis in position_labels):
    print(f"Value not valid. The valid values of the axes are: {position_labels}")
    axis = input("Enter axis: ")

axis_unique = df[axis].unique()
axis_unique.sort()


print(f"The valid values of {axis} are:{axis_unique}")
axis_val = float(input(f"Enter {axis}: "))

while(not axis_val in axis_unique):
    print(f"Value not valid. The valid values of {axis} are:{axis_unique}")
    axis_val = float(input(f"Enter {axis}: "))

df_z = df[df[axis]==axis_val]
df_z.sort_values(by= ['z', 'y', 'x']) 




fig, axs = plt.subplots(1, 3)
new_labels = position_labels.copy()
new_labels.remove(axis)

for i in range(1):
    for j in range(3):
        ax = axs[j]
        M = magnetic_labels[j]
        pos = position_labels[j]
        plane_data = df_z.pivot(columns=new_labels[0], index = new_labels[1], values = M).sort_index(axis=0).sort_index(axis=1)
        sns.heatmap(plane_data, ax=ax)
        ax.set_xlabel(new_labels[0])
        ax.set_ylabel(new_labels[1])
        ax.set_title(M)
plt.show()

step = int(input("Enter step for vector field: "))

x, y = df_z.iloc[::step]["x"], df_z.iloc[::step]['y'], 
Mx, My, Mz = df_z.iloc[::step]["Mx"], df_z.iloc[::step]['My'], df_z.iloc[::step]['Mz']
Mnorm = np.sqrt(Mx**2+My**2 + Mz**2)
Mxdir, Mydir = Mx/Mnorm, My/Mnorm 
Mlog = np.log10(Mnorm-Mnorm.min()+1)
Mxlog, Mylog = Mxdir*Mlog, Mydir*Mlog

colormap='jet'
#plt.quiver(x, y, Mxlog, Mylog, Mnorm, cmap=colormap, norm = LogNorm(vmin = Mnorm.min(), vmax = Mnorm.max()))
#looks cool
#plt.quiver(x, y, Mxlog, Mylog, Mnorm, cmap=colormap, scale=50)

plt.quiver(x, y, Mxlog, Mylog, Mz, cmap=colormap, scale=100)
plt.colorbar()
plt.ylabel("y")
plt.xlabel("x")
plt.show()




df_z = df[(df["x"]==0) & ((df["y"]==0))]
df_z.sort_values(by= ['z', 'y', 'x']) 
z = df_z["z"]
Bz = df_z["Mz"]
plt.plot(z, Bz, "bo")
plt.show()

