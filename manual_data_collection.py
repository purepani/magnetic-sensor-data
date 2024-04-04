import os
import pandas as pd
import numpy as np
from git import Repo

from Sensors import MLXSensor as Sensor

def record_magnetic_field_measurements(sensor, num_measurements=5):
    measurements = np.zeros((num_measurements, 3))

    for i in range(num_measurements):
        B = np.array(sensor.get_magnetometer())
        measurements[i] = [B[0], B[1], B[2]]
    return measurements

def calculate_average(measurements):
    return np.mean(measurements, axis=0)

def commit_and_push(repo, folder_name):
    repo.git.add(A=True)
    repo.index.commit(f"Add data from {folder_name}")
    origin = repo.remote()
    origin.push()

folder_name = input("Enter a folder name to save the data: ")

if not os.path.exists(folder_name):
    os.mkdir(folder_name)

repo = Repo(os.getcwd())  # Use the current working directory as the repository path

# Create instances for four sensors
#magnet started from (-2,2) and moved to (2,-2) in (x,y)
sensor1 = Sensor(address=0x10) #top left
#sensor1 is the second quadrant, multiply b_x and b_y by negative 1
sensor2 = Sensor(address=0x11) #top right
#sensor2 is in the first quadrant
sensor3 = Sensor(address=0x12) #bottom left
#sensor3 is in the third quadrant, multiply b_x and b_y by negative 1
sensor4 = Sensor(address=0x13) #bottom right
#sensor4 is in the fourth quadrant

# Define a list of sensor addresses
sensor_addresses = [0x10, 0x11, 0x12, 0x13]

# Create instances for each sensor using the list of addresses
sensors = []

for addr in sensor_addresses:
    while True:
        try:
            sensor = Sensor(address=addr, gain=4)
            # If the sensor is successfully created, break out of the loop
            break
        except Exception as e:
            print(f"Error connecting to sensor at address {addr}: {e}")
            print("Retrying...")
            # You can add a delay here if needed, e.g., time.sleep(1)

    sensors.append(sensor)

num_measurements = 10

bg_measurements = [record_magnetic_field_measurements(sensor, num_measurements) for sensor in sensors]
background_values = [calculate_average(bg_measurement) for bg_measurement in bg_measurements]

print("Background Values:")
for i, bg_value in enumerate(background_values):
    print(f"Sensor {i + 1}: {bg_value}")

input("Press enter when magnets are in position")

measurements_lists = [[] for _ in range(len(sensors))]
averages_lists = [[] for _ in range(len(sensors))]

while True:
    for i, sensor in enumerate(sensors):
        magnetic_field_measurements = record_magnetic_field_measurements(sensor, num_measurements)
        average_magnetic_field = calculate_average(magnetic_field_measurements)

        # Subtract the background value from each average
        average_magnetic_field -= background_values[i]

        measurements_lists[i].append(magnetic_field_measurements)
        averages_lists[i].append(average_magnetic_field)

    for i, sensor in enumerate(sensors):
        data_measurements = {
            f"Sensor {i + 1} Measurement (x, y, z)": [meas.tolist() for meas in measurements_lists[i]]
        }

        data_averages = {
            f"Sensor {i + 1} Average (x, y, z)": [avg.tolist() for avg in averages_lists[i]]
        }

        df_measurements = pd.DataFrame(data_measurements)
        df_averages = pd.DataFrame(data_averages)

        measurements_output_file = os.path.join(folder_name, f"sensor_{i + 1}_measurements.csv")
        averages_output_file = os.path.join(folder_name, f"sensor_{i + 1}_averages.csv")

        df_measurements.to_csv(measurements_output_file, index=False)
        df_averages.to_csv(averages_output_file, index=False)

    print("\nRecorded Measurements:\n")
    print(df_measurements)

    print("\nComputed Averages:\n")
    print(df_averages)

    repeat = input("\nDo you want to repeat the recording? (y/n): ").lower()
    if repeat == "n":
        break

print(f"Data for each sensor saved to '{folder_name}' folder.")

# Commit and push changes to the Git repository
commit_and_push(repo, folder_name)

print("Data pushed to the git repository")
