import pandas as pd
import numpy as np

from Sensors import PIMSensor as Sensor

def record_magnetic_field_measurements(num_measurements=5):
    measurements = np.zeros((num_measurements, 3))

    for i in range(num_measurements):
        B = np.array(sensor.get_magnetometer())
        measurements[i] = [B[0], B[1], B[2]]
    return measurements

def calculate_average(measurements):
    return np.mean(measurements, axis=0)


sensor = Sensor(address = 0x1d, i2c_dev=1)

num_measurements = 5

bg_measurement = record_magnetic_field_measurements(num_measurements)

background_value = np.array([-0.0111, -0.0102, -0.0450])

print(f"{background_value}")
input("Press enter when magnet is in position")

measurements_list = []
averages_list = []

while True:
	magnetic_field_measurements = record_magnetic_field_measurements(num_measurements)
	average_magnetic_field = calculate_average(magnetic_field_measurements)

	# Subtract the background value from each average
	average_magnetic_field -= background_value

	measurements_list.append(magnetic_field_measurements)
	averages_list.append(average_magnetic_field)

	data_measurements = {
		"Measurement (x, y, z)": [meas.tolist() for meas in measurements_list]
	}

	data_averages = {
		"Average (x, y, z)": [avg.tolist() for avg in averages_list]
	}

	df_measurements = pd.DataFrame(data_measurements)
	df_averages = pd.DataFrame(data_averages)

	print("\nRecorded Measurements:\n")
	print(df_measurements)

	print("\nComputed Averages:\n")
	print(df_averages)

	repeat = input("\nDo you want to repeat the recording? (y/n): ").lower()
	if repeat != "y":
		break

    # Save averages to CSV files
output_measurements_file = input("Enter the measurements output file name (e.g., recorded_measurements.csv): ")
output_averages_file = input("Enter the averages output file name (e.g., computed_averages.csv): ")

df_measurements.to_csv(output_measurements_file, index=False)
df_averages.to_csv(output_averages_file, index=False)

print(f"\nRecorded measurements saved to '{output_measurements_file}'.")
print(f"Computed averages saved to '{output_averages_file}'.")
