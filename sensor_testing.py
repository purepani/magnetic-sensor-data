import time
import statistics
import board
import adafruit_mlx90393

def take_measurements(num_measurements, filter_value, oversampling_value):
    # Initialize the I2C bus and magnetometer sensor
    i2c = board.I2C()
    sensor = adafruit_mlx90393.MLX90393(i2c)

    # Set user-defined filter and oversampling values
    sensor.filter = filter_value
    sensor.oversampling = oversampling_value

    # Initialize lists to store magnetic field components
    magnetic_fields_x = []
    magnetic_fields_y = []
    magnetic_fields_z = []

    # Record the start time
    start_time = time.time()

    # Take measurements
    for _ in range(num_measurements):
        # Read magnetic field components
        mag_x, mag_y, mag_z = sensor.magnetic
        
        magnetic_fields_x.append(mag_x)
        magnetic_fields_y.append(mag_y)
        magnetic_fields_z.append(mag_z)

    # Record the end time
    end_time = time.time()

    # Calculate average and standard deviation
    avg_x = statistics.mean(magnetic_fields_x)
    avg_y = statistics.mean(magnetic_fields_y)
    avg_z = statistics.mean(magnetic_fields_z)

    std_dev_x = statistics.stdev(magnetic_fields_x)
    std_dev_y = statistics.stdev(magnetic_fields_y)
    std_dev_z = statistics.stdev(magnetic_fields_z)

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the results
    print(f"Average Magnetic Field (X): {avg_x}")
    print(f"Average Magnetic Field (Y): {avg_y}")
    print(f"Average Magnetic Field (Z): {avg_z}")
    print(f"Standard Deviation (X): {std_dev_x}")
    print(f"Standard Deviation (Y): {std_dev_y}")
    print(f"Standard Deviation (Z): {std_dev_z}")
    print(f"Total Elapsed Time: {elapsed_time} seconds")

# Get user input for filter and oversampling values
filter_value = int(input("Enter the filter value: "))
oversampling_value = int(input("Enter the oversampling value: "))

# Get the number of measurements from the user
num_measurements = int(input("Enter the number of measurements: "))

# Call the function to take measurements and display results
take_measurements(num_measurements, filter_value, oversampling_value)
