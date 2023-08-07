import os
import time
import pandas as pd
from printrun.printcore import printcore
from Sensors import PIMSensor
from git import Repo

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def git_push(repo_path, branch_name):
    try:
        repo = Repo(repo_path)
        repo.git.add(all=True)
        repo.index.commit("Add measurement data and average measurements")
        origin = repo.remote(name='origin')
        origin.push(refspec=branch_name)

    except Exception as e:
        print("Error during Git push: ", e)

def record_measurements(port, baudrate):
    try:
        printer = printcore("/dev/serial/by-id/usb-Prusa_Research__prusa3d.com__Original_Prusa_i3_MK2_CZPX1017X003XC14071-if00", 115200)
        printer.send(f"G91")
        #printer.send(f"G92 X{50} Y{50} Z{50}")
        
        sen1 = PIMSensor(address=0x1e, i2c_dev=1)
        sen2 = PIMSensor(address=0x1d, i2c_dev=1)
        
        # Prompt the user to input the folder name
        user_folder = input("Enter the folder name to save data: ")
        data_directory = os.path.join(os.getcwd(), user_folder)
        create_directory(data_directory)
                                
        print("Connected to the 3D printer. Sensor 1 = 0x1d, i2c 1. Sensor 2 = 0x1e, i2c 5")

        try:
            x_min = float(input("Enter the minimum X-coordinate (mm): "))
            x_max = float(input("Enter the maximum X-coordinate (mm): "))
            x_step = float(input("Enter the step size for X-coordinate (mm): "))

            y_min = float(input("Enter the minimum Y-coordinate (mm): "))
            y_max = float(input("Enter the maximum Y-coordinate (mm): "))
            y_step = float(input("Enter the step size for Y-coordinate (mm): "))

            z_min = float(input("Enter the minimum Z-coordinate (mm): "))
            z_max = float(input("Enter the maximum Z-coordinate (mm): "))
            z_step = float(input("Enter the step size for Z-coordinate (mm): "))
            
            # Correct negative minimum values and adjust the maximum values accordingly
            if x_min < 0:
                x_max += abs(x_min)
                printer.send(f"G92 X{abs(x_min)}")
                x_min = 0

            if y_min < 0:
                y_max += abs(y_min)
                printer.send(f"G92 Y{abs(y_min)}")
                y_min = 0

            if z_min < 0:
                z_max += abs(z_min)
                printer.send(f"G92 Z{abs(z_min)}")
                z_min = 0
            
        except ValueError:
            print("Invalid input. Please enter numeric values for coordinates and step size.")
            return
                
        # Lists to store the magnetic field measurements from Sensor 1 and Sensor 2
        measurements_sensor1 = []
        measurements_sensor2 = []

        for z in range(int(z_min), int(z_max) + 1, int(z_step)):
            for y in range(int(y_min), int(y_max) + 1, int(y_step)):
                for x in range(int(x_min), int(x_max) + 1, int(x_step)):
					
                    # Move the printer head to the specified coordinates
                    printer.send("G1 X{:.2f}".format(x))
                    printer.send("G1 Y{:.2f}".format(y))
                    printer.send("G1 Z{:.2f}".format(z))
                    print(f"Printer moving to ({x}, {y}, {z})")

                    # Wait for the move to complete (you may adjust the duration based on your printer's speed)
                    time.sleep(5)
                    
                    if x == x_min:
                        time.sleep(15)                 
                    
                    for i in range(100):
                        field1 = sen1.get_magnetometer()
                        magnetic_field_x_sensor1 = field1[0]
                        magnetic_field_y_sensor1 = field1[1]
                        magnetic_field_z_sensor1 = field1[2]

                        field2 = sen2.get_magnetometer()
                        magnetic_field_x_sensor2 = field2[0]
                        magnetic_field_y_sensor2 = field2[1]
                        magnetic_field_z_sensor2 = field2[2]

                        measurements_sensor1.extend([(x, y, z, magnetic_field_x_sensor1, magnetic_field_y_sensor1, magnetic_field_z_sensor1)])
                        measurements_sensor2.extend([(x, y, z, magnetic_field_x_sensor2, magnetic_field_y_sensor2, magnetic_field_z_sensor2)])
                    print("Measurements completed at this location!")

        print("Printing and measurements completed!")

    except Exception as e:
        print("Error: ", e)
    finally:
        if printer:
            printer.disconnect()
            print("Disconnected from the 3D printer.")

        printer.send(f"G0 X{0} Y{0} Z{120}")
        measurements_bg1 = []
        measurements_bg2 = []
        
        for i in range(100):
                        field1 = sen1.get_magnetometer()
                        magnetic_field_x_sensor1 = field1[0]
                        magnetic_field_y_sensor1 = field1[1]
                        magnetic_field_z_sensor1 = field1[2]

                        field2 = sen2.get_magnetometer()
                        magnetic_field_x_sensor2 = field2[0]
                        magnetic_field_y_sensor2 = field2[1]
                        magnetic_field_z_sensor2 = field2[2]

                        measurements_bg1.extend([(magnetic_field_x_sensor1, magnetic_field_y_sensor1, magnetic_field_z_sensor1)])
                        measurements_bg2.extend([(magnetic_field_x_sensor2, magnetic_field_y_sensor2, magnetic_field_z_sensor2)])
                        
        # Convert the measurements data lists to pandas DataFrames for Sensor 1 and Sensor 2
        columns = ['MagneticField_X', 'MagneticField_Y', 'MagneticField_Z']
        df_bg1 = pd.DataFrame(measurements_bg1, columns=columns)
        df_bg2 = pd.DataFrame(measurements_bg2, columns=columns)

        # Calculate the average magnetic field measurements for Sensor 1 and Sensor 2
        bg1 = df_bg1.mean()
        bg2 = df_bg2.mean()
        
        # Input background magnetic field values for Sensor 1 and Sensor 2
        bg_magnetic_field_sensor1_x = bg1[0]
        bg_magnetic_field_sensor1_y = bg1[1]
        bg_magnetic_field_sensor1_z = bg1[2]
        bg_magnetic_field_sensor2_x = bg2[0]
        bg_magnetic_field_sensor2_y = bg2[1]
        bg_magnetic_field_sensor2_z = bg2[2]
        
        print("The background field for sensor 1 is " + str(bg1))
        print("The background field for sensor 2 is " + str(bg2))
        
        # Convert the measurements data lists to pandas DataFrames for Sensor 1 and Sensor 2
        columns = ['X', 'Y', 'Z', 'MagneticField_X', 'MagneticField_Y', 'MagneticField_Z']
        df_sensor1 = pd.DataFrame(measurements_sensor1, columns=columns)
        df_sensor2 = pd.DataFrame(measurements_sensor2, columns=columns)

        # Calculate the average magnetic field measurements for Sensor 1 and Sensor 2 at each location
        avg_magnetic_fields_sensor1 = df_sensor1.groupby(['X', 'Y', 'Z']).mean().reset_index()
        avg_magnetic_fields_sensor2 = df_sensor2.groupby(['X', 'Y', 'Z']).mean().reset_index()

        # Subtract the background magnetic field values for Sensor 1 and Sensor 2 from the average measurements
        avg_magnetic_fields_sensor1['MagneticField_X'] -= bg_magnetic_field_sensor1_x
        avg_magnetic_fields_sensor1['MagneticField_Y'] -= bg_magnetic_field_sensor1_y
        avg_magnetic_fields_sensor1['MagneticField_Z'] -= bg_magnetic_field_sensor1_z

        avg_magnetic_fields_sensor2['MagneticField_X'] -= bg_magnetic_field_sensor2_x
        avg_magnetic_fields_sensor2['MagneticField_Y'] -= bg_magnetic_field_sensor2_y
        avg_magnetic_fields_sensor2['MagneticField_Z'] -= bg_magnetic_field_sensor2_z

        # Save the measurements and average as CSV files for Sensor 1 and Sensor 2 in the user-named folder
        measurements_file_sensor1 = os.path.join(data_directory, 'measurements_sensor1.csv')
        measurements_file_sensor2 = os.path.join(data_directory, 'measurements_sensor2.csv')
        avg_measurements_file_sensor1 = os.path.join(data_directory, 'average_measurements_sensor1.csv')
        avg_measurements_file_sensor2 = os.path.join(data_directory, 'average_measurements_sensor2.csv')

        df_sensor1.to_csv(measurements_file_sensor1, index=False)
        df_sensor2.to_csv(measurements_file_sensor2, index=False)

        avg_magnetic_fields_sensor1.to_csv(avg_measurements_file_sensor1, index=False)
        avg_magnetic_fields_sensor2.to_csv(avg_measurements_file_sensor2, index=False)

        printer.send(f"G0 X{x_max/2} Y{y_max/2} Z{z_min}")
        
        print("\nMeasurement data and average measurements saved to the folder: '{}'".format(user_folder))

if __name__ == "__main__":
    # Replace '/dev/ser/by-id/usb-Prusa_Research__prusa3d.com__Original_Prusa_i3_MK2_CZPX1017X003XC14071-if00' with your actual serial port and set the correct baudrate
    record_measurements('/dev/serial/by-id/usb-Prusa_Research__prusa3d.com__Original_Prusa_i3_MK2_CZPX1017X003XC14071-if00', 115200)
    
    # Push to the Git repository
    git_push("/home/raspberrypi/Documents/magnetic-sensor-data", "main")
