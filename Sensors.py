import lsm303d
import smbus
import adafruit_mlx90393
import board


class PIMSensor:
    def __init__(self, address=0x1D, i2c_dev=1):
        self.sensor = lsm303d.LSM303D(address, i2c_dev=smbus.SMBus(i2c_dev))

    def get_magnetometer(self):  # return tuple
        x, y, z = self.sensor.magnetometer()
        return x / 10, y / 10, z / 10


class MLXSensor:
    def __init__(
        self,
        address,
        gain=adafruit_mlx90393.GAIN_1X,
        resolution=adafruit_mlx90393.RESOLUTION_16,
        filt=adafruit_mlx90393.FILTER_7,
        oversampling=adafruit_mlx90393.OSR_3,
    ):
        i2c = board.I2C()
        self.sensor = adafruit_mlx90393.MLX90393(
            i2c,
            address=address,
            gain=gain,
            resolution=resolution,
            filt=1,
            oversampling=1,
        )

    def get_magnetometer(self):
        x, y, z = self.sensor.magnetic
        return x/1000, y/1000, z/1000
