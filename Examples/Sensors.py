import TLV
import lsm303d
class TLVSensor:
    def __init__(self):
        self.sensor = TLV.TLV493D()
    
    def get_magnetometer(self): #return tuple
        self.sensor.update_data() 
        x = self.sensor.get_x()
        y = self.sensor.get_y()
        z = self.sensor.get_z()
        return x, y, z

class PIMSensor:
    def __init__(self, address=0x1d):
        self.sensor = lsm303d.LSM303D(address)
    
    def get_magnetometer(self): #return tuple
        x, y, z = self.sensor.magnetometer()
        return x, y, z
