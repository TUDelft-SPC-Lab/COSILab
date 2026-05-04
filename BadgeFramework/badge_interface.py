from badge import *
from ble_badge_connection import *

class BadgeInterface():
    def __init__(self, address):
        self.address = address
        self.connection = None
        self.badge = None

    def connect(self):
        self.connection = BLEBadgeConnection.get_connection_to_badge(self.address)
        self.connection.connect()
        self.badge = OpenBadge(self.connection)
        print("Connected!")

    def get_status(self):
        return self.badge.get_status()
    
    def get_free_sdc_space(self):
        return self.badge.get_free_sdc_space()
    
    def start_imu(self):
        return self.badge.start_imu()
    
    def start_microphone(self, mode):
        #print("MODE interface ",mode)
        return self.badge.start_microphone(mode=mode)
    
    def stop_microphone(self):
        return self.badge.stop_microphone()
    
    def stop_imu(self):
        return self.badge.stop_imu()
    
    def start_scan(self):
        return self.badge.start_scan()
    
    def stop_scan(self):
        return self.badge.stop_scan()
    
    def get_imu_data(self):
        return self.badge.get_imu_data()
    
    def sdc_errase_all(self):
        return self.badge.sdc_errase_all()
    
    def get_fw_version(self):
        return self.badge.get_fw_version()
    