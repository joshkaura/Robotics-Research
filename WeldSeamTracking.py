''' Introduction: 
The robot uses a laser distance sensor to detect the seam position relative to its nominal path. 
The goal is to keep the welding torch centered on the seam while welding. The laser sensor finds 
the seam position in a plane perpendicular to the welding path.

This is a simple implementation/ framework of a seam tracking system using a laser sensor - taking
the minimum of an scan (array) of laser distance values as the location of the seam. 
Other features that can be be built in include more sophisticated data cleaning and feature finding 
(e.g. see V Groove analysis); joint type and welding parameter selection
'''

import numpy as np
import time


class LaserSensor:
    def __init__(self):
        #self.distance_data = np.array(data)
        self.feature_x = 0 #x index of the feature in the scan data (perpendicular to the welding path)
        self.feature_y = 0 #y value (distance away from the sensor) of the feature - not essential if welding piece assumed flat
    
    def clean_data(self, scan_data):

        self.cleaned_data = np.array(scan_data.copy(), dtype=np.float64)

        # replace 0 and None values with NaN
        zeros = np.where(self.cleaned_data == 0.0)[0]
        nones = np.where(self.cleaned_data == None)[0]
        self.cleaned_data[zeros] = np.nan
        self.cleaned_data[nones] = np.nan

        #interpolate missing values
        nans = np.isnan(self.cleaned_data)
        not_nans = ~nans
        if np.any(not_nans):
            indices = np.arange(len(self.cleaned_data))
            self.cleaned_data[nans] = np.interp(indices[nans], indices[not_nans], self.cleaned_data[not_nans])

        print(self.cleaned_data)

        return self.cleaned_data
    
    def find_feature(self, scan_data):
        # clean the data
        self.cleaned_data = self.clean_data(scan_data)

        # for this example, find min value as the feature to be tracked
        self.feature_y = np.min(self.cleaned_data)
        self.feature_x = np.argmin(self.cleaned_data)

        return self.feature_x, self.feature_y


class WeldingController:
    def __init__(self):
        self.torch_offset = 0.0 # offset of the torch from the seam feature (perpendicular distance w.r.t weld path)
        self.torch_centre_index = 1 # index of torch centre in the scan data

    def calc_torch_offset(self):
        # Calculate the offset of the torch from the seam feature
        self.torch_offset = self.feature_x - self.torch_centre_index
        return self.torch_offset

    def process_scan(self, scan_data):
        self.feature_x, self.feature_y = LaserSensor().find_feature(scan_data)
        print(f"Featured found at X: {self.feature_x}, Y: {self.feature_y}")

        self.torch_offset = self.calc_torch_offset()
        
    def __repr__(self):
        return f"Torch offset: {self.torch_offset}"
    
if __name__ == "__main__":
    robot = WeldingController()
    scan_data = [[5,4,5],[3,3,2],[3,2,3],[1,2,0],[6,6,5],[2,1,2]]
    weld_start_pos = [0,0]
    weld_end_pos = [len(scan_data) - 1, 0]


    for scan in scan_data:
        robot.process_scan(scan)
        print(robot)
        time.sleep(0.5)