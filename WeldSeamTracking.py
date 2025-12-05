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
import matplotlib.pyplot as plt

class LaserScanGenerator:
    """
    Simple simulator for laser scans.

    - Each scan has `scan_length` values.
    - First seam position can be anywhere.
    - For all *subsequent* scans, the seam index is forced to be within
      `max_offset_from_centre` indices of the torch centre index.
    """

    def __init__(
        self,
        scan_length: int = 61,
        base_distance: float = 10.0,
        seam_depth: float = 3.0,
        noise_std: float = 0.3,
        max_offset_from_centre: int = 3,
    ):
        self.scan_length = scan_length
        self.base_distance = base_distance
        self.seam_depth = seam_depth
        self.noise_std = noise_std
        self.max_offset_from_centre = max_offset_from_centre

        self._first_scan = True
        self.current_seam_index = None

    def _choose_initial_seam_index(self):
        # Seam can start anywhere for the first scan
        self.current_seam_index = np.random.randint(0, self.scan_length)

    def _choose_followup_seam_index(self, centre_index: int):
        # For subsequent scans, constrain seam index near the centre
        low = max(0, centre_index - self.max_offset_from_centre)
        high = min(self.scan_length - 1, centre_index + self.max_offset_from_centre)
        self.current_seam_index = np.random.randint(low, high + 1)

    def generate_scan(self, centre_index: int):
        """
        Generate a single scan given the current torch centre index.
        Returns (scan, true_seam_index).
        """
        if self._first_scan:
            self._first_scan = False
            self._choose_initial_seam_index()
        else:
            self._choose_followup_seam_index(centre_index)

        # Start with roughly flat surface + noise
        scan = self.base_distance + np.random.normal(
            0.0, self.noise_std, size=self.scan_length
        )

        # Carve a "valley" at the seam position
        scan[self.current_seam_index] -= self.seam_depth

        return scan, self.current_seam_index

class LaserSensor:
    def __init__(self, scan_length = 61):
        self.scan_length = scan_length
        self.sensor_centre_index = (scan_length - 1)//2
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

        #print(self.cleaned_data)

        return self.cleaned_data
    
    def find_feature(self, scan_data):
        # clean the data
        self.cleaned_data = self.clean_data(scan_data)

        # for this example, find min value as the feature to be tracked
        self.feature_y = np.min(self.cleaned_data)
        self.feature_x = np.argmin(self.cleaned_data)

        return self.feature_x, self.feature_y


class WeldingController:
    def __init__(self, sensor: LaserSensor, max_offset_indices: int = 4):
        self.sensor = sensor
        self.max_offset_indices = max_offset_indices
        self.torch_offset = 0.0 # offset of the torch from the seam feature (perpendicular distance w.r.t weld path)
        self.torch_position = 0.0 #position in robot "world"
        self.feature_x = None
        self.feature_y = None
        self.welding = True

        self.path=[] #path as list of (scan_index, torch_pos)

    def calc_torch_offset(self):
        # Calculate the offset of the torch from the seam feature
        index_offset = self.feature_x - self.sensor.sensor_centre_index
        self.torch_offset = index_offset
        return index_offset

    def process_scan(self, scan_data, scan_index):
        # Find seam feature
        self.feature_x, self.feature_y = self.sensor.find_feature(scan_data)
        print(f"Feature found at X: {self.feature_x}, Y: {self.feature_y:.3f}")


        offset_indices = self.calc_torch_offset()
        print(
            f"Torch scan centre index = {self.sensor.sensor_centre_index}, "
            f"offset = {offset_indices} indices"
        )

        #Stopping Criterion:
        if abs(offset_indices) <= self.max_offset_indices:
            self.torch_position += offset_indices
            print(
                f"Torch moves by {offset_indices} index units. "
                f"New torch_position (index units) = {self.torch_position}"
            )
        else:
            self.stop_recentre()

        #update stored path
        self.path.append((scan_index, self.torch_position))


    def stop_recentre(self):
        #welding stops if seam too far from torch centre
        self.welding = False

        print(f"Welding stopped - seam {self.feature_x} is more than {self.max_offset_indices} indices from centre {self.sensor.sensor_centre_index}")

        # Reposition: move torch centre index to seam index
        offset_indices = self.torch_offset
        self.torch_position += offset_indices

        print(
            f"Robot repositions by {offset_indices} units.\n"
            f"New torch position = {self.torch_position}"
        )

        # After repositioning, welding resumed
        self.welding = True
        print("Welding Resumed")

    def __repr__(self):
        return (
            f"Torch offset (indices): {self.torch_offset}, "
            f"Torch position (indices): {self.torch_position}, "
            f"Welding: {self.welding}"
        )
    

class PathPlotter:
    def __init__(self, xlim=(-40, 40), ylim=(0, 20)):
        plt.ion()
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_xlabel("Torch position (index units)")
        self.ax.set_ylabel("Scan number")
        self.ax.set_title("Robot Welding Path")

        # scatter plot initially empty
        self.scatter = self.ax.scatter([], [])
        self.line, = self.ax.plot([], [], '-o', color='red', linewidth=1.5)

    def update(self, path):
        """
        Update the plot with the current path.
        path must be a list of (scan_index, torch_position)
        """
        if len(path) == 0:
            return
        
        ys, xs = zip(*path)  # unpack scan numbers & torch positions

        #update torch pos scatter points
        self.scatter.set_offsets(np.c_[xs, ys])

        #updating welding line
        self.line.set_data(xs, ys)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_blocking(self):
        plt.ioff()
        plt.show()
    
if __name__ == "__main__":
    # Create sensor and generator
    sensor = LaserSensor(scan_length=61)
    generator = LaserScanGenerator(scan_length=sensor.scan_length)
    robot = WeldingController(sensor)

    n_scans = 20
    plotter = PathPlotter(xlim=(-40,40), ylim=(0,n_scans))

    # Simulate Scans
    for i in range(n_scans):
        scan, seam_idx = generator.generate_scan(sensor.sensor_centre_index)
        print(f"\nScan {i + 1}, true seam index (sim) = {seam_idx}")

        robot.process_scan(scan, scan_index = i)
        print(robot)

        #update path plot
        plotter.update(robot.path)

        time.sleep(0.5)

    plotter.show_blocking()