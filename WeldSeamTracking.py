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

    - Each scan has fixed scan length.
    - First seam position can be anywhere.
    - For subsequent scans, the seam index is forced to be within
      a max offset from the torch centre index.
    """

    def __init__(
        self,
        scan_length: int = 101,
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

    def _choose_next_seam_index(self, last_seam_pos: int):
        # For subsequent scans, constrain seam index near the centre
        low = max(0, last_seam_pos - self.max_offset_from_centre)
        high = min(self.scan_length - 1, last_seam_pos + self.max_offset_from_centre)
        self.current_seam_index = np.random.randint(low, high + 1)

    def generate_scan(self, last_pos: int):
        """
        Generate a scan given the torch centre is above the centre of the laser FOV .
        Returns (scan, true_seam_index).
        """
        if self._first_scan:
            self._first_scan = False
            self._choose_initial_seam_index()
        else:
            self._choose_next_seam_index(last_pos)

        # Start with roughly flat surface + noise
        scan = self.base_distance + np.random.normal(
            0.0, self.noise_std, size=self.scan_length
        )

        # Carve a "valley" at the seam position
        scan[self.current_seam_index] -= self.seam_depth

        return scan, self.current_seam_index

class LaserSensor:
    def __init__(self, scan_length = 101, fov_half_width_mm = 50.0):
        self.scan_length = scan_length
        self.sensor_centre_index = (scan_length - 1)//2
        self.feature_x = 0 #x index of the feature in the scan data (perpendicular to the welding path)
        self.feature_y = 0 #y value (distance away from the sensor) of the feature - not essential if welding piece assumed flat

        #fov_half_width_mm: half of FOV in mm (e.g. 20mm each side => [-20, +20])
        self.fov_half_width_mm = fov_half_width_mm

        self.x_profile_mm = np.linspace(-(self.fov_half_width_mm), fov_half_width_mm, scan_length)

        self.feature_index = 0 #x index of the feature in the scan data (perpendicular to the welding path)
        self.feature_x_mm = 0.0
        self.feature_y = 0.0 #y value (distance away from the sensor) of the feature

        

    
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
        self.feature_index = int(np.argmin(self.cleaned_data))
        self.feature_x_mm = float(self.x_profile_mm[self.feature_index])


        return self.feature_index, self.feature_x_mm, self.feature_y


class WeldingController:
    def __init__(self, sensor: LaserSensor, max_offset_mm: float = 4.0):
        self.sensor = sensor
        self.max_offset_mm = max_offset_mm
        self.torch_offset = 0.0 # offset of the torch from the seam feature (perpendicular distance w.r.t weld path)
        self.torch_position = 0.0 #position in robot "world"

        self.feature_index = None
        self.feature_x_mm = None
        self.feature_y = None
        self.welding = True

        self.path=[] #path as list of (scan_index, torch_pos(mm))

    def calc_torch_offset(self):
        # Calculate the offset of the torch from the seam feature - assuming torch at centre of laser FOV (0mm)
        offset_mm = self.feature_x_mm # seam pos mm - 0
        self.torch_offset = offset_mm
        return offset_mm

    def process_scan(self, scan_data, scan_index):
        # Find seam feature
        self.feature_index, self.feature_x_mm, self.feature_y = self.sensor.find_feature(scan_data)
        print(f"Feature found at X (mm): {self.feature_x_mm:.2f}, Y: {self.feature_y:.3f}")


        offset_mm = self.calc_torch_offset()
        print(
            f"Torch at X=0 of sensor FOV"
            f"Torch offset required = {offset_mm} indices"
        )

        #Stopping Criterion:
        if abs(offset_mm) <= self.max_offset_mm:
            self.torch_position += offset_mm
            print(
                f"Torch moves by {offset_mm} mm "
                f"New torch X position (mm) = {self.torch_position}"
            )
        else:
            self.stop_recentre()

        #update stored path
        self.path.append((scan_index, self.torch_position))


    def stop_recentre(self):
        #welding stops if seam too far from torch centre
        self.welding = False

        print(
            f"***WELDING STOPPED***"
            f"Seam at {self.feature_x_mm}mm is more than {self.max_offset_mm} mm from current torch position"
        )

        # Reposition: move torch to seam pos
        offset_mm = self.torch_offset
        self.torch_position += offset_mm

        time.sleep(1)

        print(
            f"Robot repositions by {offset_mm} units.\n"
            f"New torch position = {self.torch_position}"
        )

        # After repositioning, welding resumed
        self.welding = True
        print("***WELDING RESUMED***")

    def __repr__(self):
        return (
            f"Torch offset (indices): {self.torch_offset:.3f}, "
            f"Torch position (indices): {self.torch_position:.2f}, "
            f"Welding: {self.welding}"
        )
    

class PathPlotter:
    def __init__(self, xlim=(-40, 40), ylim=(0, 20)):
        plt.ion()
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_xlabel("Torch position (mm)")
        self.ax.set_ylabel("Scan number")
        self.ax.set_title("Robot Welding Path")

        # scatter plot initially empty
        self.scatter = self.ax.scatter([], [], s=1)
        self.line, = self.ax.plot([], [], '-', color='red', linewidth=0.5)

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

    def save(self, filepath="Results/WeldSeamTracking.png", dpi=300):
        self.fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        print(f"Saved final plot to: {filepath}")

    def show_blocking(self):
        plt.ioff()
        plt.show()
    
    
if __name__ == "__main__":
    #choose scan_length (no. lds points in scan)
    scan_length = 101
    #choose laser FOV
    fov_half_width_mm = 20.0
    #choose maximum seam position offset (mm) to continue welding continuously
    max_offset_mm = 5.0

    # Create sensor and scan generator
    sensor = LaserSensor(scan_length=scan_length, fov_half_width_mm=fov_half_width_mm)
    generator = LaserScanGenerator(scan_length=sensor.scan_length)

    # initialise welding robot
    robot = WeldingController(sensor, max_offset_mm=max_offset_mm)

    n_scans = 100
    plotter = PathPlotter(xlim=(-(fov_half_width_mm + 20),fov_half_width_mm+20), ylim=(0,n_scans))

    # Simulate Scans
    for i in range(n_scans):
        scan, seam_idx = generator.generate_scan(sensor.sensor_centre_index)
        print(f"\nScan {i + 1}, true seam index (sim) = {seam_idx}")

        robot.process_scan(scan, scan_index = i)
        print(robot)

        #update path plot
        plotter.update(robot.path)

        time.sleep(0.05)

    plotter.save()
    plotter.show_blocking()
    plt.close()

