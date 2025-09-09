# Program that tracks just the movement of a surface - no training data required

import cv2
from ids_peak import ids_peak
import ids_peak_ipl.ids_peak_ipl as ids_ipl
import ids_peak.ids_peak_ipl_extension as ids_ipl_extension
import numpy as np
import pandas as pd
import pickle
from scipy.signal import find_peaks
import os
import re
from collections import deque
import warnings

'''
Important Info:
Camera FOV = ...mm (width) x ...mm (height)
Radius of pipe = ... => circumference (surface area) = ...

51 pixels per degree of rotation

'''




def ids_init(exposure=2000):
    '''Initialise IDS camera and start acquisition.'''

    # Initialise IDS peak library
    ids_peak.Library.Initialize()

    # Open device manager - first available camera
    device_manager = ids_peak.DeviceManager.Instance()
    device_manager.Update()
    device_descriptors = device_manager.Devices()

    print("Found Devices: " + str(len(device_descriptors)))
    for device_descriptor in device_descriptors:
        print(device_descriptor.DisplayName())

    if not device_descriptors:
        raise RuntimeError("No camera found")

    device = device_descriptors[0].OpenDevice(ids_peak.DeviceAccessType_Control)
    print("Opened Device: " + device.DisplayName())
    remote_device_nodemap = device.RemoteDevice().NodeMaps()[0]

    # Disable trigger mode for continuous acquisition - live mode
    try:
        remote_device_nodemap.FindNode("TriggerMode").SetCurrentEntry("Off")
    except:
        print("Failed to disable trigger mode, continuing with default")

    # Open data stream
    datastream = device.DataStreams()[0].OpenDataStream()
    payload_size = remote_device_nodemap.FindNode("PayloadSize").Value()

    # Allocate and announce buffers
    for i in range(datastream.NumBuffersAnnouncedMinRequired()):
        buffer = datastream.AllocAndAnnounceBuffer(payload_size)
        datastream.QueueBuffer(buffer)

    # Start acquisition
    datastream.StartAcquisition()
    remote_device_nodemap.FindNode("AcquisitionStart").Execute()

    # Set Exposure
    remote_device_nodemap.FindNode("ExposureTime").SetValue(exposure) # in microseconds

    return device, remote_device_nodemap, datastream


def ids_get_frame(datastream):
    '''Retrieve live frame'''

    buffer = datastream.WaitForFinishedBuffer(1000)  # 1s timeout
    if buffer is None:
        print("Failed to acquire buffer")
        #break
        return True

    # Convert buffer to image                                                                                                          
    raw_image = ids_ipl_extension.BufferToImage(buffer)

    # Convert to BGR
    colour_image = raw_image.ConvertTo(ids_ipl.PixelFormatName_BGR8)

    # Get (3D for colour) numpy array (for use with OpenCV)
    frame = colour_image.get_numpy_3D()

    # Queue buffer back to data stream
    datastream.QueueBuffer(buffer)

    # Create copy of frame for use with analysis (not linked to datastream and buffer)
    frame = frame.copy()

    return frame

def ids_close(device, remote_device_nodemap, datastream):
    if remote_device_nodemap:
        try:
            remote_device_nodemap.FindNode("AcquisitionStop").Execute()
            print("Stopped Acquisition.")
        except:
            pass
    if datastream:
        try:
            datastream.StopAcquisition()
            print("Stopped datastream.")
        except:
            pass
    ids_peak.Library.Close()
    cv2.destroyAllWindows()

def distance_converted(pixel_distance, pixels_per_unit_distance=10):
    # for pixels_per_unit_distance pixels per mm 
    converted_distance = pixel_distance/ pixels_per_unit_distance

    return np.round(converted_distance, decimals=1)


def motion_shift(prev_frame, curr_frame, frame_height):
    feature_tracker = cv2.ORB_create(nfeatures=30)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    crop_amount = 0.5

    top_crop = int((crop_amount/2)*frame_height)
    bottom_crop = int((1-(crop_amount/2))*frame_height)

    prev_frame_crop = prev_frame[top_crop:bottom_crop,:]
    curr_frame_crop = curr_frame[top_crop:bottom_crop,:]

    #frame_crop = cv2.equalizeHist(frame_crop)

    prev_kp, prev_des = feature_tracker.detectAndCompute(prev_frame_crop, None)
    #print(f"des_live dtype: {des_live.dtype}, shape: {des_live.shape}")
    curr_kp, curr_des = feature_tracker.detectAndCompute(curr_frame_crop, None)

    matches = matcher.knnMatch(curr_des, prev_des, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.65 * n.distance]
    #print(len(good_matches))

    ver_shifts = []
    hor_shifts = []
    for match in good_matches[:3]:  # Use top 3 matches
        idx_curr = match.queryIdx
        idx_prev = match.trainIdx

        #print(idx_live, idx_ref)

        pt_curr = curr_kp[idx_curr].pt  # (x, y)
        pt_prev = prev_kp[idx_prev].pt       # already a (x, y) tuple

        x_shift = pt_prev[0] - pt_curr[0]
        y_shift = pt_prev[1] - pt_curr[1]  # vertical pixel displacement

        #print(x_shift, y_shift)

        hor_shifts.append(x_shift)
        ver_shifts.append(y_shift)


    #print("Displacements: ", shifts)
    #vertical_shifts = shifts[:, 1]
    #print("Vertical Displacements: ", vertical_shifts)

    # vertical shift is how far above (in pixels) the current frame is from the angle found
    # so if vertical shift is negative, then the angle found is below current frame

    horizontal_shift = np.median(hor_shifts) if hor_shifts else 0
    vertical_shift = np.median(ver_shifts) if ver_shifts else 0

    if horizontal_shift == 0:
        if vertical_shift >= 0:
            motion_angle = np.deg2rad(90)
        else: 
            motion_angle = np.deg2rad(-90)
    else:
        motion_angle = np.arctan(vertical_shift/horizontal_shift) #angle of motion from x axis (pointing directly right)

    if np.isnan(motion_angle):
        motion_angle = 0

    distance = np.sqrt((horizontal_shift**2) + (vertical_shift**2))

    distance_delta = distance if (vertical_shift >= 0) else (-1*distance)

    

    return distance_delta, motion_angle

def draw_motion_angle(motion_angle, frame):
    #print(motion_angle)
    #arrow params
    length = 100
    centre = (200,200)

    end_x = int(centre[0] + length*np.cos(motion_angle))
    end_y = int(centre[1] - length*np.sin(motion_angle))
    end_point = (end_x, end_y)

    cv2.arrowedLine(frame, centre, end_point, color=(0,255,0), thickness=2, tipLength=0.2)

    return frame


def live_tracking():
    #Setup ORB and matcher
    feature_tracker = cv2.ORB_create(nfeatures=30)
    #feature_tracker = cv2.AKAZE_create()

    #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # for bfmatch
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # for bf.knnMatch

    #Initialise camera
    device, remote_device_nodemap, datastream = ids_init()

    # get first n frames while camera adjusts
    for i in range(10):
        frame = ids_get_frame(datastream)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    distance = 0.0

    motion_angles = deque(maxlen=3)

    while True:
        frame = ids_get_frame(datastream)
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        delta, motion_angle = motion_shift(prev_frame, curr_frame, frame_height)

        distance += delta

        motion_angles.append(motion_angle)

        motion_angle = np.median(np.array(motion_angles))

        converted_distance = distance_converted(distance)

        prev_frame = curr_frame

        # central line for reference
        cv2.line(frame, (0, frame_height//2), (frame_width, frame_height//2), color=(0,255,0), thickness=1)

        # Display Results
        #Motion Angle
        frame = draw_motion_angle(motion_angle, frame)
        #Rotation
        cv2.putText(frame, f"Distance: {converted_distance}mm", (frame_width - 350, (frame_height//2)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        #show frame
        cv2.imshow("Analysis", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    ids_close(device, remote_device_nodemap, datastream)

    return


if __name__ == '__main__':
    live_tracking()