# Program that tracks just the movement of a surface - no training data required

import cv2
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

def distance_converted(pixel_distance, pixels_per_unit_distance=10):
    # for pixels_per_unit_distance pixels per mm 
    converted_distance = pixel_distance/ pixels_per_unit_distance

    return np.round(converted_distance, decimals=1)


def motion_shift(matcher, prev_kp, prev_des, curr_kp, curr_des):

    matches = matcher.knnMatch(curr_des, prev_des, k=2)

    matches = matcher.knnMatch(curr_des, prev_des, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

    num_matches = len(good_matches)
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

    

    return distance_delta, motion_angle, num_matches

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


def tracking(live = True):
    if live:
        video_source = 0 #use default capture device (webcam)
    else:
        video_source = "Sample_Data/"

    #Setup ORB and matcher
    feature_tracker = cv2.ORB_create(nfeatures=30)
    #feature_tracker = cv2.AKAZE_create()

    #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # for bfmatch
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # for bf.knnMatch


    #Initialise video source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame = cap.read()

    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    distance = 0.0

    motion_angles = deque(maxlen=3)

    kp_prev, des_prev = feature_tracker.detectAndCompute(prev_frame, None)


    while cap.isOpened():
        ret, frame = cap.read()
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp_curr, des_curr = feature_tracker.detectAndCompute(curr_frame, None)

        delta, motion_angle, num_matches = motion_shift(matcher, kp_prev, des_prev, kp_curr, des_curr)

        distance += delta

        motion_angles.append(motion_angle)

        motion_angle = np.median(np.array(motion_angles))

        converted_distance = distance_converted(distance)

        kp_prev = kp_curr
        des_prev = des_curr

        # central line for reference
        cv2.line(frame, (0, frame_height//2), (frame_width, frame_height//2), color=(0,255,0), thickness=1)

        # Display Results
        #Motion Angle
        frame = draw_motion_angle(motion_angle, frame)
        #Rotation
        cv2.putText(frame, f"Distance: {converted_distance}mm", (frame_width - 350, (frame_height//2)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        #num matches
        cv2.putText(frame, f"Matches: {num_matches}", (frame_width - 350, (frame_height//2)-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        #show frame
        cv2.imshow("Analysis", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    tracking(live = True)