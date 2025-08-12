# Robotics
Data Science programs for various robotics applications, notably applications in automated robotic welding.

WeldSeamTracking.py - Simple python implementation for weld seam tracking using a laser distance sensor for seam searching.

Calibration.py - Program for calibrating a laser distance sensor with a (FANUC) robot Torch Centre Point: finds the translation from an (x,y) sensor coordinate to an (x,y,z,w,p,r) robot coordinate.

CNN_Feature_Extraction.py - Pytorch framework for geometric feature extraction from 1 dimensional shape data (originally for laser distance values to find wleding joint features). Utilises 1D convolution layer and model training via training and validation sets. Includes model evaluation via error calculation on a testing set as well as graphically via plotting predictions using matplotlib.

Track_Flow.py - Method of tracking the movement distance, direction and speed of any surface (including essentially featureless surfaces) using OpenCV and ORB feature detection.

Data_PreProcessing.py - Research into methods of cleaning noisy signal data with ignificant numbers of outliers - originally developed for research into laser distance data for particularly reflective welding surfaces.

V_Groove_Analysis.py - geometric feature analysis of laser distance profiles of various welding joints. Uses data cleaning and (RANSAC) regression methods for accurately finding edges and corners within noisy signal data with significant numbers of outliers.

