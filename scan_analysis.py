import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore, linregress
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor, LinearRegression
import time

filepaths = [
    "Sample_Data/lds_scan1.csv",
]

def load_data(filepath, flip=True):
    data = pd.read_csv(filepath)
    data = data.replace(0, np.nan)
    if flip:
        data = -data
    return data

def remove_outliers_zscore(idx, profile, threshold=2.2):
    z_scores = zscore(profile, nan_policy='omit')
    outliers = np.abs(z_scores) > threshold
    #print(f"Row {idx} — Outliers: {np.sum(outliers)}")
    profile[outliers] = np.nan
    return profile

def detect_slope_spikes(idx, profile, threshold=3):
    diff = np.abs(np.diff(profile, prepend=profile[0]))
    spikes = diff > threshold
    #print(f"Row {idx} — Spikes: {np.sum(spikes)}")
    profile[spikes] = np.nan
    return profile

def interpolate_nan(profile):
    nans = np.isnan(profile)
    not_nans = ~nans
    if np.any(not_nans):
        indices = np.arange(len(profile))
        profile[nans] = np.interp(indices[nans], indices[not_nans], profile[not_nans])
    return profile

def smooth_savgol(profile, window_length=51, polyorder=2):
    window_length = min(window_length, len(profile) // 2 * 2 - 1)
    return savgol_filter(profile, window_length, polyorder)

def extract_features(profile):
    gradient = np.gradient(profile)
    left = np.argmax(gradient[:len(gradient)//2] < -0.2)
    right = len(gradient) - 1 - np.argmax(gradient[::-1][:len(gradient)//2] > 0.2) #scanning gradient backwards
    bottom = np.argmin(profile)
    #print(left, bottom, right)
    return left, bottom, right

def fit_line_segment(x, y):
    slope, intercept, _, _, _ = linregress(x, y)
    return slope, intercept

def extract_lines(profile):
    x = np.arange(len(profile))
    smooth = smooth_savgol(profile)
    left_end, bottom, right_start = extract_features(smooth)

    segments = {
        'left_top': (0, left_end),
        'left_slope': (left_end, bottom),
        'right_slope': (bottom, right_start),
        'right_top': (right_start, len(profile))
    }

    line_params = {}
    for name, (start, end) in segments.items():
        x_seg = x[start:end]
        y_seg = profile[start:end]
        if len(x_seg) >= 2:
            m, b = fit_line_segment(x_seg, y_seg)
            line_params[name] = {'start': start, 'end': end, 'slope': m, 'intercept': b}

    return line_params, [left_end, bottom, right_start]

def remove_internal_reflections(profile, slope_margin=20):
    x = np.arange(len(profile))
    #y = profile.copy()

    # Smooth to reduce noise for slope detection
    y_smooth = smooth_savgol(profile)

    # Compute gradient to find inflection points
    gradient = np.gradient(y_smooth)

    # Identify key points: left slope end, bottom, right slope start
    left_top_end = np.argmax(gradient[:len(gradient)//2] < -0.2)
    print(f"Left top end: {left_top_end}")

    right_top_start = len(gradient) - np.argmax(gradient[::-1][:len(gradient)//2] > 0.2) #scanning gradient backwards
    print(f"Right top start: {right_top_start}")

    # V bottom = minimum point between the slopes
    bottom_index = np.argmin(y_smooth[left_top_end:right_top_start]) + left_top_end
    print(f"Bottom: {bottom_index}")

    # Fit lines to the left and right slopes
    x_left = x[left_top_end:bottom_index]
    y_left = y_smooth[left_top_end:bottom_index]
    x_right = x[bottom_index:right_top_start]
    y_right = y_smooth[bottom_index:right_top_start]

    # Linear fit to left and right slopes
    left_fit = np.polyfit(x_left, y_left, 1)
    right_fit = np.polyfit(x_right, y_right, 1)

    # Expected values along each slope
    expected_left = np.polyval(left_fit, x_left)
    expected_right = np.polyval(right_fit, x_right)

    # Mark points significantly *above* the slope line as reflection
    mask = np.full_like(profile, False, dtype=bool)
    margin = slope_margin

    mask[left_top_end:bottom_index] = profile[left_top_end:bottom_index] < (expected_left - margin)
    mask[bottom_index:right_top_start] = profile[bottom_index:right_top_start] < (expected_right - margin)

    print(f"Outside wall boundary: {len(mask)}")

    cleaned = profile
    cleaned[mask] = np.nan

    return cleaned, mask


def ransac_line_fit(x, y):
    """Fits a line using RANSAC and returns slope, intercept, inlier mask."""
    x_reshaped = x.reshape(-1, 1)
    model = RANSACRegressor(LinearRegression(), residual_threshold=0.5, max_trials=100)
    model.fit(x_reshaped, y)
    slope = model.estimator_.coef_[0]
    intercept = model.estimator_.intercept_
    return slope, intercept, model.inlier_mask_

def calculate_groove_lines(profile, left, right):
    x = np.arange(len(profile))

    fit_distance = (right - left)//2
    
    # Get slope regions
    x_left = x[left:left + fit_distance]
    y_left = profile[left:left + fit_distance]
    x_right = x[right - fit_distance:right]
    y_right = profile[right - fit_distance:right]

    # Interpolate any NaNs
    y_left_clean = interpolate_nan(y_left.copy())
    y_right_clean = interpolate_nan(y_right.copy())

    # Standard linear fit
    coeffs_left = np.polyfit(x_left, y_left_clean, deg=1)
    coeffs_right = np.polyfit(x_right, y_right_clean, deg=1)

    m1, c1 = coeffs_left
    m2, c2 = coeffs_right

    x_intersect = (c2 - c1) / (m1 - m2) if m1 != m2 else None
    y_intersect = m1 * x_intersect + c1 if x_intersect is not None else None

    # RANSAC fit
    slope_left, intercept_left, inliers_left = ransac_line_fit(x_left.copy(), y_left_clean.copy())
    slope_right, intercept_right, inliers_right = ransac_line_fit(x_right.copy(), y_right_clean.copy())

    x_ransac_intersect = (intercept_right - intercept_left) / (slope_left - slope_right) if slope_left != slope_right else None
    y_ransac_intersect = slope_left * x_ransac_intersect + intercept_left if x_ransac_intersect is not None else None

    return {
        "x": x,
        "x_left": x_left, "y_left": y_left, "y_left_clean": y_left_clean,
        "x_right": x_right, "y_right": y_right, "y_right_clean": y_right_clean,
        "polyfit_left": coeffs_left, "polyfit_right": coeffs_right,
        "ransac_left": (slope_left, intercept_left, inliers_left),
        "ransac_right": (slope_right, intercept_right, inliers_right),
        "intersection_polyfit": (int(x_intersect), y_intersect),
        "intersection_ransac": (int(x_ransac_intersect), y_ransac_intersect)
    }

def plot_groove_lines(result_dict):
    fig, axs = plt.subplots(2, 2, figsize=(18, 5))
    for ax in axs[:, 0]:
        ax.set_xlim(0, 2000)
        ax.set_ylim(-900, -450)
    for ax in axs[:, 1]:
        ax.set_xlim(0, 2000)
        ax.set_ylim(-900, -450)

    x_left = result_dict["x_left"]
    y_left = result_dict["y_left"]
    y_left_clean = result_dict["y_left_clean"]
    x_right = result_dict["x_right"]
    y_right = result_dict["y_right"]
    y_right_clean = result_dict["y_right_clean"]

    # --- Top row: polyfit ---
    axs[0, 0].scatter(x_left, y_left, s=2)
    axs[0, 1].scatter(x_right, y_right, s=2)

    m1, c1 = result_dict["polyfit_left"]
    m2, c2 = result_dict["polyfit_right"]

    axs[0, 0].plot(x_left, m1 * x_left + c1, 'b-', label='Polyfit Left')
    axs[0, 1].plot(x_right, m2 * x_right + c2, 'b-', label='Polyfit Right')

    x_int, y_int = result_dict["intersection_polyfit"]
    if x_int is not None:
        axs[0, 0].scatter([x_int], [y_int], color='red', label='Bottom (polyfit)')
        axs[0, 1].scatter([x_int], [y_int], color='red')

    # --- Bottom row: RANSAC ---
    slope_l, intercept_l, inliers_l = result_dict["ransac_left"]
    slope_r, intercept_r, inliers_r = result_dict["ransac_right"]

    axs[1, 0].scatter(x_left, y_left_clean, s=2, label="Left raw")
    axs[1, 0].scatter(x_left[inliers_l], y_left_clean[inliers_l], color='green', s=4, label="RANSAC inliers")
    axs[1, 0].plot(x_left, slope_l * x_left + intercept_l, 'g--', label='RANSAC fit')

    axs[1, 1].scatter(x_right, y_right_clean, s=2, label="Right raw")
    axs[1, 1].scatter(x_right[inliers_r], y_right_clean[inliers_r], color='orange', s=4, label="RANSAC inliers")
    axs[1, 1].plot(x_right, slope_r * x_right + intercept_r, 'orange', linestyle='--', label='RANSAC fit')

    x_int_r, y_int_r = result_dict["intersection_ransac"]
    if x_int_r is not None:
        axs[1, 0].scatter([x_int_r], [y_int_r], color='red', label='Bottom (RANSAC)')
        axs[1, 1].scatter([x_int_r], [y_int_r], color='red')

    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()

    #time.sleep(0.1)
    plt.tight_layout()
    plt.waitforbuttonpress()
    plt.close()


def clean_v_groove(df):
    cleaned_rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        profile = row.values.astype(np.float32)
        profile = remove_outliers_zscore(idx, profile)
        profile = detect_slope_spikes(idx, profile)
        #profile = detect_slope_spikes(idx, profile, threshold=3)
        #profile = detect_slope_spikes(idx, profile, threshold=3)

        #profile = interpolate_nan(profile)
        #profile, mask = remove_internal_reflections(profile, slope_margin=10)

        profile = interpolate_nan(profile)
        #profile = smooth_savgol(profile)
        cleaned_rows.append(profile)

    return pd.DataFrame(cleaned_rows, columns=df.columns)

def save_features(cleaned_data):
    features = []
    for i in range(len(cleaned_data)):
        #profile = smooth_savgol(row)
        clean_y = cleaned_data.iloc[i].values
        lines, points = extract_lines(clean_y)

        left = points[0]
        #bottom = points[1]
        right = points[2]

        result = calculate_groove_lines(clean_y, left, right)
        bottom, y_bottom = result["intersection_ransac"]
        features.append([left, bottom, right])

    labels_df = pd.DataFrame(features, columns=['left', 'bottom', 'right'])

    labels_df.to_csv("Sample_Data/laser_scan_labels.csv", index=False)
    
    print("Saved labels.")
    


def plot_data(data, cleaned_data):
    for i in range(int((len(cleaned_data))*0.1)):
        y = data.iloc[i].values
        clean_y = cleaned_data.iloc[i].values

        x = np.arange(len(y))
        lines, features = extract_lines(clean_y)

        left_index = features[0]
        #bottom_index = features[1]
        right_index = features[2]

        result = calculate_groove_lines(clean_y, left_index, right_index)
        bottom_index, y_bottom = result["intersection_ransac"]

        #start = [0, y[0]]
        #end = [len(y), y[-1]]
        left = [left_index, clean_y[left_index]]
        bottom = [bottom_index, y_bottom]
        right = [right_index, clean_y[right_index]]

        print(left[0], bottom[0], right[0])
        
        plt.figure(figsize=(18, 5))

        # Raw
        plt.subplot(1, 3, 1)
        plt.xlim(0, 2000)
        plt.ylim(-900, -450)
        plt.title("Raw Data")
        plt.scatter(x, y, color="blue", s=3)

        # Cleaned
        plt.subplot(1, 3, 2)
        plt.xlim(0, 2000)
        plt.ylim(-900, -450)
        plt.title("Cleaned Data")
        plt.scatter(x, clean_y, color="blue", s=3)

        # Analysis
        plt.subplot(1, 3, 3)
        plt.xlim(0, 2000)
        plt.ylim(-900, -450)
        plt.title(f"Profile {i+1} Analysis")
        plt.scatter(x, y, color="blue", s=3)

        # Plot the line between two points

        #plt.plot([start[0], left[0]], [start[1], left[1]], color='black', linewidth = 2) 
        plt.plot([left[0], bottom[0]], [left[1], bottom[1]], color='black', linewidth = 2)   
        plt.plot([bottom[0], right[0]], [bottom[1], right[1]], color='black', linewidth = 2)  
        #plt.plot([right[0], end[0]], [right[1], end[1]], color='black', linewidth = 2)  
        '''
        #plot lines of best fit 
        for name, params in lines.items():
            xs = np.arange(params['start'], params['end'])
            ys = params['slope'] * xs + params['intercept']
            plt.plot(xs, ys, label=name)
        '''

        '''
        for feature in features:
            #plt.axvline(feature, color='r', linestyle=':', linewidth=1)
            plt.plot(feature, clean_y[feature], 'kx', markersize=8, markeredgewidth=2)
        '''
        plt.plot(left[0], left[1], 'kx', markersize=8, markeredgewidth=2)
        plt.plot(bottom[0], bottom[1], 'kx', markersize=8, markeredgewidth=2)
        plt.plot(right[0], right[1], 'kx', markersize=8, markeredgewidth=2)


        plt.legend()
        plt.tight_layout()
        plt.waitforbuttonpress()
        plt.close()

if __name__ == '__main__':
    filepath = filepaths[0]
    data = load_data(filepath)
    cleaned_data = clean_v_groove(data)
    #print(data.shape)

    for i in range(1):
        y = data.iloc[i].values
        clean_y = cleaned_data.iloc[i].values
        left, bottom, right = extract_features(clean_y)
        result = calculate_groove_lines(clean_y, left, right)
        plot_groove_lines(result)
        bottom, y_bottom = result["intersection_ransac"] 

    #save_features(cleaned_data.copy())
    #plot_data(data, cleaned_data.copy())
    plt.close()
