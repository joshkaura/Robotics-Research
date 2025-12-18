'''
Program that builds a 3D model from a preliminary scan of a joint with a laser distance sensor
and plots the approximate/recommended weld path along that 3D joint model.
'''

from dataclasses import dataclass
import numpy as np
import pandas as pd 
from scipy.stats import zscore
import random
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

@dataclass
class PathPlanningParams:
    smoothing_factor: float = 0.3

# Load data
def load_data_csv(filepath, flip=True):
    data = pd.read_csv(filepath)
    data = data.replace(0, np.nan)
    if flip:
        data = -data
    return data

def save_plot(fig, filepath="Results/data_preprocessing.png", dpi=300):
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Saved final plot to: {filepath}")

def generate_v_groove(length=1000, reflection = False):

    def generate_profile(length = 1000, last_bottom_index = None):
        width_range = (700, 800)
        flat_top_height = -470
        noise_level = 0.01

        profile = np.zeros(length)

        # Random total groove width at the top
        top_width = random.randint(*width_range)

        # Random bottom height (tip of the V)
        bottom_height = random.randint(flat_top_height - 200, flat_top_height - 180)

        # Decide bottom index first
        if last_bottom_index is not None:
            bottom_index = random.randint(
                max(50, last_bottom_index - 5),
                min(length - 50, last_bottom_index + 5),
            )
        else:
            bottom_index = random.randint(length//4, length - length//4)

        # Derive left and right edges from bottom_index and top_width
        half_width = top_width // 2
        left_edge = max(0, bottom_index - half_width)
        right_edge = min(length - 1, bottom_index + half_width)

        # Flat top left
        profile[:left_edge] = flat_top_height

        # Left slope: connect top-left → bottom
        profile[left_edge:bottom_index + 1] = np.linspace(
            flat_top_height, bottom_height, bottom_index - left_edge + 1
        )

        # Right slope: connect bottom → top-right
        profile[bottom_index:right_edge + 1] = np.linspace(
            bottom_height, flat_top_height, right_edge - bottom_index + 1
        )

        # Flat top right
        profile[right_edge:] = flat_top_height

        # Add noise
        profile += np.random.normal(0, noise_level, length)

        return profile.astype(np.float32), [left_edge, bottom_index, right_edge]

    def create_dataset(n_samples=1000, profile_len=1000):
        profiles = []
        labels = []

        last_bottom_index = None
        for _ in range(n_samples):
            profile, features = generate_profile(length=profile_len, last_bottom_index=last_bottom_index)
            profiles.append(profile)
            labels.append(features)
            last_bottom_index = features[1]

        #print(labels)
        return pd.DataFrame(profiles), labels

    data, labels = create_dataset()

    return data

#Clean Data
def clean_data(df):
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

#Joint Analysis
def detect_seam_features(profile):

    def smooth_savgol(profile, window_length=51, polyorder=2):
        window_length = min(window_length, len(profile) // 2 * 2 - 1)
        return savgol_filter(profile, window_length, polyorder)

    def extract_edges(profile):
        profile = smooth_savgol(profile)
        gradient = np.gradient(profile)
        left = np.argmax(gradient[:len(gradient)//2] < -0.2)
        right = len(gradient) - 1 - np.argmax(gradient[::-1][:len(gradient)//2] > 0.2) #scanning gradient backwards
        bottom_simple = np.argmin(profile)
        #print(left, bottom, right)
        return left, bottom_simple, right

    def ransac_line_fit(x, y):
        """Fits a line using RANSAC and returns slope, intercept, inlier mask."""
        x_reshaped = x.reshape(-1, 1)
        model = RANSACRegressor(LinearRegression(), residual_threshold=0.5, max_trials=100)
        model.fit(x_reshaped, y)
        slope = model.estimator_.coef_[0]
        intercept = model.estimator_.intercept_
        return slope, intercept, model.inlier_mask_

    x = np.arange(len(profile))

    left, bottom_simple, right = extract_edges(profile)

    fit_distance = (right - left)//2
    
    # Get slope regions
    x_left = x[left:left + fit_distance]
    y_left = profile[left:left + fit_distance]
    x_right = x[right - fit_distance:right]
    y_right = profile[right - fit_distance:right]

    # Interpolate any NaNs
    #y_left_clean = interpolate_nan(y_left.copy())
    #y_right_clean = interpolate_nan(y_right.copy())

    '''
    # Standard linear fit
    coeffs_left = np.polyfit(x_left, y_left, deg=1)
    coeffs_right = np.polyfit(x_right, y_left, deg=1)

    m1, c1 = coeffs_left
    m2, c2 = coeffs_right

    x_intersect = (c2 - c1) / (m1 - m2) if m1 != m2 else 0.0
    y_intersect = m1 * x_intersect + c1 if x_intersect != 0.0 else 0.0

    features_ols = {
      "x": x,
        "x_left": x_left, "y_left": y_left, "y_left_clean": y_left,
        "x_right": x_right, "y_right": y_right, "y_right_clean": y_right,
        "polyfit_left": coeffs_left, "polyfit_right": coeffs_right,
        "intersection_polyfit": (int(x_intersect), y_intersect),
    }
    '''

    # RANSAC fit
    slope_left, intercept_left, inliers_left = ransac_line_fit(x_left.copy(), y_left.copy())
    slope_right, intercept_right, inliers_right = ransac_line_fit(x_right.copy(), y_right.copy())

    x_ransac_intersect = (intercept_right - intercept_left) / (slope_left - slope_right) if slope_left != slope_right else 0.0
    y_ransac_intersect = slope_left * x_ransac_intersect + intercept_left if x_ransac_intersect != 0.0 else 0.0


    features_ransac = {
        "x": x,
        "x_left": x_left, "y_left": y_left, "y_left_clean": y_left,
        "x_right": x_right, "y_right": y_right, "y_right_clean": y_right,
        "ransac_left": (slope_left, intercept_left, inliers_left),
        "ransac_right": (slope_right, intercept_right, inliers_right),
        "bottom": (int(x_ransac_intersect), y_ransac_intersect)
    }

    return features_ransac["bottom"][0]

# Handle Offsets
def calculate_nominal_offsets(seam_index, nominal_index):
    return seam_index - nominal_index #note this returns tcp offset from nominal path, not the last tcp position

def smooth_offsets(offsets, smoothing_factor=3.0):
    smoothed = []
    last = offsets[0]
    for o in offsets:
        last = (1 - smoothing_factor) * last + smoothing_factor * o
        smoothed.append(last)
    return np.array(smoothed)

def calculate_tcp_offset(seam_index, prev_seam_index):
    tcp_offset = seam_index - prev_seam_index
    return tcp_offset

# Plotting
def plot_surface_and_path(df, seam_indices, smoothed_offsets, nominal_index):
    seam_indices = np.array(seam_indices)

    X, Y = np.meshgrid(range(df.shape[1]), range(df.shape[0]))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, df.values, cmap="viridis", alpha=0.8)
    
    path_x = seam_indices
    path_y = np.arange(len(seam_indices))
    path_z = [df.iloc[i, seam_indices[i]] for i in range(len(seam_indices))]
    ax.plot(path_x, path_y, path_z, color="red", label="Robot Path")
    
    ax.set_xlabel("Scan Points (X)")
    ax.set_ylabel("Scan Index (Y)")
    ax.set_zlabel("Surface Height (Z)")
    ax.legend()
    return fig

def plot_offsets(offsets, smoothed_offsets):
    scan_no = np.arange(len(offsets))
    fig, ax = plt.subplots()
    fig.set_size_inches(6,6)

    # central line for reference (TCP at origin)
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2)
    ax.plot(offsets, scan_no, label="Raw Offsets")
    ax.plot(smoothed_offsets, scan_no, label="Smoothed Offsets")
    ax.set_xlabel("Scan Index")
    ax.set_ylabel("Offset")
    ax.set_title("Robot Path")
    ax.legend()
    return fig



# Pipeline
class WeldPathPipeline:
    def __init__(self, params: PathPlanningParams):
        self.params = params
        self.tcp_position = 0.0
        self.tcp_offset = 0.0

    def run(self, df, nominal_index, synthetic = False):
        if synthetic == False:
            df_clean = clean_data(df)
        else:
            df_clean = df
        seam_indices = []
        offsets_nominal = []
        for idx, (_, scan) in enumerate(df_clean.iterrows()):
            seam_position = detect_seam_features(scan)
            offset_nominal = calculate_nominal_offsets(seam_position, nominal_index)
            seam_indices.append(seam_position)
            offsets_nominal.append(offset_nominal)

            # start robot at first seam position
            if idx == 0:
                print(f"Starting Robot at Position: {offset_nominal}")

            # move robot
        
        smoothed_offsets = smooth_offsets(offsets_nominal, self.params.smoothing_factor)

        #plotting
        fig_surface = plot_surface_and_path(df_clean, seam_indices, smoothed_offsets, nominal_index)
        fig_offsets = plot_offsets(offsets_nominal, smoothed_offsets)
        return fig_surface, fig_offsets

if __name__ == "__main__":
    synthetic = False

    if synthetic == False:
        df = load_data_csv("Sample_Data/lds_scan1.csv")
    else: 
        df = generate_v_groove()

    nominal_index = df.shape[1] // 2 # tcp behind the centre of the laser line
    pipeline = WeldPathPipeline(PathPlanningParams(smoothing_factor=0.3))
    fig_surface, fig_offsets = pipeline.run(df, nominal_index, synthetic=synthetic)

    save_plot(fig_surface, filepath="Results/weld_path_3D.png")
    save_plot(fig_offsets, filepath="Results/weld_path_2D.png")
    plt.show()
    plt.close()