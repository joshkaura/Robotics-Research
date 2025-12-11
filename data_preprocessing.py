# Data Pre-Processing Testing

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore, linregress
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from sklearn.neighbors import LocalOutlierFactor
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.cluster import DBSCAN
from skimage.restoration import denoise_tv_chambolle
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter1d

filepaths = [
    "Sample_Data/lds_scan1.csv"
]

def save(fig, filepath="Results/data_preprocessing.png", dpi=300):
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Saved final plot to: {filepath}")

def load_data(filepath, flip=True):
    data = pd.read_csv(filepath)
    #data = data.replace(0, np.nan)
    if flip:
        data = -data
    return data

def remove_outliers_zscore(profile, threshold=2.2):
    z_scores = zscore(profile, nan_policy='omit')
    outliers = np.abs(z_scores) > threshold
    #print(f"Row {idx} — Outliers: {np.sum(outliers)}")
    profile[outliers] = np.nan
    return profile

def remove_outliers_zscore_mad(profile, threshold=2.5):
    median = np.nanmedian(profile)
    mad = np.nanmedian(np.abs(profile - median))
    #print(f"MAD: {mad}")

    if mad == 0:
        raise ValueError("MAD = zero, so z-scores cannot be calculated.")

    z_scores = 0.6745 * (profile - median) / mad
    outliers = np.abs(z_scores) > threshold  # mask of outliers
    profile[outliers] = np.nan

    return profile

def remove_slope_spikes(idx, profile, threshold=3):
    diff = np.abs(np.diff(profile, prepend=profile[0]))
    spikes = diff > threshold
    #print(f"Row {idx} — Spikes: {np.sum(spikes)}")
    profile[spikes] = np.nan
    return profile

def smooth_median(profile, kernel_size = 50):
    # Apply median filtering to reduce noise
    return median_filter(profile, size=kernel_size)

def smooth_savgol(profile, window_length=51, polyorder=2):
    window_length = min(window_length, len(profile) // 2 * 2 - 1)
    return savgol_filter(profile, window_length, polyorder)

def remove_lof_outliers(profile, n_neighbors = 20) -> np.ndarray:
    # Detect and remove outliers using Local Outlier Factor
    x = np.arange(len(profile)).reshape(-1, 1)
    y = profile.reshape(-1, 1)
    data = np.hstack((x, y))
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    try:
        labels = lof.fit_predict(data)
    except ValueError:
        return profile  # Return unchanged if too few points
    profile_filtered = profile.copy()
    profile_filtered[labels == -1] = np.nan
    return profile_filtered

def remove_outliers_dbscan(profile, eps = 2.5, min_samples = 5) -> np.ndarray:
    x = np.arange(len(profile))
    coords = np.column_stack((x, profile))

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords)

    cleaned = profile.copy()
    cleaned[labels == -1] = np.nan  # remove outliers
    return cleaned

def hampel_filter(profile, window_size = 50, n_sigmas = 200.0):
    s = pd.Series(profile)
    k = 1.4826  # scale factor for Gaussian distribution
    rolling_median = s.rolling(window_size, center=True).median()
    MAD = k * (s.rolling(window_size, center=True).apply(lambda x: np.median(np.abs(x - np.median(x)))))
    diff = np.abs(s - rolling_median)
    outliers = diff > n_sigmas * MAD
    filtered = s.copy()
    filtered[outliers] = np.nan
    return filtered

def smooth_total_variation(profile, weight = 0.1):
    return denoise_tv_chambolle(profile, weight=weight)

def smooth_loess(profile, frac = 0.01):
    x = np.arange(len(profile))
    y = profile
    lowess_result = lowess(y, x, frac=frac, return_sorted=False)
    return lowess_result

def bilateral_like_smoothing(profile, sigma_s=2.0, sigma_r=0.2):
    # Not a true bilateral filter - more research required
    # Placeholder Gaussian + value-threshold smoothing hybrid.
    smoothed = gaussian_filter1d(profile, sigma_s)
    diff = np.abs(profile - smoothed)
    mask = diff < sigma_r
    result = np.where(mask, smoothed, profile)
    return result

def interpolate_missing(profile):
    nans = np.isnan(profile)
    not_nans = ~nans
    if np.any(not_nans):
        indices = np.arange(len(profile))
        profile[nans] = np.interp(indices[nans], indices[not_nans], profile[not_nans])
    return profile

def clean_data1(df):
    df = df.replace(0, np.nan)
    cleaned_rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        profile = row.values.astype(np.float32)

        profile = remove_outliers_zscore_mad(profile)

        profile = interpolate_missing(profile)

        #profile = remove_outliers_zscore(idx, profile)

        #profile = detect_slope_spikes(idx, profile)

        profile = remove_lof_outliers(profile)

        #profile = interpolate_missing(profile)

        #profile = remove_outliers_dbscan(profile)

        #profile = smooth_median(profile)

        #profile = smooth_savgol(profile)

        cleaned_rows.append(profile)
    
    return pd.DataFrame(cleaned_rows, columns=df.columns)

def clean_data2(df):
    df = df.replace(0, np.nan)
    cleaned_rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        profile = row.values.astype(np.float32)

        profile = remove_outliers_zscore_mad(profile)

        profile = interpolate_missing(profile)

        #profile = remove_lof_outliers(profile)

        profile = remove_outliers_dbscan(profile)

        #profile = remove_slope_spikes(idx, profile)

        #profile = remove_lof_outliers(profile)

        #profile = interpolate_missing(profile)

        #profile = smooth_loess(profile)

        #profile = smooth_median(profile)

        #profile = hampel_filter(profile)

        #profile = smooth_total_variation(profile)

        #profile = bilateral_like_smoothing(profile)

        #profile = remove_slope_spikes(idx, profile)

        #profile = interpolate_missing(profile)

        #profile = smooth_savgol(profile)


        cleaned_rows.append(profile)
    
    return pd.DataFrame(cleaned_rows, columns=df.columns)

def plot_comparison(raw_df, cleaned_df1, cleaned_df2, row_index=0):
    raw_profile = raw_df.iloc[row_index].values
    cleaned_profile1 = cleaned_df1.iloc[row_index].values
    cleaned_profile2 = cleaned_df2.iloc[row_index].values

    xmin = 0
    xmax = len(raw_profile)
    vmin = min(raw_profile.min(), cleaned_profile1.min(), cleaned_profile2.min())
    vmax = max(raw_profile.max(), cleaned_profile1.max(), cleaned_profile2.max())

    fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    axs[0].scatter(np.arange(len(raw_profile)),raw_profile, label='Raw Profile', color='red', alpha=0.7, s=3)
    axs[0].set_title('Raw Data')
    axs[0].set_xlim(xmin, xmax)
    axs[0].set_ylim(vmin, vmax)
    axs[0].set_xlabel('Point Index')
    axs[0].set_ylabel('Distance')
    axs[0].grid(True)

    axs[1].scatter(np.arange(len(cleaned_profile1)),cleaned_profile1, label='Cleaned Profile', color='green', alpha=0.7, s=3)
    axs[1].set_title('Cleaned Data 1')
    axs[1].set_xlim(xmin, xmax)
    axs[1].set_ylim(vmin, vmax)
    axs[1].set_xlabel('Point Index')
    axs[1].grid(True)

    axs[2].scatter(np.arange(len(cleaned_profile2)),cleaned_profile2, label='Cleaned Profile', color='green', alpha=0.7, s=3)
    axs[2].set_title('Cleaned Data 2')
    axs[2].set_xlim(xmin, xmax)
    axs[2].set_ylim(vmin, vmax)
    axs[2].set_xlabel('Point Index')
    axs[2].grid(True)

    plt.suptitle(f'Raw vs Cleaned Profiles (Row {row_index})')
    plt.tight_layout()
    save(fig)
    plt.waitforbuttonpress()
    plt.close()

    return

if __name__ == "__main__":

    raw_data = load_data(filepaths[0])

    cleaned_data1 = clean_data1(raw_data)

    cleaned_data2 = clean_data2(raw_data)

    for i in range(1):
        plot_comparison(raw_data, cleaned_data1, cleaned_data2, row_index=i)


