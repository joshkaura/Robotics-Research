import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def generate_butt_joint_profile(length=128, depth=10, noise_level=0.05,
                                valley_width_range=(20, 25),
                                slope_width_range=(5, 10),
                                plate_flat_range=(10, 20)):

    profile = np.zeros(length)

    valley_w = random.randint(*valley_width_range)
    slope_w = random.randint(*slope_width_range)
    plate_flat = random.randint(*plate_flat_range)

    # Total half-width needed from center to accommodate valley + slopes
    half_needed = slope_w + (valley_w // 2)

    # Ensure we have plate flats on both sides too
    min_center = plate_flat + half_needed
    max_center = length - plate_flat - half_needed - 1
    if min_center > max_center:
        raise ValueError("Profile length too small for requested butt joint geometry.")

    center = random.randint(min_center, max_center)

    # Exact valley slice (no off-by-one)
    valley_start = center - valley_w // 2
    valley_end = valley_start + valley_w  # exact width

    profile[valley_start:valley_end] = -depth

    # Slopes up to 0 on both sides
    # Left slope occupies [valley_start - slope_w, valley_start)
    for i in range(1, slope_w + 1):
        idx = valley_start - i
        profile[idx] = -depth + (i / slope_w) * depth

    # Right slope occupies [valley_end, valley_end + slope_w)
    for i in range(1, slope_w + 1):
        idx = valley_end + (i - 1)
        profile[idx] = -depth + (i / slope_w) * depth

    # Add noise (won't create a single-point spike now because valley is fully defined)
    profile += np.random.normal(0, noise_level, size=length)

    return profile


def generate_vgroove_joint_profile(length=128, depth=8, noise_level=0.05,
                                   flat_plate_range=(10, 20),
                                   slope_width_range=(30, 40)):

    profile = np.zeros(length)

    plate_flat = random.randint(*flat_plate_range)
    slope_w = random.randint(*slope_width_range)

    # Need room: plate_flat + slope_w on both sides + 1 bottom point
    min_center = plate_flat + slope_w
    max_center = length - plate_flat - slope_w - 1
    if min_center > max_center:
        raise ValueError("Profile length too small for requested V-groove geometry.")

    center = random.randint(min_center, max_center)

    # Set the single bottom point
    profile[center] = -depth

    # Build linear slopes up to 0 on both sides
    for i in range(1, slope_w + 1):
        height = -depth + (i / slope_w) * depth  # rises to 0 at i=slope_w
        profile[center - i] = height
        profile[center + i] = height

    # Plate flats remain 0 automatically outside the V

    # Add noise (won't create a single-point spike now because groove is fully defined)
    profile += np.random.normal(0, noise_level, size=length)

    return profile


def generate_dataset_csv(n_samples=1000, length=100, save_prefix="joint_dataset"):
    profiles = []
    labels = []

    for _ in range(n_samples):
        if random.random() < 0.5:
            # Butt joint
            depth = random.uniform(5, 15)
            profile = generate_butt_joint_profile(length=length, depth=depth)
            profiles.append(profile)
            labels.append("Butt")
        else:
            # V-groove joint
            depth = random.uniform(5, 15)
            profile = generate_vgroove_joint_profile(length=length, depth=depth)
            profiles.append(profile)
            labels.append("V-groove")

    profiles = np.array(profiles)
    labels = np.array(labels)

    # Save profiles as CSV (each row = one profile)
    profiles_df = pd.DataFrame(profiles)

    # Save labels as CSV
    labels_df = pd.DataFrame(labels, columns=["label"])

    print(f"Generated {n_samples} samples.")

    return profiles, labels

def save_dataset(profiles, labels, save_prefix="test_dataset", folder_path="Sample_Data/"):
    # Save profiles as CSV (each row = one profile)
    profiles_df = pd.DataFrame(profiles)

    # Save labels as CSV
    labels_df = pd.DataFrame(labels, columns=["label"])

    profiles_df.to_csv(f"{folder_path}{save_prefix}.csv", index=False)
    labels_df.to_csv(f"{folder_path}{save_prefix}_labels.csv", index=False)

    print(f"Saved to {save_prefix}.csv and {save_prefix}_labels.csv")

def save_plot(fig, filepath="Results/synthetic_butt_vgroove.png", dpi=300):
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Saved final plot to: {filepath}")

if __name__ == "__main__":
    profiles, labels = generate_dataset_csv(n_samples=2000, length=128)

    #save_dataset(profiles, labels, save_prefix="training_data")

    # Visualisation of one example from each class

    butt_example = profiles[np.where(np.array(labels) == "Butt")[0][0]]
    vgroove_example = profiles[np.where(np.array(labels) == "V-groove")[0][0]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(butt_example)
    axes[0].set_title("Butt/Narrow Gap Joint Example")

    axes[1].plot(vgroove_example)
    axes[1].set_title("V-Groove Joint Example")

    save_plot(fig)

    plt.show()
    plt.close()
