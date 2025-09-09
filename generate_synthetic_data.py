import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def generate_butt_joint_profile(length=100, depth=10, noise_level=0.2):
    """
    Simulate a Butt joint scan: sharp valley
    """
    x = np.linspace(0, 1, length)
    profile = np.zeros_like(x)

    # Valley in the middle
    center = length // 2
    profile[center] = -depth

    # Smooth valley with parabolic shape
    spread = random.randint(2, 6)  # narrow valley width
    for i in range(1, spread):
        profile[center - i] = -depth + (i * depth / spread)
        profile[center + i] = -depth + (i * depth / spread)

    # Add Gaussian noise
    profile += np.random.normal(0, noise_level, size=length)
    return profile


def generate_vgroove_joint_profile(length=100, depth=10, noise_level=0.2):
    """
    Simulate a V-groove joint scan: wider V-shaped valley
    """
    x = np.linspace(-1, 1, length)
    slope = random.uniform(4, 10)  # controls the V angle
    profile = -depth + slope * np.abs(x)

    # Clip so top is flat (plate surface)
    profile = np.minimum(profile, 0)

    # Add Gaussian noise
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

if __name__ == "__main__":
    profiles, labels = generate_dataset_csv(n_samples=2000, length=128)

    #save_dataset(profiles, labels, save_prefix="training_data")

    # Quick visualisation of one example from each class

    butt_example = profiles[np.where(np.array(labels) == "Butt")[0][0]]
    vgroove_example = profiles[np.where(np.array(labels) == "V-groove")[0][0]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(butt_example)
    axes[0].set_title("Butt Joint Example")

    axes[1].plot(vgroove_example)
    axes[1].set_title("V-Groove Joint Example")

    plt.waitforbuttonpress()
    plt.close()
