import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from scipy.signal import savgol_filter
from scipy.stats import zscore



# ---- Dataset Loader ----
class LaserProfileDataset(Dataset):
    def __init__(self, profiles, labels):
        self.profiles = profiles.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.length = profiles.shape[1]
        self.indices = np.tile(np.arange(self.length), (profiles.shape[0], 1)).astype(np.float32)
        #OR
        # Normalize x-channel to [0, 1]
        #self.indices = np.tile(np.linspace(0, 1, self.length), (profiles.shape[0], 1)).astype(np.float32)

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        profile = self.profiles[idx]
        index_channel = self.indices[idx]
        x = np.stack([profile, index_channel], axis=0)  # Shape (2, N)
        y = self.labels[idx]
        return torch.tensor(x), torch.tensor(y)

class VGrooveKeypointCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=21, padding=3),  # Changed from 1 to 2 input channels - 2nd input is x position (index)
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=12, padding=2),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Predict left, bottom, right indices
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.regressor(x)

def clean_data(X):
    clean_X = []

    for profile in X:
        profile = profile.astype(np.float32)
        #replace 0s with nan
        profile[profile == 0] = np.nan

        #remove outliers by zscore
        z_scores = zscore(profile, nan_policy='omit')
        outliers = np.abs(z_scores) > 2.2 #set threshold accordingly
        #print(f"Row {idx} — Outliers: {np.sum(outliers)}")
        profile[outliers] = np.nan

        #interpolate nans
        nans = np.isnan(profile)
        not_nans = ~nans
        if np.any(not_nans):
            indices = np.arange(len(profile))
            profile[nans] = np.interp(indices[nans], indices[not_nans], profile[not_nans])

        #savgol filter for smoothing
        #profile = savgol_filter(profile, window_length=21, polyorder=2)


        clean_X.append(profile)

    return np.array(clean_X)


def load_data(clean=False):
    profiles_filepath = "Sample_Data/lds_scan1.csv"
    labels_filepath = "Sample_Data/lds_scan1_labels.csv"

    X = pd.read_csv(profiles_filepath).values
    y = pd.read_csv(labels_filepath).values

    print(f"Original Profile sample: {X[0]}")
    print(f"Label sample: {y[0]}")
    print(f"Profiles shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X = -X  # Flip profile

    if clean:
        X = clean_data(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    train_dataset = LaserProfileDataset(X_train, y_train)
    test_dataset = LaserProfileDataset(X_test, y_test)

    return train_dataset, test_dataset, X_test, scaler

def train_model(device, train_dataset, n_epochs=50):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = VGrooveKeypointCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs//2.5, gamma=0.5)

    best_training_loss = float('inf')
    start_time_total = time.time()

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}, Time: {time.time() - start_time:.2f}s")

        if epoch_loss < best_training_loss:
            best_training_loss = epoch_loss
            best_model_state = model.state_dict()

        if epoch_loss <= 25:
            break

    print(f"Total training time: {(time.time()-start_time_total)/60:.2f} mins")
    torch.save(best_model_state, 'Sample_Models/weights_4.pth')
    print("✅ Model saved.")

def evaluate_model(device, test_dataset):
    model = VGrooveKeypointCNN()
    model.load_state_dict(torch.load('Sample_Models/weights_4.pth', weights_only=True))
    model.to(device)
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1)
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    mse = np.mean((all_preds - all_targets) ** 2)
    print(f"\n✅ Test MSE: {mse:.2f}")

    abs_errors = np.abs(all_preds - all_targets)
    avg_abs_error = np.mean(abs_errors)
    median_abs_error = np.median(abs_errors)
    print(f"Average absolute index error per keypoint - {avg_abs_error:.2f} ; median - {median_abs_error:.2f}")

    return all_targets, all_preds, avg_abs_error, mse

def plot_prediction(index, X_test, all_targets, all_preds, scaler, mse, mae):

    profile = np.array(scaler.inverse_transform(X_test[index].reshape(1, -1))[0])
    #print(profile[0])

    x_indices = np.arange(len(profile))

    true_left, true_bottom, true_right = all_targets[index]
    pred_left, pred_bottom, pred_right = all_preds[index]

    fig = plt.figure()

    plt.scatter(x_indices, profile, label='Profile', s=2)

    for i, feature in enumerate(all_targets[index]):
        plt.plot(feature, profile[int(feature)], 'x', color = 'green', markersize=8, markeredgewidth=2, label='True Feature' if i==0 else None)
    
    for i, feature in enumerate(all_preds[index]):
        plt.plot(feature, profile[int(feature)], 'x', color = 'red', markersize=8, markeredgewidth=2, label='Predicted Feature' if i==0 else None)


    plt.legend()
    plt.title("CNN Prediction vs Ground Truth")

    if index == 0:
        fig.text(
            0.5, 0.02,
            f"MSE (indices): {mse:.3f} | MAE (indices): {mae:.3f}",
            ha="center",
            fontsize=10
        )
        save_plot(fig)

    plt.waitforbuttonpress()

def save_plot(fig, filepath="Results/CNN_feature_extraction.png", dpi=300):
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {filepath}")

def main(train=False, clean=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset, X_test, scaler = load_data(clean=clean)

    if train:
        train_model(device, train_dataset, n_epochs=100)

    all_targets, all_preds, mae, mse = evaluate_model(device, test_dataset)

    for i in range(1):
        plot_prediction(i, X_test, all_targets, all_preds, scaler, mse, mae)
        plt.close()

if __name__ == '__main__':
    main(train=False, clean=True)