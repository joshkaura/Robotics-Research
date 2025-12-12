#3D Modelling and Analysis of Laser Scans

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


filepaths = [
    "Sample_Data/lds_scan1.csv",
    "Sample_Data/lds_scan2.csv"
]

def load_data(filepath, flip=True):
    data = pd.read_csv(filepath)
    data = data.replace(0, np.nan)
    if flip:
        data = -data
    return data

def save_plot(fig, filepath="Results/3D_surface_modelling.png", dpi=300):
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"Saved plot to: {filepath}")

def prepare_data(data):
    X = np.arange(data.shape[1])
    Y = np.arange(data.shape[0])
    X, Y = np.meshgrid(X,Y)
    Z = data.values

    return X,Y,Z

def plot_data_3D(X, Y, Z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_title("3D Model of Surface Scan with Laser")
    #plt.pause(0.5)

    save_plot(fig)

    plt.show()

    plt.close()


def main():
    data = load_data(filepaths[1], flip=False)
    #print(data.shape)
    #print(data.head())

    X, Y, Z = prepare_data(data)

    plot_data_3D(X, Y, Z)





if __name__ == "__main__":
    main()