import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from som_core import SOM
from som_visualization import compute_u_matrix, plot_u_matrix, plot_pca_projection, plot_cluster_purity

FEATURE_NAMES = [
    'Area', 'Perimeter', 'Compactness',
    'Kernel Length', 'Kernel Width',
    'Asymmetry', 'Groove Length',
]
CLASS_NAMES = ['Kama', 'Rosa', 'Canadian']
PALETTE = ['#e74c3c', '#2ecc71', '#3498db']

GRID_X = 10
GRID_Y = 10
N_EPOCHS = 5
ALPHA_INIT = 0.5
SIGMA_INIT = 5.0
SEED = 42


def load_seeds(path):
    data = np.loadtxt(path)
    X = data[:, :7].astype(np.float64)
    y = data[:, 7].astype(int) - 1
    return X, y


def plot_bmu_mapping_seeds(X, y, som):
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(9, 9))

    for i in range(som.grid_x):
        for j in range(som.grid_y):
            ax.plot(j, i, 's', color='#dddddd', markersize=18, alpha=0.3, zorder=1)

    for xi, yi in zip(X, y):
        bmu = som.find_bmu(xi)
        jitter = rng.uniform(-0.25, 0.25, 2)
        ax.scatter(bmu[1] + jitter[0], bmu[0] + jitter[1],
                   c=PALETTE[yi], s=30, alpha=0.8, zorder=2)

    handles = [mpatches.Patch(facecolor=PALETTE[i], label=CLASS_NAMES[i]) for i in range(3)]
    ax.legend(handles=handles, fontsize=11)
    ax.set_title('SOM BMU Mapping — Seeds Dataset', fontsize=13)
    ax.set_xlabel('Grid Column')
    ax.set_ylabel('Grid Row')
    ax.set_xlim(-0.5, som.grid_y - 0.5)
    ax.set_ylim(-0.5, som.grid_x - 0.5)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    X, y = load_seeds('seeds.txt')

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    som = SOM(GRID_X, GRID_Y, X_scaled.shape[1], ALPHA_INIT, SIGMA_INIT, seed=SEED)
    som.train(X_scaled, N_EPOCHS)

    plot_pca_projection(X_scaled, som.weights)
    plot_bmu_mapping_seeds(X_scaled, y, som)
    plot_cluster_purity(X_scaled, y, som, CLASS_NAMES, title='SOM Cluster Purity — Seeds Dataset')
    u_matrix = compute_u_matrix(som.weights)
    plot_u_matrix(u_matrix, title='U-Matrix — Seeds Dataset')
