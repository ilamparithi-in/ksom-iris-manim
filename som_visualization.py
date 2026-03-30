import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from som_core import SOM


def plot_pca_projection(X, weights):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    gx, gy, d = weights.shape
    weights_pca = pca.transform(weights.reshape(-1, d)).reshape(gx, gy, 2)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(X_pca[:, 0], X_pca[:, 1], c='#aaaaaa', s=18, alpha=0.5, zorder=2, label='Data')

    for i in range(gx):
        for j in range(gy):
            pos = weights_pca[i, j]
            if j + 1 < gy:
                nb = weights_pca[i, j + 1]
                ax.plot([pos[0], nb[0]], [pos[1], nb[1]],
                        color='#3498db', linewidth=0.8, alpha=0.6, zorder=3)
            if i + 1 < gx:
                nb = weights_pca[i + 1, j]
                ax.plot([pos[0], nb[0]], [pos[1], nb[1]],
                        color='#3498db', linewidth=0.8, alpha=0.6, zorder=3)

    neuron_xy = weights_pca.reshape(-1, 2)
    ax.scatter(neuron_xy[:, 0], neuron_xy[:, 1],
               c='#e74c3c', s=30, zorder=4, label='Neurons')

    ax.set_title('SOM PCA Projection')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_bmu_mapping(X, y, som):
    palette = ['#e74c3c', '#2ecc71', '#3498db']
    class_names = ['setosa', 'versicolor', 'virginica']
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(som.grid_x):
        for j in range(som.grid_y):
            ax.plot(j, i, 's', color='#dddddd', markersize=18, alpha=0.3, zorder=1)

    for xi, yi in zip(X, y):
        bmu = som.find_bmu(xi)
        jitter = rng.uniform(-0.25, 0.25, 2)
        ax.scatter(bmu[1] + jitter[0], bmu[0] + jitter[1],
                   c=palette[yi], s=25, alpha=0.75, zorder=2)

    handles = [mpatches.Patch(facecolor=palette[i], label=class_names[i]) for i in range(3)]
    ax.legend(handles=handles)
    ax.set_title('BMU Mapping')
    ax.set_xlabel('Grid Column')
    ax.set_ylabel('Grid Row')
    ax.set_xlim(-0.5, som.grid_y - 0.5)
    ax.set_ylim(-0.5, som.grid_x - 0.5)
    plt.tight_layout()
    plt.show()


def compute_u_matrix(weights):
    gx, gy, _ = weights.shape
    u_matrix = np.zeros((gx, gy))
    for i in range(gx):
        for j in range(gy):
            neighbors = []
            if i > 0:
                neighbors.append(weights[i - 1, j])
            if i < gx - 1:
                neighbors.append(weights[i + 1, j])
            if j > 0:
                neighbors.append(weights[i, j - 1])
            if j < gy - 1:
                neighbors.append(weights[i, j + 1])
            u_matrix[i, j] = np.mean([np.linalg.norm(weights[i, j] - nb) for nb in neighbors])
    return u_matrix


def plot_cluster_purity(X, y, som, class_names, title='SOM Cluster Purity'):
    grid = {}
    for xi, yi in zip(X, y):
        bmu = som.find_bmu(xi)
        grid.setdefault(bmu, []).append(yi)

    purity_map = np.full((som.grid_x, som.grid_y), np.nan)

    for (r, c), labels in grid.items():
        dominant = max(set(labels), key=labels.count)
        purity_map[r, c] = labels.count(dominant) / len(labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(purity_map, cmap='YlGn', vmin=0, vmax=1,
                   interpolation='nearest', origin='upper')
    plt.colorbar(im, ax=ax, label='Cluster purity')

    for (r, c), labels in grid.items():
        dominant = max(set(labels), key=labels.count)
        ax.text(c, r, class_names[dominant][0],
                ha='center', va='center', fontsize=7,
                color='#333333', fontweight='bold')

    ax.set_title(title)
    ax.set_xlabel('Grid Column')
    ax.set_ylabel('Grid Row')
    plt.tight_layout()
    plt.show()


def plot_u_matrix(u_matrix, title='U-Matrix'):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(u_matrix, cmap='bone_r', interpolation='nearest', origin='upper')
    plt.colorbar(im, ax=ax, label='Mean distance to neighbors')
    ax.set_title(title)
    ax.set_xlabel('Grid Column')
    ax.set_ylabel('Grid Row')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    GRID_X = 10
    GRID_Y = 10
    N_EPOCHS = 3
    ALPHA_INIT = 0.5
    SIGMA_INIT = 5.0

    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    som = SOM(GRID_X, GRID_Y, X.shape[1], ALPHA_INIT, SIGMA_INIT)
    som.train(X, N_EPOCHS)

    iris_class_names = ['setosa', 'versicolor', 'virginica']

    plot_pca_projection(X, som.weights)
    plot_bmu_mapping(X, y, som)
    plot_cluster_purity(X, y, som, iris_class_names, title='SOM Cluster Purity — Iris Dataset')
    u_matrix = compute_u_matrix(som.weights)
    plot_u_matrix(u_matrix, title='U-Matrix — Iris Dataset')
