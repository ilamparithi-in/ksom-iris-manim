import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

CLUSTER_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]


def load_seeds(filepath):
    data = np.loadtxt(filepath)
    X = data[:, :-1]
    y = data[:, -1].astype(int) - 1
    return X, y


def project_2d(X):
    pca = PCA(n_components=2)
    return pca.fit_transform(X)


def plot_cluster_map(X_2d, cluster_labels, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, color in enumerate(CLUSTER_COLORS):
        mask = cluster_labels == k
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color, label=f"Cluster {k + 1}",
            s=40, edgecolors="k", linewidths=0.4, alpha=0.85
        )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(framealpha=0.7)
    plt.tight_layout()


def compute_distance_map(X, kmeans):
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    distances = np.linalg.norm(X - centroids[labels], axis=1)
    return distances


def plot_distance_map(X_2d, distances, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=distances, cmap="plasma",
        s=40, edgecolors="k", linewidths=0.4, alpha=0.9
    )
    plt.colorbar(sc, ax=ax, label="Distance to centroid")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()


def run_pipeline(X, n_clusters, dataset_name):
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(X_norm)

    X_2d = project_2d(X_norm)
    distances = compute_distance_map(X_norm, kmeans)

    plot_cluster_map(X_2d, kmeans.labels_, f"K-Means Cluster Map — {dataset_name}")
    plot_distance_map(X_2d, distances, f"K-Means Distance Map — {dataset_name}")


if __name__ == "__main__":
    iris = load_iris()
    run_pipeline(iris.data, n_clusters=3, dataset_name="Iris")

    X_seeds, _ = load_seeds("seeds.txt")
    run_pipeline(X_seeds, n_clusters=3, dataset_name="Seeds")

    plt.show()
