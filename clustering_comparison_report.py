import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from som_core import SOM


def som_predict(som, X, X_train, y_train):
    bmu_label_lists = {}
    for x, label in zip(X_train, y_train):
        bmu = som.find_bmu(x)
        bmu_label_lists.setdefault(bmu, []).append(int(label))
    majority = {bmu: max(set(ls), key=ls.count) for bmu, ls in bmu_label_lists.items()}
    labeled_bmu_arr = np.array(list(majority.keys()))
    result = []
    for x in X:
        bmu = som.find_bmu(x)
        if bmu in majority:
            result.append(majority[bmu])
        else:
            dists = np.linalg.norm(labeled_bmu_arr - np.array(bmu), axis=1)
            nearest = tuple(labeled_bmu_arr[np.argmin(dists)])
            result.append(majority[nearest])
    return np.array(result)


def load_iris_data():
    iris = load_iris()
    X = MinMaxScaler().fit_transform(iris.data)
    return train_test_split(X, iris.target, test_size=0.2, random_state=42, stratify=iris.target)


def load_seeds_data():
    data = np.loadtxt("seeds.txt")
    X = MinMaxScaler().fit_transform(data[:, :-1])
    y = data[:, -1].astype(int) - 1
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def compute_metrics(X_test, y_test, labels):
    mask = labels != -1
    n_unique = len(np.unique(labels[mask])) if mask.sum() > 0 else 0
    if mask.sum() > 1 and n_unique > 1:
        sil = silhouette_score(X_test[mask], labels[mask])
    else:
        sil = float("nan")
    ari = adjusted_rand_score(y_test, labels)
    return sil, ari


COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]


def plot_comparison(X_test, y_test, som_labels, km_labels, pca, dataset_name):
    X_2d = pca.transform(X_test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Clustering Comparison — {dataset_name}", fontsize=14, fontweight="bold")

    label_sets = [
        (som_labels, f"{dataset_name} — SOM"),
        (km_labels, f"{dataset_name} — K-Means"),
    ]

    for ax, (labels, title) in zip(axes, label_sets):
        for lbl in np.unique(labels):
            mask = labels == lbl
            color = "#aaaaaa" if lbl == -1 else COLORS[int(lbl) % len(COLORS)]
            marker_label = "noise" if lbl == -1 else f"Cluster {lbl}"
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=color, s=55, alpha=0.85,
                label=marker_label,
                edgecolors="k", linewidths=0.3,
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.legend(fontsize=8, markerscale=1.1)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()


def print_report(dataset_name, results):
    print(f"\nDataset: {dataset_name}")
    print(f"{'Algorithm':<12} {'Silhouette':>12} {'ARI':>10}")
    print("-" * 36)
    for name, sil, ari in results:
        sil_str = f"{sil:.4f}" if not np.isnan(sil) else "     N/A"
        print(f"{name:<12} {sil_str:>12} {ari:>10.4f}")


def print_conclusions(iris_results, seeds_results):
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)

    for dataset_name, results in [("Iris", iris_results), ("Seeds", seeds_results)]:
        valid = [(n, s, a) for n, s, a in results if not np.isnan(a)]
        best = max(valid, key=lambda r: r[2]) if valid else results[0]
        print(f"\n{dataset_name}:")
        print(f"  Best algorithm by ARI : {best[0]} (ARI = {best[2]:.4f})")

    print("""
Algorithm characteristics:

  SOM       Topology-preserving; neuron weights form a 2D manifold
            that reflects local structure in the input space.
            Cluster assignment via majority labeling of BMU nodes.
            Smooth transitions between clusters reflect learned topology.

  K-Means   Minimises intra-cluster variance; assumes convex,
            similarly sized clusters. Fast and deterministic but
            sensitive to initialisation and feature scale.
            Produces hard, sharply bounded cluster assignments.
""")


def run_dataset(X_train, X_test, y_train, y_test, dataset_name):
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train)

    som = SOM(10, 10, X_train.shape[1], alpha_init=0.5, sigma_init=5.0, seed=42)
    som.train(X_train, epochs=10)
    som_labels = som_predict(som, X_test, X_train, y_train)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(X_train)
    km_labels = km.predict(X_test)

    sil_som, ari_som = compute_metrics(X_test, y_test, som_labels)
    sil_km, ari_km = compute_metrics(X_test, y_test, km_labels)

    results = [
        ("SOM", sil_som, ari_som),
        ("K-Means", sil_km, ari_km),
    ]

    print_report(dataset_name, results)
    plot_comparison(X_test, y_test, som_labels, km_labels, pca, dataset_name)

    return results


if __name__ == "__main__":
    X_tr_i, X_te_i, y_tr_i, y_te_i = load_iris_data()
    X_tr_s, X_te_s, y_tr_s, y_te_s = load_seeds_data()

    iris_results = run_dataset(X_tr_i, X_te_i, y_tr_i, y_te_i, "Iris")
    seeds_results = run_dataset(X_tr_s, X_te_s, y_tr_s, y_te_s, "Seeds")

    print_conclusions(iris_results, seeds_results)
