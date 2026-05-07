import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from scipy.stats import mode

# 1. Load and Normalize Data (No 'y' used for training or clustering!)
iris = load_iris()
X = iris.data
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_norm = (X - mean) / std

# 2. Hyperparameters
grid_x, grid_y = 8, 8
input_dim = X_norm.shape[1]
epochs = 2000
learning_rate_0 = 0.5
sigma_0 = max(grid_x, grid_y) / 2.0

# 3. Initialize Weights
np.random.seed(7)
weights = np.random.uniform(-1, 1, (grid_x, grid_y, input_dim))

print("Starting Manual SOM Training...")

# 4. Manual Training Loop (Unchanged)
time_constant = epochs / np.log(sigma_0)
for epoch in range(epochs):
    lr = learning_rate_0 * np.exp(-epoch / epochs)
    sigma = sigma_0 * np.exp(-epoch / time_constant)

    random_idx = np.random.randint(0, len(X_norm))
    x_t = X_norm[random_idx]

    distances = np.sum((weights - x_t) ** 2, axis=2)
    bmu_idx = np.unravel_index(np.argmin(distances), (grid_x, grid_y))
    bmu_x, bmu_y = bmu_idx

    for i in range(grid_x):
        for j in range(grid_y):
            dist_to_bmu_sq = (i - bmu_x)**2 + (j - bmu_y)**2
            if dist_to_bmu_sq < (sigma ** 2):
                influence = np.exp(-dist_to_bmu_sq / (2 * (sigma ** 2)))
                weights[i, j, :] += lr * influence * (x_t - weights[i, j, :])

print("Training Complete!")

# ---------------------------------------------------------
# NEW SECTION: PURE UNSUPERVISED CLUSTERING
# ---------------------------------------------------------

# 5. Calculate the U-Matrix Manually
# For every neuron, calculate the average distance to its immediate neighbors
u_matrix = np.zeros((grid_x, grid_y))
for i in range(grid_x):
    for j in range(grid_y):
        dist = 0
        count = 0
        # Check all 8 neighboring cells
        for ni in range(max(0, i-1), min(grid_x, i+2)):
            for nj in range(max(0, j-1), min(grid_y, j+2)):
                if ni == i and nj == j:
                    continue # Skip itself
                # Calculate Euclidean distance between neuron weights
                dist += np.linalg.norm(weights[i, j] - weights[ni, nj])
                count += 1
        u_matrix[i, j] = dist / count

# 6. Cluster the Neurons using K-Means (Finding the 3 valleys)
# We flatten the 8x8x4 grid into 64 rows of 4D vectors
flattened_weights = weights.reshape(-1, input_dim)

# Ask K-Means to find 3 clusters among the 64 neurons
kmeans = KMeans(n_clusters=3, random_state=42)
neuron_labels = kmeans.fit_predict(flattened_weights)

# Reshape the 1D list of 64 labels back into the 8x8 grid
cluster_map = neuron_labels.reshape(grid_x, grid_y)

# 7. Accuracy Calculation using Majority Voting
y_true = iris.target
y_pred_unsupervised = []

# Map every data point to the cluster assigned to its BMU
for x_t in X_norm:
    distances = np.sum((weights - x_t) ** 2, axis=2)
    bmu_idx = np.unravel_index(np.argmin(distances), (grid_x, grid_y))
    y_pred_unsupervised.append(cluster_map[bmu_idx[0], bmu_idx[1]])

y_pred_unsupervised = np.array(y_pred_unsupervised)

# Create a container for our aligned predictions
y_pred_aligned = np.zeros_like(y_pred_unsupervised)

for i in range(3):  # For each of our 3 K-Means clusters
    mask = (y_pred_unsupervised == i)
    if np.any(mask):
        # Find which ground-truth label (0, 1, or 2) is most common in this cluster
        majority_label = mode(y_true[mask], keepdims=True)[0][0]
        # Assign that majority label to every point in this cluster
        y_pred_aligned[mask] = majority_label

# Calculate percentage
correct_matches = np.sum(y_pred_aligned == y_true)
accuracy_percent = (correct_matches / len(y_true)) * 100

print(f"--- Clustering Results ---")
print(f"Correctly Grouped: {correct_matches} / 150")
print(f"Accuracy: {accuracy_percent:.2f}%")
# 8. Visualization
plt.figure(figsize=(10, 8))

# Draw the U-Matrix background (Mountain ranges)
plt.pcolor(u_matrix, cmap='bone_r', alpha=0.8)
plt.colorbar(label='U-Matrix: Average Distance to Neighbors')

# Plot the Unsupervised Clusters
colors = ['red', 'green', 'blue']

# Iterate through every data point and assign it to a cluster
for idx, x_t in enumerate(X_norm):
    # Find BMU for this data point
    distances = np.sum((weights - x_t) ** 2, axis=2)
    bmu_idx = np.unravel_index(np.argmin(distances), (grid_x, grid_y))

    # What cluster does this BMU belong to?
    assigned_cluster = cluster_map[bmu_idx[0], bmu_idx[1]]

    plot_x = bmu_idx[1] + 0.5 + np.random.uniform(-0.3, 0.3) # Shift by 0.5 to center in pcolor
    plot_y = bmu_idx[0] + 0.5 + np.random.uniform(-0.3, 0.3)

    plt.scatter(plot_x, plot_y, color=colors[assigned_cluster],
                edgecolors='white', s=50, alpha=0.8)

# Add grid lines for clarity
for i in range(grid_x + 1):
    plt.axhline(i, color='black', linewidth=0.5, alpha=0.3)
    plt.axvline(i, color='black', linewidth=0.5, alpha=0.3)

plt.title("Pure Unsupervised KSOM: K-Means on Trained Weights + U-Matrix")
plt.xlim(0, grid_x)
plt.ylim(0, grid_y)
plt.tight_layout()
plt.show()
