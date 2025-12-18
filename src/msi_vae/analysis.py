import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm.auto import tqdm


def get_latest_run_id(weights_dir):
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    files = [
        os.path.join(weights_dir, f)
        for f in os.listdir(weights_dir)
        if os.path.isfile(os.path.join(weights_dir, f))
    ]

    if not files:
        newest = None
    else:
        newest = max(files, key=os.path.getmtime)
        newest = os.path.basename(newest)
        newest = os.path.splitext(os.path.basename(newest))[0]

    return newest


def run_encoding(means_file, model, test_loader, device):
    if os.path.exists(means_file):
        return np.load(means_file)

    means = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Encoding"):
            data = data.to(device)
            output = model(data)
            means.append(output.mu.cpu().numpy())

    means = np.concatenate(means, axis=0)
    np.save(means_file, means)
    return means


def run_tsne(results_path, data):
    if os.path.exists(results_path):
        return np.load(results_path)

    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(data)
    np.save(results_path, tsne_results)
    return tsne_results


def visualize_latent_space_with_tsne(
    tsne_results, plot_title, labels=None, save_path=None
):
    alpha = 0.8
    s = 6
    plt.figure(figsize=(6, 6))
    if labels is not None:
        scatter = plt.scatter(
            tsne_results[:, 0],
            tsne_results[:, 1],
            c=labels,
            cmap="tab20",
            s=s,
            alpha=alpha,
        )
        plt.colorbar(scatter, ticks=np.unique(labels), label="Label")
    else:
        plt.scatter(
            tsne_results[:, 0], tsne_results[:, 1], s=s, alpha=alpha, color="steelblue"
        )

    plt.title(plot_title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg")

    plt.show()


def find_optimal_k(latent_means, k_min=2, k_max=10, random_state=42, save_path=None):
    inertias = []
    Ks = range(k_min, k_max + 1)

    for k in Ks:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        kmeans.fit(latent_means)
        inertias.append(kmeans.inertia_)
        print(f"k={k}, inertia={kmeans.inertia_:.2f}")

    plt.figure(figsize=(6, 4))
    plt.plot(Ks, inertias, "o-", linewidth=2)
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (Within-cluster SSE)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg")

    plt.show()

    return inertias


def convert_cluster_centers_to_spectra(model, kmeans, device, save_path=None):
    if save_path:
        if os.path.exists(save_path):
            return np.load(save_path)

    with torch.no_grad():
        cluster_spectra = model.decoder(
            torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        )
        cluster_spectra = cluster_spectra.detach().cpu().numpy()
        if save_path:
            np.save(save_path, cluster_spectra)
        return cluster_spectra


def plot_cluster_centroid_spectra(cluster_centroid_spectra, save_path):
    K, num_ch = cluster_centroid_spectra.shape
    x = np.arange(num_ch)

    cols = min(4, K)
    rows = int(np.ceil(K / cols))
    plt.figure(figsize=(3.5 * cols, 2.5 * rows))

    for k in range(K):
        ax = plt.subplot(rows, cols, k + 1)
        ax.plot(x, cluster_centroid_spectra[k])
        ax.set_title(f"Cluster {k} Centroid")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg")

    plt.show()


def plot_all_clusters(labels, shape, best_k, save_path=None):
    H, W = shape
    cluster_labels = np.asarray(labels)
    assert H * W == cluster_labels.shape[0]

    cols = min(4, best_k)
    rows = math.ceil(best_k / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(best_k):
        mask = (cluster_labels == i).reshape(H, W)
        ax = axes[i]
        im = ax.imshow(mask, cmap="grey")
        ax.set_title(f"Cluster {i}")
        ax.axis("off")

    for j in range(best_k, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Spatial Representation of All K-Means Clusters", fontsize=14, y=0.95)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        plt.savefig(save_path, format="svg")

    plt.show()


def plot_summary_spectra(mean_spectrum, max_spectrum, mz_axis=None, save_path=None):
    spectra = {
        "Mean Spectrum": mean_spectrum,
        "Maximum Spectrum": max_spectrum,
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for ax, (title, spec) in zip(axes, spectra.items()):
        x = mz_axis if mz_axis is not None else torch.arange(len(spec))
        ax.plot(x, spec, color="steelblue", linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Bin Index")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg")

    plt.show()
