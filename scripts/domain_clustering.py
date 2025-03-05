import argparse
from torch_geometric.datasets import QM9
import random
import os.path as osp
from sklearn.cluster import AgglomerativeClustering, Birch
from rdkit.ML.Cluster import Butina
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
import pandas as pd
import altair as alt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.manifold import MDS
from scipy.stats import f_oneway, kruskal
import polaris as po
from PIL import Image
import sys

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to 'valid_qm9_indices.pth' in the '/data' directory
data_dir = os.path.join(script_dir, '..', 'data')
valid_indices_path = os.path.join(data_dir, 'valid_qm9_indices.pth')

# Goal is to run on a server, it can be done separately from Rust, ideally do this with bootstrapping (need to ensure data is loaded identically)
# TODO: get another source for butina, this one didn't work well at all

# TODO: clean up file saving locations to be correct
# TODO: trim the tests - figure out which ones are useful and get rid of the rest

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for clustering a small molecule dataset into domains and analysing those domains.")
    parser.add_argument("-r", "--random-seed", type=int, default=42, help="Random seed (default is 42)")
    parser.add_argument("-n", "--sample-size", type=int, required=True, help="Sample size")
    parser.add_argument("-k", "--k_domains", type=int, default=1, help="Number of domains for clustering (default is 1)")
    parser.add_argument("-m", "--clustering_method", type=str, default="Agglomerative", help="Method to cluster the chemical domain (default is Agglomerative)")
    parser.add_argument("-a", "--analysis-only", type=bool, default=False, help="Only visualise molecules based on existing indices (default is False)")
    parser.add_argument("-c", "--clustering-only", type=bool, default=False, help="Only cluster molecules without analysis, useful for bootstrapping (default is False)")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset to run experiments on")
    parser.add_argument("-b", "--bootstrapping", type=int, default=1, help="Bootstrapping iterations (default is 1 ie. no bootstrapping)")
    parser.add_argument("-s", "--split", type=str, default="random", help="Method for splitting data (default is random)")
    parser.add_argument("--determine-cutoff", type=bool, default=False, help="Repeatedly run Butina with different cutoffs to find an optimal one (default is False)")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle the data, required for bootstrapping (default is True)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for loading in data from Polaris (default is 64)")
    return parser.parse_args()

def compute_molecular_descriptors(smiles):
    """
    Compute molecular descriptors for each molecule in the dataset.
    Returns a list of descriptor dictionaries.
    """
    descriptors = []
    for smiles_str in smiles:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            descriptor = {
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "HBD": Descriptors.NumHDonors(mol),
                "HBA": Descriptors.NumHAcceptors(mol),
                "Rings": Descriptors.RingCount(mol),
                "RotBonds": Descriptors.NumRotatableBonds(mol),
                "FormalCharge": Chem.GetFormalCharge(mol),
                "AromaticProp": Descriptors.FractionCSP3(mol)
            }
            descriptors.append(descriptor)
        else:
            descriptors.append(None)  # Handle invalid SMILES gracefully
    return descriptors

def compare_molecular_descriptors(domain_labels, molecular_descriptors, m, save_dir):
    """
    Compare molecular descriptors across domains and visualize the top M descriptors using Matplotlib,
    while handling cases where descriptors are constant across all samples.
    """
    # Create a DataFrame for easier analysis
    df = pd.DataFrame(molecular_descriptors)
    df["Domain"] = domain_labels

    results = {}
    for column in df.columns[:-1]:  # Exclude the "Domain" column
        # Check if the descriptor is constant
        if df[column].nunique() == 1:
            print(f"Skipping {column} as it has identical values for all samples.")
            continue

        # Perform Kruskal-Wallis test for differences between domains
        data_by_domain = [df[df["Domain"] == domain][column].dropna() for domain in df["Domain"].unique()]
        try:
            stat, p_value = kruskal(*data_by_domain)
            results[column] = p_value
        except ValueError as e:
            print(f"Skipping {column} due to error in Kruskal-Wallis test: {e}")
            continue

    # Select the top M descriptors with the smallest p-values
    if not results:
        print("No valid descriptors to compare.")
        return None

    top_descriptors = sorted(results, key=results.get)[:m]

    # Generate boxplots for the top descriptors
    num_plots = len(top_descriptors)
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 5 * num_plots), constrained_layout=True)

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable for a single plot

    for ax, descriptor in zip(axes, top_descriptors):
        # Prepare data for the descriptor
        data_by_domain = [df[df["Domain"] == domain][descriptor].dropna() for domain in df["Domain"].unique()]
        ax.boxplot(data_by_domain, labels=[f"Domain {domain}" for domain in df["Domain"].unique()])
        ax.set_title(f"Distribution of {descriptor} by Domain", fontsize=14)
        ax.set_ylabel(descriptor)
        ax.grid(True, linestyle="--", alpha=0.7)

    # Save the plot as a PNG
    filepath = os.path.join(save_dir, "descriptor_comparison.png")
    plt.savefig(filepath)
    print(f"Saved molecular descriptor plot as {filepath}")

    return results

def plot_pca(domain_labels, molecular_descriptors, save_dir):
    """
    Perform PCA on molecular descriptors and plot the results.
    """
    # Create a DataFrame from molecular descriptors
    df = pd.DataFrame(molecular_descriptors)
    df["Domain"] = domain_labels

    # Drop the "Domain" column for PCA and handle NaN values
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(df.drop(columns="Domain").fillna(0))  # Fill NaN with 0

    # Create a new DataFrame for the PCA results
    df_pca = pd.DataFrame(transformed, columns=["PC1", "PC2"])
    df_pca["Domain"] = domain_labels  # Re-add the domain labels for plotting

    # Plot the PCA results
    chart = alt.Chart(df_pca).mark_circle(size=60).encode(
        x="PC1",
        y="PC2",
        color="Domain:N",
        tooltip=["PC1", "PC2", "Domain"]
    ).interactive()

    # Save the chart
    filepath = os.path.join(save_dir, "pca_plot.html")
    chart.save(filepath)
    print("Saved PCA plot as pca_plot.html")
    return chart

def plot_mds(domain_labels, distance_matrix, save_dir):
    """
    Perform MDS on the distance matrix and plot the results.
    """
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    transformed = mds.fit_transform(distance_matrix)

    df_mds = pd.DataFrame(transformed, columns=["MDS1", "MDS2"])
    df_mds["Domain"] = domain_labels

    chart = alt.Chart(df_mds).mark_circle(size=60).encode(
        x="MDS1",
        y="MDS2",
        color="Domain:N",
        tooltip=["MDS1", "MDS2", "Domain"]
    ).interactive()

    filepath = os.path.join(save_dir, "mds_plot.html")
    chart.save(filepath)
    print("Saved MDS plot as mds_plot.html")
    return chart


def plot_tsne(domain_labels, molecular_descriptors, save_dir):
    df = pd.DataFrame(molecular_descriptors)
    df["Domain"] = domain_labels
    tsne = TSNE(n_components=2, random_state=42)
    transformed = tsne.fit_transform(df.drop(columns="Domain").fillna(0))

    df_tsne = pd.DataFrame(transformed, columns=["tSNE1", "tSNE2"])
    df_tsne["Domain"] = domain_labels

    chart = alt.Chart(df_tsne).mark_circle(size=60).encode(
        x="tSNE1",
        y="tSNE2",
        color="Domain:N",
        tooltip=["tSNE1", "tSNE2", "Domain"]
    ).interactive()

    filepath = os.path.join(save_dir, "tsne_plot.html")
    chart.save(filepath)
    print("Saved t-SNE plot as tsne_plot.html")
    return chart

def plot_umap(domain_labels, molecular_descriptors, save_dir):
    df = pd.DataFrame(molecular_descriptors)
    df["Domain"] = domain_labels
    reducer = umap.UMAP(n_components=2, random_state=42)
    transformed = reducer.fit_transform(df.drop(columns="Domain").fillna(0))

    df_umap = pd.DataFrame(transformed, columns=["UMAP1", "UMAP2"])
    df_umap["Domain"] = domain_labels

    chart = alt.Chart(df_umap).mark_circle(size=60).encode(
        x="UMAP1",
        y="UMAP2",
        color="Domain:N",
        tooltip=["UMAP1", "UMAP2", "Domain"]
    ).interactive()

    filepath = os.path.join(save_dir, "umap_plot.html")
    chart.save(filepath)
    print("Saved UMAP plot as umap_plot.html")
    return chart

def plot_tsne_from_matrix(domain_labels, distance_matrix, save_dir):
    """
    Perform t-SNE using the distance matrix and plot the results.
    """
    # Set init to 'random' since metric is precomputed
    tsne = TSNE(n_components=2, metric="precomputed", init="random", random_state=42)
    transformed = tsne.fit_transform(distance_matrix)

    # Create a DataFrame for visualization
    df_tsne = pd.DataFrame(transformed, columns=["tSNE1", "tSNE2"])
    df_tsne["Domain"] = domain_labels

    # Create an Altair scatter plot
    chart = alt.Chart(df_tsne).mark_circle(size=60).encode(
        x="tSNE1",
        y="tSNE2",
        color="Domain:N",
        tooltip=["tSNE1", "tSNE2", "Domain"]
    ).interactive()

    # Save the plot to an HTML file
    filepath = os.path.join(save_dir, "tsne_distance_matrix.html")
    chart.save(filepath)
    print("Saved t-SNE plot as tsne_distance_matrix.html")
    return chart

def plot_umap_from_matrix(domain_labels, distance_matrix, save_dir):
    """
    Perform UMAP using the distance matrix and plot the results.
    """
    reducer = umap.UMAP(n_components=2, metric="precomputed", random_state=42)
    transformed = reducer.fit_transform(distance_matrix)

    df_umap = pd.DataFrame(transformed, columns=["UMAP1", "UMAP2"])
    df_umap["Domain"] = domain_labels

    chart = alt.Chart(df_umap).mark_circle(size=60).encode(
        x="UMAP1",
        y="UMAP2",
        color="Domain:N",
        tooltip=["UMAP1", "UMAP2", "Domain"]
    ).interactive()

    filepath = os.path.join(save_dir, "umap_distance_matrix.html")
    chart.save(filepath)
    print("Saved UMAP plot as umap_distance_matrix.html")
    return chart

def visualize_molecule(smiles, title=None):
    """
    Generate a 2D visualization of a molecule from its SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol)
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.axis('off')
        if title:
            plt.title(title, fontsize=12)
        plt.show()
    else:
        print(f"Invalid SMILES: {smiles}")

def identify_and_visualise_model(domain_labels, smiles_list, distance_matrix, dataset, clustering_method, k_domains, iteration_seed, sample_size, save_dir):
    """
    Identify three representative molecules for each domain and save them in a single image, 
    with each domain occupying its own row, in the specified save directory.
    """
    # Create a mapping of domain -> indices
    domain_to_indices = {domain: [] for domain in set(domain_labels)}
    for i, domain in enumerate(domain_labels):
        domain_to_indices[domain].append(i)

    representatives = {}
    for domain, indices in domain_to_indices.items():
        # Extract the submatrix for this domain
        submatrix = distance_matrix[np.ix_(indices, indices)]

        # Calculate the mean distance for each molecule in the submatrix
        mean_distances = np.mean(submatrix, axis=1)

        # Start with the molecule closest to the centroid (smallest mean distance)
        representative_indices = [indices[np.argmin(mean_distances)]]

        # Greedily select additional molecules to maximize their distances
        while len(representative_indices) < 3 and len(representative_indices) < len(indices):
            max_distance = 0
            next_rep = None
            for i in indices:
                if i in representative_indices:
                    continue
                # Calculate minimum distance to already-selected representatives
                min_dist_to_selected = min(distance_matrix[i, rep] for rep in representative_indices)
                if min_dist_to_selected > max_distance:
                    max_distance = min_dist_to_selected
                    next_rep = i
            if next_rep is not None:
                representative_indices.append(next_rep)

        # Save the SMILES of the representatives
        representatives[domain] = [smiles_list[i] for i in representative_indices]

    # Create a separate row for each domain using Draw.MolsToGridImage
    row_images = []
    for domain, rep_smiles_list in representatives.items():
        print(f"Domain {domain}: Representative SMILES {rep_smiles_list}")
        mols = [Chem.MolFromSmiles(smiles) for smiles in rep_smiles_list]
        mols = [mol for mol in mols if mol is not None]  # Filter out invalid molecules

        # Create a row for this domain with the domain label
        row_image = Draw.MolsToGridImage(
            mols,
            molsPerRow=len(mols),  # All molecules in one row
            subImgSize=(300, 300),
            legends=[f"Domain {domain}"] * len(mols),  # Label each molecule by domain
        )
        row_images.append(row_image)

    # Combine all row images into one image using Pillow
    if row_images:
        row_heights = [img.height for img in row_images]
        total_height = sum(row_heights)
        max_width = max(img.width for img in row_images)

        # Create a blank canvas
        combined_image = Image.new("RGB", (max_width, total_height))

        # Paste each row image onto the canvas
        y_offset = 0
        for row_image in row_images:
            combined_image.paste(row_image, (0, y_offset))
            y_offset += row_image.height

        # Save the combined image with the naming scheme in the specified save_dir
        filename = f"{dataset}_{clustering_method}_k{k_domains}_seed{iteration_seed}_n{sample_size}.png"
        save_path = os.path.join(save_dir, filename)
        combined_image.save(save_path)
        print(f"Saved combined domain images to {save_path}")

    return representatives

def descriptor_analysis(domain_labels, molecular_descriptors):
    """
    Perform statistical tests to determine if differences in molecular descriptors
    are significant across domains.
    """
    # Create a DataFrame from molecular descriptors and add domain labels
    df = pd.DataFrame(molecular_descriptors)
    df["Domain"] = domain_labels

    # Store results of statistical tests
    results = {
        "Descriptor": [],
        "ANOVA p-value": [],
        "Kruskal-Wallis p-value": []
    }

    for descriptor in df.columns[:-1]:  # Exclude the "Domain" column
        if df[descriptor].nunique() <= 1:  # Skip constant or all-NaN descriptors
            print(f"Skipping descriptor {descriptor} due to insufficient variability.")
            continue

        # Group data by domains for the descriptor
        groups = [df[df["Domain"] == domain][descriptor].dropna() for domain in df["Domain"].unique()]
        print(f"Descriptor: {descriptor}, Group Sizes: {[len(group) for group in groups]}")

        # Perform ANOVA
        if len(groups) > 1 and all(len(group) > 0 for group in groups):
            try:
                anova_stat, anova_p = f_oneway(*groups)
            except ValueError as e:
                print(f"ANOVA failed for {descriptor}: {e}")
                anova_p = None
        else:
            anova_p = None  # Not enough data for ANOVA

        # Perform Kruskal-Wallis test
        if len(groups) > 1 and all(len(group) > 0 for group in groups):
            try:
                kruskal_stat, kruskal_p = kruskal(*groups)
            except ValueError as e:
                print(f"Kruskal-Wallis failed for {descriptor}: {e}")
                kruskal_p = None
        else:
            kruskal_p = None  # Not enough data for Kruskal-Wallis

        # Store results
        results["Descriptor"].append(descriptor)
        results["ANOVA p-value"].append(anova_p)
        results["Kruskal-Wallis p-value"].append(kruskal_p)

    # Convert results to a DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    print(results_df)
    return results_df

def determine_butina_cutoff(distance_matrix, save_dir, min_cutoff=0.01, max_cutoff=0.99, num_cutoffs=5):
    """
    Determine the optimal cutoff for Butina clustering.
    """
    cutoff_values = np.linspace(min_cutoff, max_cutoff, num_cutoffs)
    cluster_counts = []

    for cutoff in cutoff_values:
        domain_labels = butina_clustering(distance_matrix, cutoff)
        num_clusters = len(set(domain_labels)) - (1 if -1 in domain_labels else 0)  # Exclude noise
        cluster_counts.append((cutoff, num_clusters))

    # Plot cutoff vs. cluster count
    cutoff_df = pd.DataFrame(cluster_counts, columns=["Cutoff", "ClusterCount"])
    chart = alt.Chart(cutoff_df).mark_line(point=True).encode(
        x="Cutoff:Q",
        y="ClusterCount:Q"
    ).properties(title="Butina Cutoff vs Cluster Count")
    filepath = os.path.join(save_dir, "butina_cutoff_analysis.html")
    chart.save(filepath)

    # Return the cutoff with the desired number of clusters
    optimal_cutoff = cutoff_df.loc[cutoff_df['ClusterCount'] == max(cluster_counts, key=lambda x: x[1])[1], 'Cutoff']
    return optimal_cutoff.iloc[0] if not optimal_cutoff.empty else None

def tanimoto_distance_matrix(fps):
    n = len(fps)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            distance_matrix[i, j] = 1 - similarity
            distance_matrix[j, i] = 1 - similarity  # Symmetric matrix
    return distance_matrix

def agglomerative_clustering(distance_matrix, k_domains):
    """
    Perform Agglomerative Clustering.
    """
    clustering = AgglomerativeClustering(n_clusters=k_domains, metric='precomputed', linkage='average')
    return clustering.fit_predict(distance_matrix)

def birch_clustering(data, k_domains):
    """
    Perform BIRCH clustering on PCA-reduced data.
    """
    clustering = Birch(n_clusters=k_domains, branching_factor=50, threshold=1.5)
    clustering.fit(data)
    return clustering.predict(data)

def kmeans_clustering(data, k_domains):
    """
    Perform KMeans clustering on PCA-reduced data.
    """
    kmeans = KMeans(n_clusters=k_domains, mode='euclidean', verbose=1)
    return kmeans.fit_predict(data)

def hdbscan_clustering(data):
    """
    Perform HDBSCAN clustering on PCA-reduced data.
    """
    import hdbscan
    clustering = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
    return clustering.fit_predict(data)

def butina_clustering(distance_matrix, cutoff=0.93375):
    """
    Perform Butina clustering using a cutoff.
    """
    flattened_distance_matrix = distance_matrix[np.triu_indices(len(distance_matrix), k=1)]
    clusters = Butina.ClusterData(flattened_distance_matrix, len(distance_matrix), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    clustering_results = np.zeros(len(distance_matrix), dtype=int) - 1
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            clustering_results[idx] = cluster_id
    return clustering_results

# TODO: not all the methods are useing the distance matrix as intended, fix it?? 
def cluster(train_smiles, args, descriptors):
    """
    Perform clustering using the specified method on the dataset.
    """
    # Convert SMILES to RDKit Mol objects
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in train_smiles]
    mol_list = [mol for mol in mol_list if mol is not None]  # Filter out invalid molecules

    # Extract fingerprints or molecular descriptors
    fps = [GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in mol_list]
    distance_matrix = tanimoto_distance_matrix(fps)

    if args.clustering_method == "Agglomerative":
        domain_labels = agglomerative_clustering(distance_matrix, args.k_domains)

    elif args.clustering_method == "Birch":
        reduced_data = PCA(n_components=2).fit_transform(pd.DataFrame(descriptors).fillna(0))
        domain_labels = birch_clustering(reduced_data, args.k_domains)

    elif args.clustering_method == "KMeans":
        reduced_data = PCA(n_components=64).fit_transform(pd.DataFrame(descriptors).fillna(0))
        domain_labels = kmeans_clustering(reduced_data, args.k_domains)

    elif args.clustering_method == "HDBSCAN":
        reduced_data = PCA(n_components=64).fit_transform(pd.DataFrame(descriptors).fillna(0))
        domain_labels = hdbscan_clustering(reduced_data)

    elif args.clustering_method == "Butina":
        domain_labels = butina_clustering(distance_matrix, cutoff=0.93375)

    else:
        raise ValueError(f"Unsupported clustering method: {args.clustering_method}")

    return domain_labels, distance_matrix

def load_qm9_smiles():
    qm9 = QM9(root=osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9'))

    # Filter out molecules that cannot be processed by RDKit
    valid_indices_tensor = torch.load(valid_indices_path)
    qm9 = qm9.index_select(valid_indices_tensor)

    return qm9


def split_qm9_smiles(qm9, args):
    train_smiles = []

    if args.shuffle:
        qm9 = qm9.shuffle()
    else:
        # Ensure reproducibility with manual random seed
        torch.manual_seed(args.random_seed)
        num_data = len(qm9)
        indices = torch.randperm(num_data)
        qm9 = qm9.index_select(indices)
    data_size = args.sample_size

    if args.split == 'random':
        train_index = int(data_size * 0.8)
        test_index = train_index + int(data_size * 0.1)
        val_index = test_index + int(data_size * 0.1)
        train_idx = list(range(train_index))
    else:
        qm9_smiles = [data.smiles for data in qm9[:data_size]]
        Xs = np.zeros(len(qm9_smiles))
        dataset = dc.data.DiskDataset.from_numpy(X=Xs, ids=qm9_smiles)

        splitter = dc.splits.ScaffoldSplitter()
        train_idx, _, _ = splitter.split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    for index, data in enumerate(qm9[:data_size]):
        if index in train_idx:
            train_smiles.append(data.smiles)

    return train_smiles


def load_and_collect_train_smiles(args):
    dataset_name = "BELKA"

    if dataset_name == "BELKA":
        # Define the binding target to get the label from (e.g., 'binds_BRD4', 'binds_HSA', 'binds_sEH')
        proteins = {'BDR4': 'binds_BRD4', 'HSA': 'binds_HSA', 'sEH': 'binds_sEH'}
        binding_protein = 'BDR4'  # User-specified protein
        binding_target = proteins[binding_protein]

        # Load the dataset from PolarisHub
        dataset = po.load_dataset("leash-bio/BELKA-v1")
        dataset_size = dataset.size()[0]
    else:
        raise ValueError("Invalid dataset name specified.")

    # Select random indices for this iteration
    random_indices = np.random.choice(dataset_size, args.sample_size, replace=False)

    # Pre-allocated space-efficient storage for scaffold splitting
    Xs = np.empty(args.sample_size, dtype="U1000")  # Adjust dtype size if necessary for SMILES length
    smiles = np.empty(args.sample_size, dtype="U1000")  # Adjust dtype size if necessary

    train_smiles = []

    # Iterate over the selected random indices in batches
    batch_size = 64
    for batch_start in range(0, args.sample_size, batch_size):
        batch_indices = random_indices[batch_start:batch_start + batch_size]
        if dataset_name == "BELKA":
            smiles_isomeric_list = [dataset.get_data(row, "molecule_smiles") for row in batch_indices]
        else:
            break

        split_list = []
        if args.split == "central_core":
            split_list = [dataset.get_data(row, "split") for row in batch_indices]
        elif args.split == "library":
            split_list = [dataset.get_data(row, "split_group") for row in batch_indices]
        elif args.split == "random":
            # Randomly assign molecules to splits based on indices
            split_list = [
                "train" if i % 10 < 8 else "val" if i % 10 == 8 else "test"
                for i in range(len(smiles_isomeric_list))
            ]

        for i, smiles_isomeric in enumerate(smiles_isomeric_list):
            if args.split == "random" and split_list[i] == "train":
                train_smiles.append(smiles_isomeric)
            elif args.split != "random" and split_list[i] == "train":
                train_smiles.append(smiles_isomeric)

    # Scaffold splitting process (if needed)
    if args.split == "scaffold":
        dataset = dc.data.DiskDataset.from_numpy(X=Xs, ids=smiles)
        splitter = dc.splits.ScaffoldSplitter()
        train_idx, _, _ = splitter.split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

        # Collect SMILES for training points
        train_smiles = [Xs[index] for index in train_idx]

    return train_smiles

def redirect_output(save_dir):
    """Redirect all print statements to a log file."""
    log_file = os.path.join(save_dir, "output.log")
    sys.stdout = open(log_file, "w")
    sys.stderr = sys.stdout

def create_unique_directory(dataset, clustering_method, k_domains, iteration_seed, sample_size):
    """Create a unique directory for saving data."""
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "../data"))  # Adjust relative to /scripts
    folder_name = f"{dataset}_{clustering_method}_k{k_domains}_seed{iteration_seed}_n{sample_size}"
    save_dir = os.path.join(base_dir, folder_name)
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Directory created: {save_dir}")  # Debugging output
    except Exception as e:
        print(f"Failed to create directory {save_dir}: {e}")
    
    return save_dir

def save_domain_labels(domain_labels, iteration_seed, dataset, clustering_method, k_domains, sample_size):
    """Save domain labels in a JSON file within the unique folder."""
    save_dir = create_unique_directory(dataset, clustering_method, k_domains, iteration_seed, sample_size)
    filename = os.path.join(save_dir, "domain_labels.json")
    
    # Convert domain_labels to a list if it's a NumPy array
    if isinstance(domain_labels, np.ndarray):
        domain_labels = domain_labels.tolist()
    
    try:
        with open(filename, "w") as f:
            json.dump({"domain_labels": domain_labels}, f)
        print(f"Saved domain labels to {filename}")
    except Exception as e:
        print(f"Error saving domain labels: {e}")
    
    return save_dir

def load_domain_labels(iteration_seed, dataset, clustering_method, k_domains, sample_size):
    """Load precomputed domain labels from the unique folder."""
    save_dir = create_unique_directory(dataset, clustering_method, k_domains, iteration_seed, sample_size)
    filename = os.path.join(save_dir, "domain_labels.json")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Domain labels file not found: {filename}")
    
    with open(filename, "r") as f:
        data = json.load(f)
    
    print(f"Loaded domain labels from {filename}")
    return data["domain_labels"]

def main():
    args = parse_arguments()

    if args.dataset == "QM9":
        qm9 = load_qm9_smiles()

    for iteration in range(args.bootstrapping):
        # Generate a unique seed for this iteration
        iteration_seed = (args.random_seed ^ (iteration * 0x5DEECE66D)) & 0xFFFFFFFF  # XOR and mask for 32-bit seed

        # Set random seeds for reproducibility
        random.seed(iteration_seed)
        np.random.seed(iteration_seed)
        torch.manual_seed(iteration_seed)

        # Create the unique save directory for this iteration
        save_dir = create_unique_directory(
            dataset=args.dataset,
            clustering_method=args.clustering_method,
            k_domains=args.k_domains,
            iteration_seed=iteration_seed,
            sample_size=args.sample_size
        )

        # Redirect all output to a log file
        redirect_output(save_dir)

        distance_matrix = None

        if not args.analysis_only:
            if args.dataset == "QM9":
                train_smiles = split_qm9_smiles(qm9, args)  # Collect training SMILES
            else:
                train_smiles = load_and_collect_train_smiles(args)

            # Compute molecular descriptors
            molecular_descriptors = compute_molecular_descriptors(train_smiles)

            # Perform clustering to generate domain labels
            if args.determine_cutoff:
                determine_butina_cutoff(distance_matrix, save_dir)
            else:
                domain_labels, distance_matrix = cluster(train_smiles, args, molecular_descriptors)

            # Save domain labels
            save_domain_labels(
                domain_labels,
                iteration_seed,
                args.dataset,
                args.clustering_method,
                args.k_domains,
                args.sample_size
            )

            # Visualize domain representatives
            if not args.clustering_only and not args.determine_cutoff:
                identify_and_visualise_model(
                    domain_labels, train_smiles, distance_matrix,
                    args.dataset, args.clustering_method, args.k_domains,
                    iteration, args.sample_size, save_dir
                )
        else:
            # Load precomputed domain labels
            domain_labels = load_domain_labels(
                iteration_seed, args.dataset, args.clustering_method, args.k_domains, args.sample_size
            )

        if not args.clustering_only and not args.determine_cutoff:
            # Perform molecular descriptor calculations and visualizations
            plot_pca(domain_labels, molecular_descriptors, save_dir)
            plot_tsne(domain_labels, molecular_descriptors, save_dir)
            plot_umap(domain_labels, molecular_descriptors, save_dir)
            descriptor_analysis(domain_labels, molecular_descriptors)
            compare_molecular_descriptors(domain_labels, molecular_descriptors, 5, save_dir)
            if distance_matrix is not None:
                plot_mds(domain_labels, distance_matrix, save_dir)
                plot_tsne_from_matrix(domain_labels, distance_matrix, save_dir)
                plot_umap_from_matrix(domain_labels, distance_matrix, save_dir)

if __name__ == "__main__":
    main()