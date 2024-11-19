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
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
# Fix TODO statements before attempting to run, this will be laden with errors
# Goal is to run on a server, it can be done separately from Rust, ideally do this with bootstrapping (need to ensure data is loaded identically)
# 
# TODO: get another source for butina, this one didn't work well at all

from run_qm_qsar_models import load_qm9, split_qm9, load_and_split_polaris

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for clustering a small molecule dataset into domains and analysing those domains.")
    parser.add_argument("-r", "--random-seed", type=int, default=42, help="Random seed (default is 42)")
    parser.add_argument("-n", "--sample-size", type=int, required=True, help="Sample size")
    parser.add_argument("-k", "--k_domains", type=int, default=1, help="Number of domains for clustering (default is 1)")
    parser.add_argument("-m", "--clustering_method", type=str, default="Agglomerative", help="Method to cluster the chemical domain (default is Agglomerative)")
    parser.add_argument("-a", "--analysis-only", type=bool, default=False, help="Only visualise molecules based on existing indices (default is False)")
    parser.add_argument("-c", "--clustering-only", type=bool, default=False, help="Only cluster molecules without analysis, useful for bootstrapping (default is False)")
    parser.add_argument("-d", "--dataset", type=str, default="QM9", help="Dataset to run experiments on (default is QM9)")
    parser.add_argument("-b", "--bootstrapping", type=int, default=1, help="Bootstrapping iterations (default is 1 ie. no bootstrapping)")
    parser.add_argument("--determine-cutoff", type=bool, default=False, help="Repeatedly run Butina with different cutoffs to find an optimal one (default is False)")
    return parser.parse_args()

def compute_molecular_descriptors(dataset):
    """
    Compute molecular descriptors for each molecule in the dataset.
    Returns a list of descriptor dictionaries.
    """
    descriptors = []
    for data in dataset:
        mol = Chem.MolFromSmiles(data.smiles)
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

def compare_molecular_descriptors(domain_labels, molecular_descriptors, m):
    """
    Compare molecular descriptors across domains and visualize the top M descriptors.
    """
    # Create a DataFrame for easier analysis
    df = pd.DataFrame(molecular_descriptors)
    df["Domain"] = domain_labels

    results = {}
    for column in df.columns[:-1]:  # Exclude the "Domain" column
        # Perform Kruskal-Wallis test for differences between domains
        data_by_domain = [df[df["Domain"] == domain][column].dropna() for domain in df["Domain"].unique()]
        stat, p_value = kruskal(*data_by_domain)
        results[column] = p_value

    # Select the top M descriptors with the smallest p-values
    top_descriptors = sorted(results, key=results.get)[:m]

    # Generate boxplots for the top descriptors
    plots = []
    for descriptor in top_descriptors:
        chart = alt.Chart(df).mark_boxplot().encode(
            x="Domain:N",
            y=f"{descriptor}:Q",
            color="Domain:N"
        ).properties(
            title=f"Distribution of {descriptor} by Domain"
        )
        plots.append(chart)

    # Combine all plots into one view
    final_chart = alt.vconcat(*plots)
    final_chart.save("descriptor_comparison.html")
    return final_chart

def plot_pca(domain_labels, molecular_descriptors):
    df = pd.DataFrame(molecular_descriptors)
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(df.drop(columns="Domain").fillna(0))  # Fill NaN with 0

    df_pca = pd.DataFrame(transformed, columns=["PC1", "PC2"])
    df_pca["Domain"] = domain_labels

    chart = alt.Chart(df_pca).mark_circle(size=60).encode(
        x="PC1",
        y="PC2",
        color="Domain:N",
        tooltip=["PC1", "PC2", "Domain"]
    ).interactive()

    chart.save("pca_plot.html")
    return chart

def plot_tsne(domain_labels, molecular_descriptors):
    df = pd.DataFrame(molecular_descriptors)
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

    chart.save("tsne_plot.html")
    return chart

def plot_umap(domain_labels, molecular_descriptors):
    df = pd.DataFrame(molecular_descriptors)
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

    chart.save("umap_plot.html")
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

def identify_and_visualise_model(domain_labels, dataset):
    """
    Identify the centroid molecule for each domain based on ECFP4 Tanimoto similarity.
    """
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data.smiles), radius=2, nBits=2048)
                    for data in dataset]

    representatives = {}
    for domain in set(domain_labels):
        domain_fps = [fp for i, fp in enumerate(fingerprints) if domain_labels[i] == domain]
        centroid = domain_fps[0]  # Pick the first one for simplicity (or calculate mean similarity)
        representatives[domain] = centroid

    # Visualize (e.g., output SMILES of the centroids or save images)
    for domain, rep in representatives.items():
        print(f"Domain {domain}: Representative molecule fingerprint {rep}")
        visualize_molecule(rep, str(domain))

# Perform statistical tests (ANOVA, Kruskal-Wallis) to determine if differences in properties are significant across domains  
def descriptor_analysis(domain_labels, molecular_descriptors):
    return None

def main():
    args = parse_arguments()

    # TODO: don't import load and split since it's meant for writing to mmap files, use a data structure instead
    if args.dataset == "QM9":
        qm9 = load_qm9(args.qm_property)

    for iteration in range(args.bootstrapping):
        iteration_seed = (args.random_seed ^ (iteration * 0x5DEECE66D)) & 0xFFFFFFFF  # XOR and mask for 32-bit seed

        random.seed(iteration_seed)
        np.random.seed(iteration_seed)
        torch.manual_seed(iteration_seed)

        if not args.analysis_only:
            if args.dataset == "QM9":
                train_count, test_count = split_qm9(qm9, args, files)

            else:
                train_count, test_count = load_and_split_polaris(files, args.split)
            if args.determine_cutoff:
                determine_butina_cutoff()
            else:
                domain_labels = cluster(dataset, molecular_descriptors)
        # TODO: save domain labels based on n, iteration_seed, dataset, clustering_method, k_domains (all of that should be in the title)
        else:
            # Load domain label from the appropriate saved location 

        if not args.clustering_only and not args.determine_cutoff:
            molecular_descriptors = compute_molecular_descriptors(domain_labels)
            identify_and_visualise_model(domain_labels)
            plot_pca(domain_labels, molecular_descriptors)
            plot_tsne(domain_labels, molecular_descriptors)
            plot_umap(domain_labels, molecular_descriptors)
            descriptor_analysis(domain_labels, molecular_descriptors)

def determine_butina_cutoff(distance_matrix, min_cutoff=0.01, max_cutoff=0.99, num_cutoffs=5):
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
    chart.save("butina_cutoff_analysis.html")

    # Return the cutoff with the desired number of clusters
    optimal_cutoff = cutoff_df.loc[cutoff_df['ClusterCount'] == max(cluster_counts, key=lambda x: x[1])[1], 'Cutoff']
    return optimal_cutoff.iloc[0] if not optimal_cutoff.empty else None

def tanimoto_distance_matrix(fps):
    """
    Compute the Tanimoto distance matrix for a list of fingerprints.
    """
    n = len(fps)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            distance_matrix[i, j] = 1 - similarity
            distance_matrix[j, i] = 1 - similarity
    return distance_matrix

def agglomerative_clustering(distance_matrix, k_domains):
    """
    Perform Agglomerative Clustering.
    """
    clustering = AgglomerativeClustering(n_clusters=k_domains, affinity='precomputed', linkage='average')
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

def cluster(dataset, args, descriptors):
    """
    Perform clustering using the specified method on the dataset.
    """
    # Extract fingerprints or molecular descriptors
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(data.smiles), radius=2, nBits=2048)
           for data in dataset]

    if args.clustering_method == "Agglomerative":
        distance_matrix = tanimoto_distance_matrix(fps)
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
        distance_matrix = tanimoto_distance_matrix(fps)
        domain_labels = butina_clustering(distance_matrix, cutoff=0.93375)

    else:
        raise ValueError(f"Unsupported clustering method: {args.clustering_method}")

    return domain_labels


if __name__ == "__main__":
    main()