import numpy as np
from scipy.spatial.distance import cdist, mahalanobis
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import AgglomerativeClustering

def compute_molecular_distances(x_data, rep_type, method='tanimoto', subset_indices=None):
    """
    Compute pairwise molecular distances.
    
    Args:
        x_data: molecular representations (fingerprints, descriptors, etc.)
        rep_type: type of representation ('ecfp4', 'sns', 'pdv', etc.)
        method: distance metric to use
        subset_indices: if provided, only compute distances for these samples
    
    Returns:
        distance_matrix: pairwise distances (n_samples x n_samples or subset x subset)
    """
    import numpy as np
    from scipy.spatial.distance import cdist
    from sklearn.metrics import pairwise_distances
    
    if subset_indices is not None:
        x_data = x_data[subset_indices]
    
    n_samples = len(x_data)
    
    if method == 'tanimoto':
        # Tanimoto/Jaccard for binary fingerprints
        # Distance = 1 - (intersection / union)
        # For binary vectors: 1 - (a·b) / (a·a + b·b - a·b)
        
        if rep_type in ['ecfp4', 'sns']:
            # Binary fingerprints
            x_binary = (x_data > 0).astype(np.float32)
            
            # Compute intersections (dot products)
            intersections = x_binary @ x_binary.T
            
            # Compute unions
            sums = x_binary.sum(axis=1, keepdims=True)
            unions = sums + sums.T - intersections
            
            # Tanimoto similarity
            similarity = intersections / (unions + 1e-8)
            
            # Convert to distance
            distance_matrix = 1 - similarity
        else:
            # For non-binary, use standard Jaccard
            distance_matrix = pairwise_distances(x_data, metric='jaccard')
    
    elif method == 'euclidean':
        # Standard Euclidean distance
        distance_matrix = pairwise_distances(x_data, metric='euclidean')
    
    elif method == 'cosine':
        # Cosine distance (1 - cosine similarity)
        distance_matrix = pairwise_distances(x_data, metric='cosine')
    
    elif method == 'mahalanobis':
        # Mahalanobis distance (requires covariance matrix)
        from scipy.spatial.distance import mahalanobis
        
        # Compute covariance
        mean = x_data.mean(axis=0)
        cov = np.cov(x_data.T)
        
        # Add regularization to avoid singular matrix
        cov += np.eye(cov.shape[0]) * 1e-6
        
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            print("Warning: Singular covariance matrix, using euclidean instead")
            return pairwise_distances(x_data, metric='euclidean')
        
        # Compute pairwise Mahalanobis distances
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = mahalanobis(x_data[i], x_data[j], cov_inv)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    
    elif method == 'mmd':
        # Maximum Mean Discrepancy (kernel-based)
        # This compares distributions, not individual samples
        # We'll compute kernel matrix instead
        
        from sklearn.metrics.pairwise import rbf_kernel
        
        # Use RBF kernel
        gamma = 1.0 / x_data.shape[1]  # Rule of thumb
        kernel_matrix = rbf_kernel(x_data, gamma=gamma)
        
        # Convert kernel similarity to distance
        # d(x,y) = sqrt(k(x,x) + k(y,y) - 2*k(x,y))
        diag = np.diag(kernel_matrix)
        distance_matrix = np.sqrt(
            diag[:, None] + diag[None, :] - 2 * kernel_matrix
        )
    
    elif method == 'optimal_transport':
        # Simplified optimal transport (Wasserstein)
        # For individual molecules, we approximate with weighted distance
        
        print("Warning: Full optimal transport is expensive. Using approximation.")
        # Use earth mover's distance on feature histograms
        from scipy.stats import wasserstein_distance
        
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # Treat features as distributions
                dist = wasserstein_distance(x_data[i], x_data[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
    
    else:
        raise ValueError(f"Unknown distance method: {method}")
    
    return distance_matrix


def identify_outliers_by_distance(x_data, rep_type, method='tanimoto', 
                                   threshold_percentile=90):
    """
    Identify outlier samples based on distance to neighbors.
    
    Samples that are far from all other samples are likely outliers.
    
    Args:
        x_data: molecular representations
        rep_type: representation type
        method: distance metric
        threshold_percentile: samples above this percentile of avg distance are outliers
    
    Returns:
        outlier_mask: boolean array (True = outlier)
        avg_distances: average distance to all other samples
    """
    import numpy as np
    
    # Compute distance matrix
    distance_matrix = compute_molecular_distances(x_data, rep_type, method)
    
    # Average distance to all other samples (excluding self)
    n_samples = len(x_data)
    avg_distances = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Exclude distance to self
        distances_i = np.concatenate([distance_matrix[i, :i], distance_matrix[i, i+1:]])
        avg_distances[i] = distances_i.mean()
    
    # Identify outliers as samples with high average distance
    threshold = np.percentile(avg_distances, threshold_percentile)
    outlier_mask = avg_distances > threshold
    
    print(f"Identified {outlier_mask.sum()}/{n_samples} outliers by distance")
    print(f"Distance threshold: {threshold:.4f}")
    
    return outlier_mask, avg_distances


def cluster_by_distance(x_data, rep_type, method='tanimoto', n_clusters=5):
    """
    Cluster molecules by distance.
    
    Args:
        x_data: molecular representations
        rep_type: representation type
        method: distance metric
        n_clusters: number of clusters
    
    Returns:
        cluster_labels: cluster assignment for each sample
    """
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    
    # Compute distance matrix
    distance_matrix = compute_molecular_distances(x_data, rep_type, method)
    
    # Cluster using distance matrix
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    print(f"Clustered {len(x_data)} samples into {n_clusters} clusters")
    for i in range(n_clusters):
        count = (cluster_labels == i).sum()
        print(f"  Cluster {i}: {count} samples")
    
    return cluster_labels


def distance_weighted_sample_selection(
    x_data, losses, uncertainties, rep_type,
    method='tanimoto', keep_fraction=0.8
):
    """
    Select clean samples using loss + uncertainty + distance.
    
    Strategy: Keep samples that have:
    - Low loss
    - Low uncertainty
    - Are close to other low-loss samples (not isolated outliers)
    
    Args:
        x_data: molecular representations
        losses: per-sample losses
        uncertainties: per-sample uncertainties (or None)
        rep_type: representation type
        method: distance metric
        keep_fraction: fraction of samples to keep
    
    Returns:
        keep_mask: boolean array (True = keep sample)
        scores: combined score for each sample
    """
    import numpy as np
    
    n_samples = len(x_data)
    
    # Normalize losses and uncertainties
    loss_norm = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
    
    if uncertainties is not None:
        unc_norm = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-8)
    else:
        unc_norm = np.zeros_like(loss_norm)
    
    # Identify potential clean samples (low loss + low uncertainty)
    clean_score = loss_norm + unc_norm
    clean_threshold = np.percentile(clean_score, 50)  # Bottom 50%
    potential_clean = clean_score < clean_threshold
    
    # Compute distances among potential clean samples
    clean_indices = np.where(potential_clean)[0]
    
    if len(clean_indices) < 10:
        # Too few clean samples, just use loss-based selection
        print("Warning: Too few clean samples for distance-based selection. Using loss only.")
        n_keep = int(n_samples * keep_fraction)
        keep_indices = np.argsort(losses)[:n_keep]
        keep_mask = np.zeros(n_samples, dtype=bool)
        keep_mask[keep_indices] = True
        return keep_mask, losses
    
    # Compute distance matrix for potential clean samples
    distance_matrix = compute_molecular_distances(
        x_data, rep_type, method, subset_indices=clean_indices
    )
    
    # For each potential clean sample, compute average distance to other clean samples
    avg_distances_to_clean = np.zeros(len(clean_indices))
    for i in range(len(clean_indices)):
        distances_i = np.concatenate([
            distance_matrix[i, :i],
            distance_matrix[i, i+1:]
        ])
        avg_distances_to_clean[i] = distances_i.mean()
    
    # Distance score: samples close to other clean samples get lower score (better)
    dist_norm = (avg_distances_to_clean - avg_distances_to_clean.min()) / \
                (avg_distances_to_clean.max() - avg_distances_to_clean.min() + 1e-8)
    
    # Combined score for clean samples
    clean_combined_score = clean_score[clean_indices] + dist_norm
    
    # For non-clean samples, give them high score (will be filtered out)
    combined_score = np.ones(n_samples) * clean_combined_score.max() * 2
    combined_score[clean_indices] = clean_combined_score
    
    # Keep top fraction by combined score
    n_keep = int(n_samples * keep_fraction)
    keep_indices = np.argsort(combined_score)[:n_keep]
    
    keep_mask = np.zeros(n_samples, dtype=bool)
    keep_mask[keep_indices] = True
    
    print(f"Distance-weighted selection: keeping {n_keep}/{n_samples} samples")
    print(f"  {(keep_mask & potential_clean).sum()} from clean cluster")
    print(f"  {(keep_mask & ~potential_clean).sum()} from uncertain region")
    
    return keep_mask, combined_score