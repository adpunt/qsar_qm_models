import numpy as np
import gc
from sklearn.cluster import MiniBatchKMeans
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
import struct
import warnings

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")


def butina_clustering(fingerprints, k_domains, cutoff=0.35):
    """
    Butina clustering using RDKit's clustering algorithm.
    
    Args:
        fingerprints: List or array of binary fingerprints
        k_domains: Number of desired clusters
        cutoff: Tanimoto distance cutoff (default 0.35)
    
    Returns:
        np.array: Cluster labels for each fingerprint
    """
    from rdkit.ML.Cluster import Butina
    
    print(f"  Running Butina clustering with cutoff={cutoff}...")
    
    # Convert fingerprints to ExplicitBitVect if needed
    n_samples = len(fingerprints)
    fp_list = []
    
    for fp in fingerprints:
        if isinstance(fp, np.ndarray):
            # Convert numpy array to ExplicitBitVect
            bitvect = DataStructs.ExplicitBitVect(len(fp))
            for i, bit in enumerate(fp):
                if bit:
                    bitvect.SetBit(i)
            fp_list.append(bitvect)
        else:
            fp_list.append(fp)
    
    # Calculate distance matrix (Tanimoto distance)
    dists = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            sim = DataStructs.TanimotoSimilarity(fp_list[i], fp_list[j])
            dist = 1.0 - sim
            dists.append(dist)
    
    # Perform Butina clustering
    clusters = Butina.ClusterData(dists, n_samples, cutoff, isDistData=True)
    
    # Convert cluster assignments to labels
    labels = np.zeros(n_samples, dtype=np.uint8)
    for cluster_id, cluster_members in enumerate(clusters):
        for member_idx in cluster_members:
            labels[member_idx] = cluster_id % k_domains  # Ensure we stay within k_domains
    
    print(f"  Butina clustering created {len(clusters)} clusters, mapped to {k_domains} domains")
    return labels


def splito_clustering(smiles_list, k_domains):
    """
    Splito clustering - splits molecules based on chemical diversity using ECFP4 fingerprints
    and maximizes the diversity within each domain.
    
    This implements a simple version of diversity-based splitting where molecules are
    iteratively assigned to domains to maximize intra-domain diversity.
    
    Args:
        smiles_list: List of SMILES strings
        k_domains: Number of desired domains
    
    Returns:
        np.array: Domain labels for each molecule
    """
    print(f"  Running Splito clustering for {len(smiles_list)} molecules into {k_domains} domains...")
    
    # Generate ECFP4 fingerprints for all molecules
    fps = []
    valid_indices = []
    
    for idx, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(fp)
                valid_indices.append(idx)
        except:
            pass
    
    n_valid = len(fps)
    if n_valid == 0:
        print("  Warning: No valid molecules for Splito clustering, using random assignment")
        return np.random.randint(0, k_domains, size=len(smiles_list), dtype=np.uint8)
    
    # Initialize domain assignments
    labels = np.zeros(len(smiles_list), dtype=np.uint8)
    domain_members = [[] for _ in range(k_domains)]
    
    # Calculate all pairwise similarities
    print("  Calculating pairwise similarities...")
    similarity_matrix = np.zeros((n_valid, n_valid))
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    
    # Start by assigning the most dissimilar molecules to different domains
    assigned = set()
    
    # Find k_domains most dissimilar molecules as seeds
    if n_valid >= k_domains:
        # Start with the first molecule
        current_idx = 0
        domain_members[0].append(valid_indices[current_idx])
        assigned.add(current_idx)
        
        # For each subsequent domain, find the molecule most dissimilar to all previously selected
        for domain_id in range(1, k_domains):
            max_min_dist = -1
            best_idx = -1
            
            for candidate_idx in range(n_valid):
                if candidate_idx in assigned:
                    continue
                
                # Find minimum similarity to already assigned molecules
                min_sim = min(similarity_matrix[candidate_idx, assigned_idx] 
                            for assigned_idx in assigned)
                
                if min_sim > max_min_dist:
                    max_min_dist = min_sim
                    best_idx = candidate_idx
            
            if best_idx != -1:
                domain_members[domain_id].append(valid_indices[best_idx])
                assigned.add(best_idx)
    
    # Assign remaining molecules to domains to maximize diversity within each domain
    print("  Assigning remaining molecules...")
    unassigned = [i for i in range(n_valid) if i not in assigned]
    
    for idx in unassigned:
        # For each domain, calculate average similarity to molecules in that domain
        domain_avg_sims = []
        
        for domain_id in range(k_domains):
            if len(domain_members[domain_id]) == 0:
                domain_avg_sims.append(0.0)
            else:
                domain_indices = [valid_indices.index(vi) if vi in valid_indices else -1 
                                for vi in domain_members[domain_id]]
                domain_indices = [di for di in domain_indices if di != -1]
                
                if len(domain_indices) > 0:
                    avg_sim = np.mean([similarity_matrix[idx, di] for di in domain_indices])
                    domain_avg_sims.append(avg_sim)
                else:
                    domain_avg_sims.append(0.0)
        
        # Assign to domain with lowest average similarity (most diverse)
        best_domain = np.argmin(domain_avg_sims)
        domain_members[best_domain].append(valid_indices[idx])
    
    # Create final label array
    for domain_id, members in enumerate(domain_members):
        for member_idx in members:
            labels[member_idx] = domain_id
    
    # Handle invalid molecules (assign to random domain)
    invalid_indices = set(range(len(smiles_list))) - set(valid_indices)
    for idx in invalid_indices:
        labels[idx] = np.random.randint(0, k_domains)
    
    print(f"  Splito clustering complete. Domain sizes: {[len(m) for m in domain_members]}")
    return labels


def extract_smiles_from_mmap(mmap_file, entry_count, molecular_representations, logging=False):
    """
    Extract SMILES strings from mmap file for domain methods that need them.
    
    Args:
        mmap_file: Opened memory-mapped file
        entry_count: Number of entries to read
        molecular_representations: List of representations available
        logging: Enable debug logging
    
    Returns:
        list: List of canonical SMILES strings
    """
    smiles_list = []
    mmap_file.seek(0)  # Reset to beginning
    
    for entry_idx in range(entry_count):
        try:
            # Read isomeric SMILES
            iso_len = struct.unpack("I", mmap_file.read(4))[0]
            mmap_file.read(iso_len)
            
            # Read canonical SMILES
            canon_len = struct.unpack("I", mmap_file.read(4))[0]
            canonical_smiles = mmap_file.read(canon_len).decode("utf-8")
            smiles_list.append(canonical_smiles)
            
            # Skip the rest of the entry
            mmap_file.read(4)  # target value
            
            if "randomized_smiles" in molecular_representations:
                rand_len = struct.unpack("I", mmap_file.read(4))[0]
                if rand_len > 0:
                    mmap_file.read(rand_len)
            
            if "sns" in molecular_representations:
                mmap_file.read(128)
            
            if "pdv" in molecular_representations:
                mmap_file.read(25)
            
            mmap_file.read(4)  # processed target
            
            # Skip remaining fields
            if "smiles" in molecular_representations:
                ohe_len = struct.unpack("I", mmap_file.read(4))[0]
                mmap_file.read(ohe_len)
            
            if "randomized_smiles" in molecular_representations:
                ohe_len = struct.unpack("I", mmap_file.read(4))[0]
                mmap_file.read(ohe_len)
            
            if "ecfp4" in molecular_representations:
                mmap_file.read(256)
                
        except Exception as e:
            if logging:
                print(f"  Warning: Error reading SMILES at entry {entry_idx}: {e}")
            smiles_list.append("")  # Add empty string for failed entries
            
    return smiles_list


def extract_and_cluster_for_domains(args, file_no, train_idx, test_idx, val_idx, parse_mmap):
    """
    Extract domain representation from mmap, cluster, return domain labels.
    Cleans up intermediate data structures.
    
    Args:
        args: Parsed arguments with domain_method, domain_representation, k_domains
        file_no: File identifier for mmap files
        train_idx, test_idx, val_idx: Index lists (for size only)
        parse_mmap: The parse_mmap function from the main module
    
    Returns:
        dict or None: {'train': np.array, 'test': np.array, 'val': np.array} of domain labels (uint8)
                      Returns None if k_domains=1 or domain_method='none'
    """
    # Early exit for no clustering
    if args.k_domains == 1 or args.domain_method == 'none':
        print("Skipping domain clustering (k_domains=1 or domain_method='none')")
        return None
    
    print(f"\n{'='*60}")
    print(f"Starting domain extraction and clustering")
    print(f"Method: {args.domain_method}")
    print(f"Representation: {args.domain_representation}")
    print(f"Number of domains: {args.k_domains}")
    print(f"{'='*60}\n")
    
    try:
        # Open memory-mapped files in read-only mode
        train_file = open(f'train_{file_no}.mmap', 'rb')
        test_file = open(f'test_{file_no}.mmap', 'rb')
        val_file = open(f'val_{file_no}.mmap', 'rb')
        
        files = {
            'train': train_file,
            'test': test_file,
            'val': val_file
        }
        
        counts = {
            'train': len(train_idx),
            'test': len(test_idx),
            'val': len(val_idx)
        }
        
        # Handle special cases that need SMILES
        if args.domain_method in ['splito', 'scaffold', 'molecular_weight']:
            print("Extracting SMILES strings for domain method...")
            
            all_smiles = {}
            for split_name, file_handle in files.items():
                file_handle.seek(0)
                smiles_list = extract_smiles_from_mmap(
                    file_handle, 
                    counts[split_name], 
                    args.molecular_representations,
                    args.logging
                )
                all_smiles[split_name] = smiles_list
                print(f"  Extracted {len(smiles_list)} SMILES from {split_name} set")
            
            # Combine all SMILES for clustering
            combined_smiles = (all_smiles['train'] + all_smiles['test'] + all_smiles['val'])
            
            if args.domain_method == 'splito':
                # Run splito clustering on combined data
                all_labels = splito_clustering(combined_smiles, args.k_domains)
                
            elif args.domain_method == 'scaffold':
                print("Scaffold-based domain splitting not yet implemented in this function")
                print("Falling back to random assignment")
                all_labels = np.random.randint(0, args.k_domains, 
                                             size=len(combined_smiles), 
                                             dtype=np.uint8)
            
            elif args.domain_method == 'molecular_weight':
                print("Molecular weight-based domain splitting not yet implemented")
                print("Falling back to random assignment")
                all_labels = np.random.randint(0, args.k_domains, 
                                             size=len(combined_smiles), 
                                             dtype=np.uint8)
            
            # Split labels back into train/test/val
            train_size = len(train_idx)
            test_size = len(test_idx)
            
            domain_labels = {
                'train': all_labels[:train_size],
                'test': all_labels[train_size:train_size + test_size],
                'val': all_labels[train_size + test_size:]
            }
            
            # Clean up
            del all_smiles, combined_smiles, all_labels
            
        else:
            # Extract representations using parse_mmap
            print("Extracting molecular representations...")
            representations = {}
            
            for split_name, file_handle in files.items():
                file_handle.seek(0)  # Reset file pointer
                
                print(f"  Processing {split_name} set ({counts[split_name]} entries)...")
                x_data, y_data, y_data_original = parse_mmap(
                    file_handle,
                    counts[split_name],
                    args.domain_representation,
                    args.molecular_representations,
                    args.k_domains,
                    0,  # sigma=0 for domain extraction
                    logging=args.logging
                )
                
                representations[split_name] = x_data
                print(f"  Extracted {split_name}: shape {x_data.shape}")
                
                # Clean up y_data as we don't need it
                del y_data, y_data_original
            
            # Combine all representations for clustering
            combined_data = np.vstack([
                representations['train'],
                representations['test'],
                representations['val']
            ])
            print(f"\nCombined data shape: {combined_data.shape}")
            
            # Perform clustering based on method
            if args.domain_method == 'random':
                print("\nPerforming random domain assignment...")
                all_labels = np.random.randint(0, args.k_domains, 
                                             size=combined_data.shape[0], 
                                             dtype=np.uint8)
                print(f"Random assignment complete")
                
            elif args.domain_method == 'fingerprint_kmeans':
                if args.domain_representation not in ['ecfp4', 'sns']:
                    raise ValueError(f"fingerprint_kmeans requires ecfp4 or sns representation, got {args.domain_representation}")
                
                print(f"\nPerforming MiniBatchKMeans clustering on fingerprints...")
                kmeans = MiniBatchKMeans(
                    n_clusters=args.k_domains,
                    random_state=args.random_seed,
                    batch_size=1000,
                    max_iter=100,
                    n_init=10
                )
                
                all_labels = kmeans.fit_predict(combined_data).astype(np.uint8)
                print(f"KMeans clustering complete")
                print(f"  Cluster sizes: {np.bincount(all_labels)}")
                
                del kmeans
                
            elif args.domain_method == 'descriptor':
                if args.domain_representation != 'pdv':
                    raise ValueError(f"descriptor method requires pdv representation, got {args.domain_representation}")
                
                print(f"\nPerforming MiniBatchKMeans clustering on descriptors...")
                kmeans = MiniBatchKMeans(
                    n_clusters=args.k_domains,
                    random_state=args.random_seed,
                    batch_size=1000,
                    max_iter=100,
                    n_init=10
                )
                
                all_labels = kmeans.fit_predict(combined_data).astype(np.uint8)
                print(f"KMeans clustering complete")
                print(f"  Cluster sizes: {np.bincount(all_labels)}")
                
                del kmeans
                
            elif args.domain_method == 'butina':
                if args.domain_representation not in ['ecfp4', 'sns']:
                    raise ValueError(f"butina requires ecfp4 or sns representation, got {args.domain_representation}")
                
                # Butina clustering
                all_labels = butina_clustering(combined_data, args.k_domains)
                
            else:
                raise ValueError(f"Unknown domain_method: {args.domain_method}")
            
            # Split labels back into train/test/val
            train_size = len(train_idx)
            test_size = len(test_idx)
            
            domain_labels = {
                'train': all_labels[:train_size],
                'test': all_labels[train_size:train_size + test_size],
                'val': all_labels[train_size + test_size:]
            }
            
            # Clean up representations
            del representations, combined_data, all_labels
        
        # Print statistics
        print(f"\n{'='*60}")
        print("Domain assignment complete!")
        for split_name in ['train', 'test', 'val']:
            labels = domain_labels[split_name]
            counts_per_domain = np.bincount(labels, minlength=args.k_domains)
            print(f"{split_name.capitalize()} domain distribution: {counts_per_domain}")
        print(f"{'='*60}\n")
        
        # Close files
        for file_handle in files.values():
            file_handle.close()
        
        # Explicit garbage collection
        gc.collect()
        
        return domain_labels
        
    except Exception as e:
        print(f"\nError in extract_and_cluster_for_domains: {e}")
        import traceback
        traceback.print_exc()
        
        # Attempt cleanup
        try:
            for file_handle in files.values():
                file_handle.close()
        except:
            pass
        
        gc.collect()
        
        # Return None or raise depending on preference
        raise


# Example usage within your main code:
"""
# After creating train/test/val mmap files and before running models:

domain_labels = extract_and_cluster_for_domains(
    args=args,
    file_no=file_no,
    train_idx=train_idx,
    test_idx=test_idx,
    val_idx=val_idx,
    parse_mmap=parse_mmap  # Pass the function reference
)

if domain_labels is not None:
    # Use domain_labels['train'], domain_labels['test'], domain_labels['val']
    # to write back to mmap files or pass to models
    print(f"Domain labels generated successfully")
else:
    print("No domain clustering performed")
"""