# DBSCAN
# Density-Based Spatial Clustering of Applications with Noise
# Groups data points into clusters by identifying regions of high density 
# Epsilon - defines the maximum distance a point can be from another to be considered in the same neighbourhood 
# MinPoints - minimum number of points required to form a dense region
# Points that do not belong to any cluster are considered noise
# Clusters are built by connecting core points (at least MinPoints with epsilon) with their neighbourhoods
# Application of anomaly detection
# Modification on DBSCAN to smooth noise
# TODO: make logic for this
# Regression: 
# Classification:

def dbscan(dataset):
    return None

def hdbscan_glosh(dataset):
    return None

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for clustering a small molecule dataset into domains and analysing those domains.")
    parser.add_argument("-f", "--dataset-file", type=int, default=42, help="Random seed (default is 42)")
    parser.add_argument("-c", "--classification", type=int, required=True, help="Sample size")
    parser.add_argument("-m", "--method", type=int, default=1, help="Number of domains for clustering (default is 1)")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # TODO: actually do this:
    dataset = open(args.dataset_file)

    if args.method == "DBSCAN":
        modificed_datset = dbscan(dataset)

    if args.method == "HDBSCAN":
        modificed_datset = hdbscan_glosh(dataset)

if __name__ == "__main__":
    main()