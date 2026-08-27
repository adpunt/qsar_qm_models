import numpy as np
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap


def forest_aleatoric_epistemic(forest, X_train, y_train, X):
    """Split a fitted sklearn RandomForestRegressor's predictive variance into
    aleatoric (mean within-leaf variance) and epistemic (variance of per-tree means).

    Law of total variance over the mixture of per-tree leaf distributions:
        Var[y|x] = E_t[ Var(y | leaf_t(x)) ]  +  Var_t[ E(y | leaf_t(x)) ]
                   \_______ aleatoric _______/   \______ epistemic ______/

    Returns (mean, aleatoric, epistemic), each shape (n_samples,), in y units^2.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=float).ravel()
    n_train = X_train.shape[0]

    # Which leaf every training point and every query point lands in, per tree.
    train_leaves = forest.apply(X_train)          # (n_train, n_trees)
    test_leaves = forest.apply(X)                 # (n_query, n_trees)

    if forest.bootstrap:
        n_boot = _get_n_samples_bootstrap(n_train, forest.max_samples)

    tree_means = np.empty((test_leaves.shape[0], forest.n_estimators))
    tree_vars = np.empty_like(tree_means)

    for i, est in enumerate(forest.estimators_):
        # Reproduce exactly the rows this tree was grown on (in-bag, with repeats).
        if forest.bootstrap:
            idx = _generate_sample_indices(est.random_state, n_train, n_boot)
        else:
            idx = np.arange(n_train)

        leaf_of = train_leaves[idx, i]
        y_of = y_train[idx]

        n_nodes = est.tree_.node_count
        cnt = np.bincount(leaf_of, minlength=n_nodes).astype(float)
        s1 = np.bincount(leaf_of, weights=y_of, minlength=n_nodes)
        s2 = np.bincount(leaf_of, weights=y_of ** 2, minlength=n_nodes)

        safe = np.where(cnt > 0, cnt, 1.0)
        leaf_mean = s1 / safe
        leaf_var = np.maximum(s2 / safe - leaf_mean ** 2, 0.0)

        tree_means[:, i] = leaf_mean[test_leaves[:, i]]
        tree_vars[:, i] = leaf_var[test_leaves[:, i]]

    mean = tree_means.mean(axis=1)
    aleatoric = tree_vars.mean(axis=1)                    # E_t[Var within leaf]
    epistemic = tree_means.var(axis=1)                    # Var_t[leaf mean]
    return mean, aleatoric, epistemic
