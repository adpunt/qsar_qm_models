#!/usr/bin/env python
"""The split the QM9 graph models actually see (RERUN_PLAN.md §2.10b).

`split_qm9` shuffles with `qm9.index_select(randperm(...))`. PyG returns a NEW
dataset from that call, so the assignment rebinds the local name and nothing
outside the function sees it. Every index the function returns -- and every
molecule it featurises and writes to the mmap -- is a position in the shuffled
order. main() used to keep the indices and drop the shuffled object, then pass
the ORIGINAL dataset to `run_qm9_graph_model`, where the graphs are read as
`qm9[train_idx]`. The graph at row k and the label at row k then belong to two
different molecules, and `qm9[train_idx]` is an arbitrary subset rather than a
held-out-scaffold partition.

Two properties, each failing if its half of the fix is removed:

  1. the dataset `split_qm9` returns is the one its indices index;
  2. main() hands that object, not the original, to the graph models.

Measured before the fix: 0 of 160 training rows had matching SMILES.

Run it directly:  python scripts/test_qm9_split_alignment.py
(Needs data/QM9 and takes about a minute -- it loads the real dataset.)
"""

import os
import sys
import tempfile
import traceback
import types
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import process_and_train as P  # noqa: E402

SAMPLE_SIZE = 200


def _recording_writer(store):
    """Stand in for write_to_mmap and remember what each row held."""

    # Mirrors write_to_mmap, which lost `randomized_smiles` and `max_vocab` on
    # 2026-08-28 when one-hot SMILES and randomized SMILES were deleted.
    def write(smiles_isomeric, smiles_canonical, pdv,
              chemberta, mhggnn, avalon, y, category, files,
              reps, k_domains, sns_fp, ecfp4=None):
        store.append((category, smiles_isomeric, float(y)))

    return write


def _agreement(written, dataset, idx):
    """How many of the rows written as `train` sit at dataset[idx] in order."""
    pairs = [(s, y) for c, s, y in written if c == "train"]
    n = min(len(pairs), len(idx))
    assert n > 0, "no training rows were written"
    smiles_ok = sum(1 for k in range(n)
                    if pairs[k][0] == dataset[idx[k]].smiles)
    label_ok = sum(1 for k in range(n)
                   if abs(pairs[k][1] - float(dataset[idx[k]].y.item())) < 1e-5)
    return smiles_ok, label_ok, n


def the_returned_dataset_is_the_one_the_indices_index():
    written = []
    real_writer = P.write_to_mmap
    P.write_to_mmap = _recording_writer(written)
    try:
        args = types.SimpleNamespace(
            split="scaffold", sample_size=SAMPLE_SIZE,
            molecular_representations=[], k_domains=1,
            logging=False, target="homo_lumo_gap",
        )
        torch.manual_seed(0)
        dataset = P.load_qm9(args.target)
        returned = P.split_qm9(dataset, args, {})
        assert len(returned) == 5, (
            f"split_qm9 returned {len(returned)} values, not the shuffled "
            f"dataset plus four -- the caller cannot reach the object its "
            f"indices belong to")
        split_dataset, train_idx, test_idx, val_idx, groups = returned

        smiles_ok, label_ok, n = _agreement(written, split_dataset, train_idx)
        assert smiles_ok == n, (
            f"{n - smiles_ok} of {n} training rows carry a different "
            f"molecule's graph than the label written at that row")
        assert label_ok == n, (
            f"{n - label_ok} of {n} training rows carry a different label")
        print(f"    {n} training rows, graph and label agree on all of them")

        # And the original object is NOT interchangeable with it -- otherwise
        # this test would pass for the wrong reason on some future dataset.
        wrong_smiles, _, _ = _agreement(written, dataset, train_idx)
        assert wrong_smiles < n, (
            "indexing the unshuffled dataset gives the same molecules, so this "
            "check cannot tell the two apart")
        print(f"    indexing the unshuffled dataset instead: "
              f"{wrong_smiles}/{n} agree")
    finally:
        P.write_to_mmap = real_writer


def main_passes_the_split_dataset_to_the_graph_models():
    written = []
    seen = {}
    real_writer, real_run = P.write_to_mmap, P.process_and_run
    real_argv, real_cwd = sys.argv, os.getcwd()

    def capture(args, iteration, iteration_seed, file_no, train_idx, test_idx,
                val_idx, target_domain, env, rust_executable_path, files, s,
                dataset, scaffold_groups=None):
        seen["dataset"] = dataset
        seen["train_idx"] = train_idx
        return []

    P.write_to_mmap = _recording_writer(written)
    P.process_and_run = capture
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            sys.argv = [
                "process_and_train.py", "-d", "QM9", "-m", "gin",
                "-r", "graph", "-n", str(SAMPLE_SIZE),
                "--noise-level", "0.0", "-b", "1", "-s", "scaffold",
            ]
            P.main()

        assert "dataset" in seen, "process_and_run was never reached"
        smiles_ok, label_ok, n = _agreement(
            written, seen["dataset"], seen["train_idx"])
        assert smiles_ok == n, (
            f"main() passed a dataset in which {n - smiles_ok} of {n} training "
            f"rows carry another molecule's graph -- the shuffled object was "
            f"dropped again")
        assert label_ok == n, (
            f"main() passed a dataset in which {n - label_ok} of {n} training "
            f"rows carry another molecule's label")
        print(f"    {n} training rows reach the graph models correctly paired")
    finally:
        P.write_to_mmap, P.process_and_run = real_writer, real_run
        sys.argv = real_argv
        os.chdir(real_cwd)


def the_scaffold_split_holds_out_whole_groups_and_keeps_the_acyclic_ones():
    """DeepChem's splitter put every acyclic molecule in one pseudo-group.

    `MurckoScaffoldSmiles` returns the EMPTY STRING for an acyclic molecule and
    DeepChem does not special-case it, so they all shared one group -- and its
    splitter fills training from the largest groups first, so that group went
    entirely into training. Measured on the first 2,000 QM9 molecules: 851 of
    them (42.5%) are acyclic, and they were 53.2% of training and 0.0% of
    validation and of test (RERUN_PLAN.md §2.13).
    """
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    import numpy as np

    import process_and_train as P

    dataset = P.load_qm9("homo_lumo_gap")
    n = 2000
    smiles = [dataset[i].smiles for i in range(n)]

    def core(i):
        try:
            c = MurckoScaffoldSmiles(smiles=smiles[i], includeChirality=False)
        except Exception:  # noqa: BLE001
            c = None
        return c if c else f"__acyclic__{i}"

    def acyclic(i):
        return core(i).startswith("__acyclic__")

    np.random.seed(42)
    train, val, test = P.scaffold_split_indices(smiles)
    assert len(train) + len(val) + len(test) == n, (len(train), len(val), len(test))

    shared = {core(i) for i in train} & {core(i) for i in test}
    assert not shared, f"{len(shared)} scaffold group(s) are in both train and test"

    population = sum(1 for i in range(n) if acyclic(i)) / n
    parts = {name: sum(1 for i in idx if acyclic(i)) / len(idx)
             for name, idx in (("train", train), ("val", val), ("test", test))}
    assert population > 0.2, (
        f"only {population:.1%} of this sample is acyclic, so the check has "
        f"nothing to catch")

    # Each split's acyclic share must sit NEAR the population's, not merely be
    # non-zero. A floor alone is too weak: it passes both ways this can go wrong.
    # DeepChem's largest-first ordering gave train 53.2% / val 0.0% / test 0.0%;
    # largest-first WITH acyclic singletons gives val 100% and test 82%. Both are
    # a swept split, and a `> 10%` floor catches only the first.
    band = 0.25
    for name, frac in parts.items():
        assert abs(frac - population) <= band, (
            f"the {name} split is {frac:.1%} acyclic against a population of "
            f"{population:.1%} -- more than {band:.0%} away, so the acyclic "
            f"molecules are being swept into one split")
    print(f"    {population:.1%} of the sample is acyclic; split fractions "
          f"train {parts['train']:.3f}, val {parts['val']:.3f}, "
          f"test {parts['test']:.3f}; 0 groups shared between train and test")


def check(name, fn):
    print(f"  {name}")
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        print(f"    FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False
    print("    ok")
    return True


def main():
    print("QM9 split alignment (RERUN_PLAN.md §2.10b)")
    results = [
        check("split_qm9 returns the dataset its indices index",
              the_returned_dataset_is_the_one_the_indices_index),
        check("main passes that dataset to the graph models",
              main_passes_the_split_dataset_to_the_graph_models),
        check("the scaffold split holds out whole groups and keeps the acyclic "
              "molecules in every split",
              the_scaffold_split_holds_out_whole_groups_and_keeps_the_acyclic_ones),
    ]
    if not all(results):
        print("\nFAIL: the QM9 graph models are trained on mismatched molecules")
        return 1
    print("\nOK: graph, label and split agree for every QM9 graph-model row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
