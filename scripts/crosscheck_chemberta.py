#!/usr/bin/env python3
"""Prove that the two pipelines read SMILES for ChemBERTa the same way, and that
the way they read it is the way the published model was pretrained.

WHY THIS FILE EXISTS
--------------------
Encode a molecule with ChemBERTa-77M-MTR and decode it back and it returns a
DIFFERENT molecule. Chlorobenzene comes back as toluene. Both alanine
enantiomers come back as the same achiral string. A quaternary ammonium comes
back neutral. That is real and it is measured, not read.

The cause is in the checkpoint. Its `merges.txt` holds nothing but a version
header, so the byte-pair reader never merges anything, and the 543
multi-character chemical entries that DO sit in its vocabulary -- Cl, Br,
[C@H], [C@@H], [N+], [O-] -- are unreachable. The reader falls back to single
characters and the characters l, r, [, ], +, @ and H have no entry at all.

The obvious conclusion is that the checkpoint is mispackaged and wants the
atom-level SMILES reader DeepChem ships for this model family. THAT CONCLUSION
IS WRONG, and gate 2 below is what stops anyone acting on it. The model was
PRETRAINED through the same character-level fallback. Its own weights say so:
of the 591 vocabulary entries, exactly 28 single characters plus [CLS] and
[SEP] carry trained embeddings, and every one of the 543 multi-character
chemical tokens still sits at its random initialisation, indistinguishable from
the ten [unused] rows that cannot have been trained by construction. Feeding
the model `Cl` as one token hands it a vector it has never seen.

Measured 2026-08-29 on 648 hERG molecules, predicting molecular weight through
the checkpoint's own regression head and comparing against RDKit:

    reader                                  R2 vs true MolWt    mean error
    byte-level, drops unknown characters          0.9921          7.4 Da
    byte-level, substitutes [UNK]                 0.0925         64.2 Da
    atom-level, DeepChem's SMILES regex           0.9738         13.3 Da

So the reader that drops characters is the one the model was trained with, and
it is the one both pipelines must use. The [UNK] variant is catastrophic
because [UNK] is ALSO untrained -- substituting it injects a random vector at
every halogen, bracket and charge.

WHAT THIS COSTS, and it is a real cost, not a fixed bug: the encoder cannot
tell chlorobenzene from toluene. Two different molecules can receive one
vector. Gate 4 counts how often that happens on the real data so the number is
on the record rather than assumed. It counts on the two datasets the study
trains ChemBERTa on and writes results/chemberta_collisions.json and .csv, so a
figure script reads the rate rather than a document quoting it.
`load_collision_counts()` below is the reader for that file.

TWO RATES PER DATASET, because they differ by more than an order of magnitude
on QM9 and only one of them is what a fit experiences. The pool rate is over
every molecule the dataset holds. The per-run rate is over the molecules one job
carries -- a QM9 job draws 10,000 rows from a pool of 129,428, and two molecules
share a vector inside that fit only if both are drawn. hERG carries its whole
set, so its two rates are one number.

Run:  python scripts/crosscheck_chemberta.py
      python scripts/crosscheck_chemberta.py --gates 4     # the count alone
"""

import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# The one model id, and the one reader class, that both pipelines must agree on.
# `AutoTokenizer` resolves to this; naming the class here is what lets the gate
# state which reader it checked rather than trusting the resolution silently.
MODEL_ID = "DeepChem/ChemBERTa-77M-MTR"
READER_CLASS = "RobertaTokenizerFast"

# Molecules the failure is visible on. Each pair differs by exactly the
# chemistry the reader cannot see.
PROBES = [
    ("c1ccccc1Cl", "chlorobenzene"),
    ("Cc1ccccc1", "toluene"),
    ("c1ccccc1Br", "bromobenzene"),
    ("C[C@H](N)C(=O)O", "L-alanine"),
    ("C[C@@H](N)C(=O)O", "D-alanine"),
    ("CC(N)C(=O)O", "alanine, no stereochemistry"),
    ("C[N+](C)(C)C", "tetramethylammonium"),
    ("CC(=O)[O-]", "acetate"),
    ("CC(=O)O", "acetic acid"),
]


def _snapshot():
    """The local checkpoint directory, so the gates read the real files."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.txt", "*.bin"]))


def gate_1_the_checkpoint_is_what_we_think(failures):
    """591 vocabulary entries, and not one merge rule to build them with."""
    print("gate 1 -- the checkpoint's tokenizer files")
    snap = _snapshot()
    vocab = json.load(open(snap / "vocab.json"))
    merges = [m for m in open(snap / "merges.txt").read().split("\n")
              if m.strip() and not m.startswith("#")]
    print(f"  vocabulary entries : {len(vocab)}")
    print(f"  merge rules        : {len(merges)}")
    if len(vocab) != 591:
        failures.append(f"vocabulary has {len(vocab)} entries, expected 591")
    if merges:
        # If merges ever appear, the character-level fallback stops happening and
        # every ChemBERTa number in the study changes meaning. Stop.
        failures.append(f"merges.txt now has {len(merges)} rules -- the reader's "
                        "behaviour has changed and every ChemBERTa result must "
                        "be rebuilt")
    missing = [c for c in ("l", "r", "[", "]", "+", "@", "H") if c in vocab]
    if missing:
        failures.append(f"characters that were absent are now present: {missing}")
    present = [t for t in ("Cl", "Br", "[C@H]", "[C@@H]", "[N+]", "[O-]") if t not in vocab]
    if present:
        failures.append(f"chemical tokens missing from the vocabulary: {present}")
    print("  the whole-atom tokens are present but unreachable, as expected")
    print()
    return snap


def gate_2_the_model_was_trained_character_by_character(snap, failures):
    """The weights themselves say which reader was used.

    [unused1]..[unused10] cannot have appeared in any training text, so their
    embeddings are still at initialisation and give an exact never-trained
    reference. Anything the model actually saw sits an order of magnitude above
    it.
    """
    import torch
    print("gate 2 -- what the pretrained weights say the reader was")
    sd = torch.load(snap / "pytorch_model.bin", map_location="cpu")
    W = sd["roberta.embeddings.word_embeddings.weight"].double().numpy()
    spread = W.std(axis=1)
    vocab = json.load(open(snap / "vocab.json"))
    inv = {i: t for t, i in vocab.items()}

    never = [i for i in range(W.shape[0]) if inv.get(i, "").startswith("[unused")]
    never += list(range(593, W.shape[0]))          # past the vocabulary entirely
    ref = spread[never].mean()
    print(f"  never-trained reference ({len(never)} rows): {ref:.5f}")

    # Tokens only an atom-level reader could emit. All must be UNTRAINED.
    for tok in ("Cl", "Br", "[C@H]", "[C@@H]", "[N+]", "[O-]", "[nH]", "[UNK]"):
        ratio = spread[vocab[tok]] / ref
        verdict = "untrained" if ratio < 2.0 else "TRAINED"
        print(f"    {tok:8s} {ratio:6.2f}x  {verdict}")
        if ratio >= 2.0:
            failures.append(
                f"{tok} now looks trained ({ratio:.2f}x). The premise of this "
                "gate -- that the model never saw whole-atom tokens -- no longer "
                "holds; re-measure before changing any reader.")

    # Single characters the model plainly did see.
    for tok in ("C", "c", "O", "N", "1", "="):
        ratio = spread[vocab[tok]] / ref
        print(f"    {tok:8s} {ratio:6.2f}x  trained")
        if ratio < 5.0:
            failures.append(f"{tok} does not look trained ({ratio:.2f}x)")

    multi = [i for t, i in vocab.items()
             if len(t) > 1 and not t.startswith("[unused")
             and t not in ("[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]")]
    trained_multi = [inv[i] for i in multi if spread[i] > 3 * ref]
    print(f"  multi-character chemical tokens carrying trained embeddings: "
          f"{len(trained_multi)} of {len(multi)}")
    if trained_multi:
        failures.append(f"multi-character tokens now look trained: {trained_multi[:10]}")
    print("  => the model was pretrained through the character-level fallback.")
    print("     An atom-level reader would hand it 543 random vectors.")
    print()


def _qm9_side():
    sys.path.insert(0, str(REPO / "scripts"))
    import process_and_train as qm9
    return qm9


def _lab_side():
    root = Path(os.environ.get("KIRBY_ROOT", REPO.parent / "KIRBy"))
    sys.path.insert(0, str(root / "src"))
    from kirby.representations import molecular as lab
    return lab


def gate_3_both_pipelines_use_one_reader(failures):
    """The two halves must load the same reader class and split a molecule into
    the same tokens.

    Token ids are compared EXACTLY -- that is the invariant, and it is what was
    broken. The pooled vectors are compared to float32 rounding rather than
    bit-identically, because the validation side encodes in batches of 32 and
    the QM9 side one molecule at a time; padding and a different matmul shape
    change the last bit or two of a float32 accumulation. That is arithmetic
    noise at 1e-7. The defect this gate exists for moved coordinates by up to
    3.03, seven orders of magnitude larger, so the two are in no danger of being
    confused.
    """
    print("gate 3 -- the two pipelines agree")
    qm9, lab = _qm9_side(), _lab_side()

    tok_qm9, _ = qm9.get_chemberta_model()
    print(f"  QM9 pipeline reader        : {type(tok_qm9).__name__}")
    if type(tok_qm9).__name__ != READER_CLASS:
        failures.append(f"QM9 pipeline uses {type(tok_qm9).__name__}, expected {READER_CLASS}")

    smiles = [s for s, _ in PROBES]
    lab_vectors = lab.create_chemberta(smiles, batch_size=8)
    tok_lab = lab._CHEMBERTA_TOKENIZER
    print(f"  validation pipeline reader : {type(tok_lab).__name__}")
    if type(tok_lab).__name__ != READER_CLASS:
        failures.append(f"validation pipeline uses {type(tok_lab).__name__}, expected {READER_CLASS}")

    ROUNDING = 1e-4          # float32 accumulation; the defect was 3.03
    worst = 0.0
    for i, (smi, name) in enumerate(PROBES):
        ids_a = tok_qm9(smi)["input_ids"]
        ids_b = tok_lab(smi)["input_ids"]
        if ids_a != ids_b:
            failures.append(f"the two readers split {name} differently: {ids_a} vs {ids_b}")
        gap = float(np.abs(qm9.chemberta_fingerprint(smi) - lab_vectors[i]).max())
        worst = max(worst, gap)
        flag = "" if gap <= ROUNDING else "   <-- DISAGREE"
        print(f"    {name:28s} same tokens: {str(ids_a == ids_b):5s}  "
              f"vector gap {gap:.2e}{flag}")
        if gap > ROUNDING:
            failures.append(f"the two pipelines disagree on {name} by {gap:.3e}, "
                            f"which is above float32 rounding")
    print(f"  largest vector gap across all probes: {worst:.2e} "
          f"(float32 rounding; the defect was 3.03)")
    print()


# Where gate 4 writes its count. A figure script reads this file; nothing
# quotes the number from a document.
COLLISIONS_JSON = REPO / "results" / "chemberta_collisions.json"
COLLISIONS_CSV = REPO / "results" / "chemberta_collisions.csv"

# The hERG the study trains on is the ChEMBL Ki extract the laboratory runner
# loads in `load_chembl_herg`, NOT `KIRBy/data/herg.tab`. That file is the TDC
# fluidigm blocker/non-blocker set, 655 rows, and no run in this study reads it;
# gate 4 counted on it until 2026-09-12, so the 648-molecule figure recorded in
# RERUN_PLAN.md 2.8k is a count on a dataset the study does not use.
HERG_KI_CSV = REPO.parent / "KIRBy" / "tests" / "data_cache" / "chembl_herg_ki.csv"

# HOW MANY MOLECULES ONE JOB CARRIES, WHICH IS NOT HOW MANY THE DATASET HOLDS.
# A QM9 job shuffles the pool and keeps the first `--sample-size` rows
# (`split_qm9`, scripts/process_and_train.py:1160 and :1173). That flag defaults
# to 10000 (:300) and slurm_scripts_qm9_rerun/generate_scripts.py prints a
# warning at :1965 if a run asks for anything else. Two molecules can only
# receive one vector inside a fit if BOTH are drawn, so the rate a run carries
# is far below the rate the pool carries. Both are counted and both are written;
# neither is called "the" rate.
#
# hERG has no such draw. `load_chembl_herg`
# (KIRBy/tests/alternative_data_noise_robustness.py:1112, called at :4540) hands
# every molecule it loads to `run_dataset` with no cap, so its two numbers are
# the same number and the per-run pass is exact rather than simulated.
QM9_MOLECULES_PER_RUN = 10000
DRAWS = 25
DRAW_SEED = 20260912


def _herg_ki_molecules():
    """The hERG molecules as the laboratory pipeline builds them, and the exact
    string it hands ChemBERTa for each.

    `load_chembl_herg` (KIRBy/tests/alternative_data_noise_robustness.py:1112)
    applies `standardise_smiles` -- largest fragment, RDKit canonical, and
    `isomericSmiles` left at its True default -- then groups on that string and
    takes the median pKi. So one row of the dataset is one standardised string,
    and that same string is what `generate_representations` passes to
    `create_chemberta`. Identity and input are therefore the same string here.

    `molecules_per_run` is None because the whole set is fitted: the caller at
    :4540 passes every molecule `load_chembl_herg` returns straight to
    `run_dataset`, with no sample-size argument anywhere on the path.
    """
    import csv

    from rdkit import Chem

    if not HERG_KI_CSV.exists():
        return None

    pairs = {}
    unparsed = 0
    rows = 0
    with open(HERG_KI_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            rows += 1
            smi = (row.get("SMILES") or "").strip()
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                unparsed += 1
                continue
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if not frags:
                unparsed += 1
                continue
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            std = Chem.MolToSmiles(mol, canonical=True)
            pairs[std] = std            # identity -> the string ChemBERTa reads
    return {
        "pairs": pairs,
        # The dataset is one row per standardised string after the median
        # collapse, so the rows a fit carries ARE the distinct molecules.
        "run_rows": sorted(pairs),
        "unparsed": unparsed,
        "rows_read": rows,
        "molecules_per_run": None,
        "how_the_run_draws_its_molecules": (
            "every molecule, no draw: load_chembl_herg passes the whole set to "
            "run_dataset (alternative_data_noise_robustness.py:4540)"),
        "tokenised_string": "standardised canonical, stereochemistry kept",
    }


def _qm9_molecules():
    """The QM9 molecules as the QM9 pipeline builds them, and the exact string
    it hands ChemBERTa for each.

    `load_qm9` keeps the 129,428 rows named by `data/valid_qm9_indices.pth`.
    `split_and_write` then parses each row's SMILES and features
    `Chem.MolToSmiles(mol, isomericSmiles=False)` (process_and_train.py:1077,
    :1106), so the string ChemBERTa reads has had its stereochemistry removed
    before the reader ever sees it. Two QM9 rows that differ only by a
    stereocentre are therefore one vector for a reason that has nothing to do
    with the checkpoint, which is why the two strings are kept apart: the
    identity is the isomeric canonical form, the input is the stripped one.

    `run_rows` is one entry per kept ROW, in pool order, because that is what a
    job draws from: `split_qm9` shuffles all 129,428 rows and keeps the first
    `--sample-size`. The 190 rows that repeat a molecule already in the pool are
    rows, so they can both be drawn, and they are kept here for that reason --
    the draw then collapses them the way the pool count does.
    """
    import torch
    from rdkit import Chem
    from torch_geometric.datasets import QM9

    root = REPO / "data" / "QM9"
    if not (root / "processed").exists():
        return None

    dataset = QM9(root=str(root))
    keep = torch.load(REPO / "data" / "valid_qm9_indices.pth")
    pairs = {}
    run_rows = []
    unparsed = 0
    for i in keep.tolist():
        mol = Chem.MolFromSmiles(dataset[i].smiles)
        if mol is None:
            unparsed += 1
            continue
        identity = Chem.MolToSmiles(mol)                       # keeps @/@@, E/Z
        fed = Chem.MolToSmiles(mol, isomericSmiles=False)       # what is featurised
        if not fed:
            unparsed += 1
            continue
        pairs[identity] = fed
        run_rows.append(identity)
    return {
        "pairs": pairs,
        "run_rows": run_rows,
        "unparsed": unparsed,
        "rows_read": len(keep),
        "molecules_per_run": QM9_MOLECULES_PER_RUN,
        "how_the_run_draws_its_molecules": (
            f"{QM9_MOLECULES_PER_RUN} rows drawn uniformly without replacement "
            f"from the pool: split_qm9 shuffles every row with torch.randperm "
            f"under the replicate seed and keeps the first --sample-size "
            f"(process_and_train.py:1160, :1173; the flag defaults to "
            f"{QM9_MOLECULES_PER_RUN} at :300)"),
        "tokenised_string": "canonical, stereochemistry stripped (isomericSmiles=False)",
    }


def _count_collisions(tok, pairs, batch=2000):
    """How many molecules receive a vector some other molecule also receives.

    Token ids are compared, not embeddings. The encoder is deterministic and
    frozen, so two molecules with identical token ids have identical 384
    coordinates by construction -- comparing ids settles it without a forward
    pass, and it cannot be confused with float32 rounding the way a coordinate
    comparison can.

    `pairs` maps one molecule's identity string to the string the pipeline
    actually feeds the reader. A group of identities sharing one token sequence
    is a clash. The split reported alongside it says whether the pipeline had
    already merged them before tokenising -- for QM9 that is stereochemistry
    stripped at featurisation, and it is ours, not the checkpoint's.
    """
    return _summarise(pairs, _token_ids_by_identity(tok, pairs, batch=batch))


def _token_ids_by_identity(tok, pairs, batch=2000):
    """Every molecule's token ids, tokenised in chunks so the whole QM9 pool
    fits. The ids are the expensive part and both counts below reuse them."""
    ids_by_identity = {}
    identities = sorted(pairs)
    for start in range(0, len(identities), batch):
        chunk = identities[start:start + batch]
        encoded = tok([pairs[k] for k in chunk])["input_ids"]
        for identity, ids in zip(chunk, encoded):
            ids_by_identity[identity] = tuple(ids)
    return ids_by_identity


def _summarise(pairs, ids_by_identity, identities=None):
    """The counts over a set of molecules, given their token ids."""
    identities = sorted(pairs) if identities is None else sorted(identities)
    by_ids = {}
    for identity in identities:
        by_ids.setdefault(ids_by_identity[identity], []).append(identity)

    clashing = [g for g in by_ids.values() if len(g) > 1]
    affected = sum(len(g) for g in clashing)
    already_merged = sum(len(g) for g in clashing
                         if len({pairs[k] for k in g}) == 1)
    examples = [sorted(g) for g in sorted(clashing, key=len, reverse=True)[:5]]
    return {
        "distinct_molecules": len(identities),
        "distinct_token_sequences": len(by_ids),
        "token_sequences_serving_more_than_one_molecule": len(clashing),
        "molecules_sharing_a_vector": affected,
        "percent_molecules_sharing_a_vector":
            round(100.0 * affected / len(identities), 2) if identities else 0.0,
        "of_those_already_one_string_before_tokenising": already_merged,
        "of_those_merged_by_the_reader": affected - already_merged,
        "largest_clash_groups": examples,
    }


def _rate_inside_one_run(run_rows, ids_by_identity, molecules_per_run,
                         draws=DRAWS, seed=DRAW_SEED):
    """The same count, over the molecules ONE JOB carries rather than over the
    whole pool.

    Two molecules receive one vector inside a fit only if both are drawn, so the
    pool rate overstates what a QM9 run carries by roughly the square of the
    fraction drawn. This draws `molecules_per_run` rows uniformly without
    replacement, which is the same draw `split_qm9` makes (torch.randperm, first
    --sample-size rows), repeats it `draws` times under a fixed seed, and
    reports the median across draws with the smallest and largest.

    The seed here is this script's own, not a replicate seed from a run. It
    fixes the answer so a re-run reproduces it; it does not reproduce any
    particular job's molecules.

    When the run carries every row -- hERG -- there is nothing to draw. One
    exact pass is made over the whole set and `simulated` is False.
    """
    import numpy as np

    rows = list(run_rows)
    whole_set = molecules_per_run is None or molecules_per_run >= len(rows)
    drawn_size = len(rows) if whole_set else int(molecules_per_run)
    rng = np.random.default_rng(seed)

    per_draw = []
    for _ in range(1 if whole_set else draws):
        take = range(len(rows)) if whole_set else rng.choice(
            len(rows), size=drawn_size, replace=False)
        # A repeated row is the same molecule twice, not two molecules, so it is
        # collapsed first -- the same convention the pool count uses.
        distinct = {rows[i] for i in take}
        by_ids = {}
        for identity in distinct:
            by_ids[ids_by_identity[identity]] = (
                by_ids.get(ids_by_identity[identity], 0) + 1)
        shared = sum(c for c in by_ids.values() if c > 1)
        per_draw.append((len(distinct), shared, 100.0 * shared / len(distinct)))

    def _stat(index, how):
        values = sorted(v[index] for v in per_draw)
        if how == "median":
            middle = len(values) // 2
            return (values[middle] if len(values) % 2
                    else 0.5 * (values[middle - 1] + values[middle]))
        return min(values) if how == "min" else max(values)

    return {
        "rows_drawn_per_run": drawn_size,
        "simulated": not whole_set,
        "draws": len(per_draw),
        "draw_seed": seed,
        "distinct_molecules_median": _stat(0, "median"),
        "molecules_sharing_a_vector_median": _stat(1, "median"),
        "molecules_sharing_a_vector_lowest_draw": _stat(1, "min"),
        "molecules_sharing_a_vector_highest_draw": _stat(1, "max"),
        "percent_molecules_sharing_a_vector_median": round(_stat(2, "median"), 3),
        "percent_molecules_sharing_a_vector_lowest_draw": round(_stat(2, "min"), 3),
        "percent_molecules_sharing_a_vector_highest_draw": round(_stat(2, "max"), 3),
    }


def load_collision_counts():
    """The written count, for a figure or table script to read.

        from crosscheck_chemberta import load_collision_counts
        counts = load_collision_counts()
        counts["datasets"]["qm9"]["inside_one_run"][
            "percent_molecules_sharing_a_vector_median"]

    Raises if the file is not there rather than returning a default, because a
    missing file means gate 4 has not been run and there is no rate to quote.
    """
    if not COLLISIONS_JSON.exists():
        raise FileNotFoundError(
            f"{COLLISIONS_JSON} has not been written. Run "
            f"`python scripts/crosscheck_chemberta.py --gates 4`.")
    return json.loads(COLLISIONS_JSON.read_text())


def gate_4_the_blind_spot_is_counted(failures):
    """Two different molecules CAN receive one vector. That is inherent to the
    published checkpoint, so it is counted and printed rather than refused --
    refusing it would mean dropping ChemBERTa, which is the author's decision
    and not this gate's.

    The count is written to results/chemberta_collisions.json and .csv so a
    figure script reads it instead of a number being copied into prose.
    """
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    print("gate 4 -- how often two molecules collapse to one vector")

    # Prefer the QM9 pipeline's own loader, so the count is made through the
    # reader a run uses. That import pulls deepchem and therefore tensorflow,
    # which does not import on every machine; when it does not, load the reader
    # straight from the same model id and say so in the output and the file.
    # The two are the same object either way -- `get_chemberta_model` calls
    # `AutoTokenizer.from_pretrained(CHEMBERTA_MODEL_ID)` and nothing else
    # (process_and_train.py:1398) -- and gate 3 is what proves the class.
    reader_source = "scripts/process_and_train.py get_chemberta_model"
    try:
        tok, _ = _qm9_side().get_chemberta_model()
    except Exception as exc:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        reader_source = (f"transformers.AutoTokenizer on {MODEL_ID}; the QM9 "
                         f"pipeline would not import here ({type(exc).__name__})")
        print(f"  reader loaded directly: {reader_source}")
    if type(tok).__name__ != READER_CLASS:
        failures.append(f"gate 4 loaded {type(tok).__name__}, expected {READER_CLASS}")

    sources = {
        "herg_ki": (str(HERG_KI_CSV), _herg_ki_molecules),
        "qm9": (str(REPO / "data" / "QM9") + " filtered by data/valid_qm9_indices.pth",
                _qm9_molecules),
    }

    record = {
        "measured_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "reader_class": type(tok).__name__,
        "reader_loaded_from": reader_source,
        "what_is_counted": (
            "A molecule counts as sharing a vector when its token ids are "
            "identical to those of a different molecule in the same dataset. A "
            "molecule is one distinct canonical SMILES; the string tokenised is "
            "the one that pipeline feeds ChemBERTa. TWO POPULATIONS ARE "
            "COUNTED, and they are different numbers. 'in_the_pool' is every "
            "molecule the dataset holds. 'inside_one_run' is the molecules one "
            "job carries: a QM9 job draws 10000 rows from its pool, and two "
            "molecules share a vector inside a fit only if both are drawn, so "
            "that rate is the lower of the two. hERG carries its whole set, so "
            "its two numbers are the same number."),
        "datasets": {},
    }

    for name, (source, build) in sources.items():
        built = build()
        if built is None:
            print(f"  {name}: source not on this machine ({source}); not counted")
            failures.append(f"gate 4 could not read {name} at {source}")
            continue
        pairs, rows = built["pairs"], built["rows_read"]
        unparsed = built["unparsed"]

        ids_by_identity = _token_ids_by_identity(tok, pairs) if pairs else {}
        pool = _summarise(pairs, ids_by_identity)
        entry = {
            "source": source,
            "rows_read": rows,
            "smiles_rdkit_could_not_parse": unparsed,
            "rows_repeating_a_molecule_already_counted":
                rows - unparsed - pool["distinct_molecules"],
            "tokenised_string": built["tokenised_string"],
            "how_the_run_draws_its_molecules":
                built["how_the_run_draws_its_molecules"],
            "in_the_pool": pool,
        }
        if pool["distinct_molecules"]:
            entry["inside_one_run"] = _rate_inside_one_run(
                built["run_rows"], ids_by_identity, built["molecules_per_run"])
        record["datasets"][name] = entry

        # A source file that exists but yields nothing must not pass. Pointing
        # the hERG source at KIRBy/data/herg.tab -- which is tab separated and
        # has no SMILES column -- produced a row of zeroes, printed it and
        # returned PASSED, and the guard accepted it too. The cache has already
        # been written with two different label-column spellings by two
        # versions of `fetch_chembl_herg_ki`, so a column rename is the live
        # way this happens.
        if pool["distinct_molecules"] == 0:
            failures.append(
                f"gate 4 read {name} at {source} and built no molecules from "
                f"it: {rows} rows read, {unparsed} SMILES RDKit could not "
                f"parse. The count in results/chemberta_collisions.json is "
                f"empty and must not be quoted.")
            print(f"  {name}: {rows} rows read and no molecule built from any "
                  f"of them -- refused, not counted")
            continue
        if unparsed:
            print(f"  {name}: {unparsed} of {rows} rows had a SMILES RDKit "
                  f"could not parse")

        run = entry["inside_one_run"]
        print(f"  {name}")
        print(f"    the whole pool: {pool['distinct_molecules']} distinct "
              f"molecules -> {pool['distinct_token_sequences']} distinct token "
              f"sequences")
        print(f"      token sequences serving more than one molecule: "
              f"{pool['token_sequences_serving_more_than_one_molecule']}")
        print(f"      molecules sharing a vector with a different molecule: "
              f"{pool['molecules_sharing_a_vector']} of "
              f"{pool['distinct_molecules']} "
              f"({pool['percent_molecules_sharing_a_vector']:.2f}%)")
        print(f"      of those, already one string before tokenising: "
              f"{pool['of_those_already_one_string_before_tokenising']}; "
              f"merged by the reader: "
              f"{pool['of_those_merged_by_the_reader']}")
        if run["simulated"]:
            print(f"    inside one run, which carries {run['rows_drawn_per_run']} "
                  f"drawn rows ({run['draws']} draws, seed {run['draw_seed']}): "
                  f"median {run['molecules_sharing_a_vector_median']:.0f} of "
                  f"{run['distinct_molecules_median']:.0f} molecules share a "
                  f"vector "
                  f"({run['percent_molecules_sharing_a_vector_median']:.3f}%), "
                  f"lowest draw "
                  f"{run['molecules_sharing_a_vector_lowest_draw']}, highest "
                  f"{run['molecules_sharing_a_vector_highest_draw']}")
        else:
            print(f"    inside one run: the run carries every molecule, so the "
                  f"rate is the pool rate, "
                  f"{run['molecules_sharing_a_vector_median']:.0f} of "
                  f"{run['distinct_molecules_median']:.0f} "
                  f"({run['percent_molecules_sharing_a_vector_median']:.3f}%)")
        for group in pool["largest_clash_groups"][:2]:
            print("      one vector for: " + " AND ".join(group))

    COLLISIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    COLLISIONS_JSON.write_text(json.dumps(record, indent=1) + "\n")

    pool_columns = ["distinct_molecules", "distinct_token_sequences",
                    "token_sequences_serving_more_than_one_molecule",
                    "molecules_sharing_a_vector",
                    "percent_molecules_sharing_a_vector",
                    "of_those_already_one_string_before_tokenising",
                    "of_those_merged_by_the_reader"]
    run_columns = ["rows_drawn_per_run", "simulated", "draws",
                   "distinct_molecules_median",
                   "molecules_sharing_a_vector_median",
                   "molecules_sharing_a_vector_lowest_draw",
                   "molecules_sharing_a_vector_highest_draw",
                   "percent_molecules_sharing_a_vector_median",
                   "percent_molecules_sharing_a_vector_lowest_draw",
                   "percent_molecules_sharing_a_vector_highest_draw"]
    flat_columns = ["rows_read", "rows_repeating_a_molecule_already_counted",
                    "smiles_rdkit_could_not_parse", "tokenised_string", "source"]
    header = (["dataset"] + [f"pool_{c}" for c in pool_columns]
              + [f"run_{c}" for c in run_columns] + flat_columns)
    lines = [",".join(header)]
    for name, entry in record["datasets"].items():
        row = [name]
        row += [str(entry["in_the_pool"][c]) for c in pool_columns]
        row += [str(entry.get("inside_one_run", {}).get(c, ""))
                for c in run_columns]
        row += [str(entry[c]) for c in flat_columns]
        lines.append(",".join(f'"{v}"' if "," in v else v for v in row))
    COLLISIONS_CSV.write_text("\n".join(lines) + "\n")

    print(f"  written: {COLLISIONS_JSON.relative_to(REPO)}")
    print(f"  written: {COLLISIONS_CSV.relative_to(REPO)}")
    print("  This is the published checkpoint's blind spot to halogens, charge,")
    print("  aromatic-NH tautomers and stereochemistry, not a defect in this code.")
    print()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gates", default="1,2,3,4",
        help="which gates to run, comma separated. Gate 4 is the collision "
             "count and is the only one that writes a file; running it alone "
             "is what produces results/chemberta_collisions.json.")
    args = parser.parse_args(argv)
    wanted = {int(g) for g in args.gates.split(",") if g.strip()}

    failures = []
    snap = None
    if 1 in wanted:
        snap = gate_1_the_checkpoint_is_what_we_think(failures)
    if 2 in wanted:
        if snap is None:
            snap = _snapshot()
        gate_2_the_model_was_trained_character_by_character(snap, failures)
    if 3 in wanted:
        gate_3_both_pipelines_use_one_reader(failures)
    if 4 in wanted:
        gate_4_the_blind_spot_is_counted(failures)

    if failures:
        print(f"FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    # Name the gates that ran. `--gates 4` used to print the same sentence as a
    # full run, so a count made without gate 3 read as proof that both
    # pipelines load one reader, which that run never checked.
    ran = ",".join(str(g) for g in sorted(wanted))
    if wanted == {1, 2, 3, 4}:
        print("PASSED -- both pipelines read SMILES the way the model was "
              "pretrained.")
    else:
        print(f"PASSED -- gates {ran} only. The claim that both pipelines read "
              f"SMILES the way the model was pretrained needs all four.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
