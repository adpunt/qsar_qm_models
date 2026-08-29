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
on the record rather than assumed.

Run:  python scripts/crosscheck_chemberta.py
"""

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


def gate_4_the_blind_spot_is_counted(failures):
    """Two different molecules CAN receive one vector. That is inherent to the
    published checkpoint, so it is counted and printed rather than refused --
    refusing it would mean dropping ChemBERTa, which is the author's decision
    and not this gate's."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    print("gate 4 -- how often two molecules collapse to one vector")
    qm9 = _qm9_side()
    tok, _ = qm9.get_chemberta_model()

    herg = REPO.parent / "KIRBy" / "data" / "herg.tab"
    if not herg.exists():
        print(f"  hERG file not found at {herg}; skipping the count")
        print()
        return

    molecules = set()
    for line in herg.read_text().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        mol = Chem.MolFromSmiles(parts[1].strip('"'))
        if mol is not None:
            molecules.add(Chem.MolToSmiles(mol))

    by_ids = {}
    for canon in sorted(molecules):
        by_ids.setdefault(tuple(tok(canon)["input_ids"]), []).append(canon)
    clashing = [group for group in by_ids.values() if len(group) > 1]
    n_molecules_affected = sum(len(g) for g in clashing)

    print(f"  hERG: {len(molecules)} distinct molecules "
          f"-> {len(by_ids)} distinct token sequences")
    print(f"  token sequences serving more than one molecule: {len(clashing)}")
    print(f"  molecules sharing a vector with a different molecule: "
          f"{n_molecules_affected} ({100.0 * n_molecules_affected / len(molecules):.1f}%)")
    for group in clashing[:3]:
        print("    one vector for: " + " AND ".join(group))
    print("  This is the published checkpoint's blind spot to halogens, charge")
    print("  and stereochemistry, not a defect in this code.")
    print()


def main():
    failures = []
    snap = gate_1_the_checkpoint_is_what_we_think(failures)
    gate_2_the_model_was_trained_character_by_character(snap, failures)
    gate_3_both_pipelines_use_one_reader(failures)
    gate_4_the_blind_spot_is_counted(failures)

    if failures:
        print(f"FAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED -- both pipelines read SMILES the way the model was pretrained.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
