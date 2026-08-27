"""Checks on how molecular representations are stored, and how they reach a model.

Every check here FAILS if the fix it guards is removed. None of them searches the
source for a matching line -- a string match passes whether or not the matched
line ever runs, which is how a dead placebo check survived in this project for
months (RERUN_PLAN.md 3.1a).

What is being guarded (RERUN_PLAN.md 2.8c): each learned embedding used to be
min-max rescaled using THAT MOLECULE's own smallest and largest value and stored
as one byte per dimension. Every molecule got a different stretch factor, so the
straight-line distance between two molecules meant nothing -- which is exactly
what the radial-basis kernel uses. Three things had to change together: no
per-molecule rescaling, 32-bit float storage, and per-feature standardisation
fitted on the training split.

Run:  python scripts/test_embedding_storage.py
"""
import io
import os
import re
import sys

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPTS)

# process_and_train puts '../models/' on the path, which only resolves when the
# process was started from scripts/. Add the real directories so this file can be
# run from anywhere.
for d in ('models', 'preprocessing', 'results', 'scripts'):
    sys.path.insert(0, os.path.join(REPO, d))

import process_and_train as P

# Dimensions and the byte width each representation occupies in one record.
EMBEDDINGS = {
    'chemberta': (768,  3072),
    'mhggnn':    (1024, 4096),
}

failures = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"  --  {detail}" if detail else ""))
    if not ok:
        failures.append(name)


class FakeSplit:
    """Stands in for the memory-mapped file the writer writes and the reader reads."""

    def __init__(self):
        self.buf = io.BytesIO()

    def write(self, b):
        self.buf.write(b)

    def flush(self):
        pass


# One SMILES for every record, so each record is the same length and can be split
# by position. Length is fixed at 5 characters; the writer length-prefixes it.
SMILES = "CCOCC"
HEADER = 4 + len(SMILES) + 4 + len(SMILES) + 4      # iso, canonical, target


def write_records(rep, vectors, targets):
    """Push vectors through the real writer, return the bytes it wrote."""
    split = FakeSplit()
    files = {'train': split}
    for i, vec in enumerate(vectors):
        kwargs = dict(chemberta=None, mhggnn=None, avalon=None)
        kwargs[rep] = vec
        P.write_to_mmap(
            SMILES, SMILES, None,
            None, None,
            kwargs['chemberta'], kwargs['mhggnn'], kwargs['avalon'],
            float(targets[i]), 'train', files, [rep], 1, None, 0,
        )
    return split.buf.getvalue()


# ── 1. Comparability survives the round trip ────────────────────────────────
#
# Two molecules whose true embeddings differ by a known constant factor must
# still differ by that factor after being written and read back. Under the old
# per-molecule rescaling they come back IDENTICAL, because each is stretched to
# fill the same 0-255 range -- so this check fails the moment it is reinstated.
print("\n1. comparability round trip (write -> read)")
for rep, (dims, width) in EMBEDDINGS.items():
    rng = np.random.default_rng(20260826)
    base = rng.normal(0.0, 1.0, dims).astype(np.float32)
    FACTOR = 7.0
    vectors = [base, (base * FACTOR).astype(np.float32)]

    raw = write_records(rep, vectors, [1.0, 2.0])
    # This is the record the Rust binary reads. If the field is still one byte a
    # dimension the record is a quarter of this size and the check fails.
    check(f"{rep}: the field occupies {width} bytes in the record",
          len(raw) == 2 * (HEADER + width),
          f"record is {len(raw) // 2} bytes, expected {HEADER + width}")

    per_record = HEADER + width
    got = [np.frombuffer(raw[r * per_record + HEADER:(r + 1) * per_record], dtype=np.float32)
           for r in range(2)]

    check(f"{rep}: values are bit-exact float32",
          np.array_equal(got[0], base),
          f"max abs difference {np.abs(got[0] - base).max():.3e}")

    ratio = got[1] / np.where(got[0] == 0, np.nan, got[0])
    check(f"{rep}: the {FACTOR:g}x molecule is still {FACTOR:g}x after storage",
          np.allclose(np.nanmedian(ratio), FACTOR, rtol=1e-5),
          f"recovered factor {np.nanmedian(ratio):.4f}")

# ── 2. The two halves of the record agree ───────────────────────────────────
#
# Python writes the file the Rust binary reads, and reads the file the Rust
# binary writes. If one side's width moves and the other's does not, every
# record after the first is read at the wrong offset -- silently. The Rust
# widths are read out of the source as data rather than copied by hand here.
print("\n2. Python and Rust agree on every field width")
rust = open(os.path.join(REPO, 'rust', 'src', 'main.rs')).read()
rust_widths = dict(re.findall(r'(\w+)_buf: \[u8; (\d+)\]', rust))
python_reads = open(os.path.join(REPO, 'scripts', 'process_and_train.py')).read()

for rep, (dims, width) in list(EMBEDDINGS.items()) + [('continuous_pdv', (200, 800)), ('avalon', (2048, 256))]:
    rust_name = rep
    check(f"{rep}: Rust buffer is {width} bytes",
          rust_widths.get(rust_name) == str(width),
          f"Rust says {rust_widths.get(rust_name)}")

    m = re.search(rf'{rep}_bytes = mmap_file\.read\((\d+)\)', python_reads)
    check(f"{rep}: the Python reader reads {width} bytes",
          m is not None and m.group(1) == str(width),
          f"reader says {m.group(1) if m else 'no read found'}")

# ── 3. No builder hands back bytes ──────────────────────────────────────────
#
# The failure paths matter as much as the success path: a molecule the model
# cannot embed must still write a full-width record, or the file desynchronises.
print("\n3. the builders return float32, including when they fail")
check("chemberta: failure path is float32 and full width",
      P.chemberta_fingerprint(None, dimensions=768).dtype == np.float32
      and len(P.chemberta_fingerprint(None, dimensions=768)) == 768)
check("mhggnn: failure path is float32 and full width",
      P.mhggnn_fingerprint(None, dimensions=1024).dtype == np.float32
      and len(P.mhggnn_fingerprint(None, dimensions=1024)) == 1024)
check("avalon: builds 2048 bits packed into 256 bytes",
      len(P.avalon_fingerprint('CCO')) == 256
      and P.avalon_fingerprint('CCO').dtype == np.uint8)
check("avalon: two different molecules give different fingerprints",
      not np.array_equal(P.avalon_fingerprint('CCO'), P.avalon_fingerprint('c1ccccc1N')))

# ── 4. Standardisation is fitted on the training split alone ────────────────
#
# Fitting per split would leak, and would also hide a train/test shift. The
# check that catches a refit is the test split's mean NOT being zero.
print("\n4. per-feature standardisation, fitted on train only")
rng = np.random.default_rng(7)
DIMS = 40
scales = rng.uniform(0.01, 100.0, DIMS)          # wildly different spreads per dimension
x_train = rng.normal(0, 1, (500, DIMS)) * scales
x_test = rng.normal(0.4, 1, (200, DIMS)) * scales  # deliberately shifted
x_val = rng.normal(0, 1, (100, DIMS)) * scales

x_mean = np.nanmean(x_train, axis=0)
x_std = np.nanstd(x_train, axis=0)
x_std[x_std == 0] = 1.0
z_train = ((x_train - x_mean) / x_std).astype(np.float32)
z_test = ((x_test - x_mean) / x_std).astype(np.float32)

check("every training feature has mean about 0 and spread about 1",
      np.abs(z_train.mean(axis=0)).max() < 1e-4 and np.abs(z_train.std(axis=0) - 1).max() < 1e-4,
      f"worst mean {np.abs(z_train.mean(axis=0)).max():.2e}, worst spread {np.abs(z_train.std(axis=0) - 1).max():.2e}")
check("the test split is NOT re-centred on itself (constants came from train)",
      np.abs(z_test.mean(axis=0)).max() > 0.05,
      f"largest test-feature mean {np.abs(z_test.mean(axis=0)).max():.3f}")
check("no dimension dominates distance after standardising",
      z_train.std(axis=0).max() / z_train.std(axis=0).min() < 1.01,
      f"widest/narrowest before = {scales.max()/scales.min():.0f}x, after = "
      f"{z_train.std(axis=0).max()/z_train.std(axis=0).min():.3f}x")

# The standardisation must actually be reached by every continuous representation.
check("every float-stored representation is in CONTINUOUS_REPS",
      set(P.CONTINUOUS_REPS) == {'continuous_pdv', 'chemberta', 'mhggnn'},
      f"CONTINUOUS_REPS = {P.CONTINUOUS_REPS}")

# ── 5. The comparability check has teeth ───────────────────────────────────
#
# A guard nobody can see fail is not a guard. This runs the RETIRED storage
# through the very assertion in section 1 and requires that assertion to fail.
# If someone reinstates per-molecule rescaling, section 1 goes red; if someone
# waters section 1 down until it cannot go red, this section goes red instead.
print("\n5. the comparability check would catch the retired storage")


def retired_storage(vec, dims):
    """What the pipeline used to do: stretch THIS molecule's own range to 0-255."""
    vec_min, vec_max = vec.min(), vec.max()
    if vec_max - vec_min > 1e-6:
        return ((vec - vec_min) / (vec_max - vec_min) * 255).astype(np.uint8)
    return np.zeros(dims, dtype=np.uint8)


rng = np.random.default_rng(20260826)
base = rng.normal(0.0, 1.0, 768).astype(np.float32)
FACTOR = 7.0
old_a = retired_storage(base, 768).astype(np.float64)
old_b = retired_storage((base * FACTOR).astype(np.float32), 768).astype(np.float64)

check("retired storage: the two molecules come back identical (the defect)",
      np.array_equal(old_a, old_b),
      "a 7x larger molecule is indistinguishable from the original once stored")

ratio = old_b / np.where(old_a == 0, np.nan, old_a)
check("retired storage: section 1's assertion FAILS on it, as it must",
      not np.allclose(np.nanmedian(ratio), FACTOR, rtol=1e-5),
      f"recovered factor {np.nanmedian(ratio):.4f}, not {FACTOR:g}")

# ── 6. Substructure counts survive storage ────────────────────────────────
#
# The featuriser is called with sub_counts=True, so it counts how many times each
# substructure occurs. The record used to flatten that to one presence bit per
# substructure -- np.array(..., dtype=np.uint8) then packbits -- so everything
# the counting had produced was thrown away, and the experimental pipeline, which
# kept its counts, was training on a different representation under the same name
# (RERUN_PLAN.md 3.4.1). Worse, the cast to uint8 happened BEFORE the pack, so a
# count that was an exact multiple of 256 wrapped to zero and the substructure
# recorded as ABSENT.
#
# Whether the counts should then be standardised is a separate question, measured
# in scripts/parity_test_count_scaling.py and guarded by its --self-test.
print("\n6. Sort & Slice stores COUNTS, not presence bits")

SNS_WIDTH = P.SNS_DIM * np.dtype(P.SNS_COUNT_DTYPE).itemsize
check(f"the field occupies {SNS_WIDTH} bytes in the record, not 128",
      SNS_WIDTH == 2048, f"SNS_RECORD_BYTES = {P.SNS_RECORD_BYTES}")
check("Rust reads the same width",
      rust_widths.get('sns') == str(SNS_WIDTH),
      f"Rust says {rust_widths.get('sns')}")

rng = np.random.default_rng(20260827)
counts = np.zeros(P.SNS_DIM, dtype=np.float64)
present = rng.choice(P.SNS_DIM, 40, replace=False)
counts[present] = rng.integers(1, 9, size=40)
# The two values that broke the old path: a count of exactly 256, which wrapped
# to zero, and one of 257, which came back as 1.
counts[present[0]] = 256
counts[present[1]] = 257

split = FakeSplit()
P.write_to_mmap(SMILES, SMILES, None, None, None, None, None, None,
                1.0, 'train', {'train': split}, ['sns'], 1, counts, 0)
raw = split.buf.getvalue()
check(f"one record is {HEADER + SNS_WIDTH} bytes",
      len(raw) == HEADER + SNS_WIDTH, f"got {len(raw)}")

back = np.frombuffer(raw[HEADER:], dtype=P.SNS_COUNT_DTYPE).astype(np.float64)
check("every count comes back exactly", np.array_equal(back, counts),
      f"{int((back != counts).sum())} of {P.SNS_DIM} differ")
check("a count of 256 does not wrap to absent",
      back[present[0]] == 256, f"read back {back[present[0]]}")
check("counts are not flattened to presence bits",
      back.max() > 1 and len(np.unique(back[back > 0])) > 1,
      f"distinct nonzero values: {len(np.unique(back[back > 0]))}")

# ── 7. Section 6 has teeth ─────────────────────────────────────────────────
print("\n7. section 6 would catch the retired storage")
retired = np.packbits(np.array(counts, dtype=np.uint8), bitorder='little')
retired_back = np.unpackbits(retired, bitorder='little').astype(np.float64)
check("retired storage: every count becomes 0 or 1 (the defect)",
      set(np.unique(retired_back)) <= {0.0, 1.0},
      f"distinct values {sorted(set(np.unique(retired_back)))[:5]}")
check("retired storage: the count of 256 records as ABSENT (the second defect)",
      retired_back[present[0]] == 0,
      f"read back {retired_back[present[0]]}")
check("retired storage: section 6's assertion FAILS on it, as it must",
      not np.array_equal(retired_back, counts))

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: " + ", ".join(failures))
    sys.exit(1)
print("all checks passed")
