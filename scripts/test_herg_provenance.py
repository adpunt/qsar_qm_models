#!/usr/bin/env python
"""Guards for the hERG cache and for the filters that built it.

Four checks, none of which touches the network by default:

1. The three filters behave as written on a hand-made input: non-binding
   assays are dropped, a molecule's repeated measurements collapse to their
   median, a molecule measured once survives, and a molecule whose
   measurements have a standard deviation above 1.0 log units is dropped.
2. The cache still hashes to what its provenance file records.
3. The re-fetch script refuses to write over the cache.
4. If a saved raw pull is on disk beside the cache, the filters applied to it
   return the cache's 1,415 molecules and the same labels.

Run:
    python scripts/test_herg_provenance.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_herg_provenance import apply_filters, STD_CUTOFF  # noqa: E402

CACHE = Path("/Users/apunt/repos/KIRBy/tests/data_cache/chembl_herg_ki.csv")
STAMP = CACHE.with_name("chembl_herg_ki.provenance.json")

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def test_filters_on_a_made_up_input():
    records = [
        # binding, one measurement, kept at its own value
        {"canonical_smiles": "CCO", "pchembl_value": "6.0", "assay_type": "B"},
        # binding, three measurements, kept at the median 7.0
        {"canonical_smiles": "CCN", "pchembl_value": "6.8", "assay_type": "B"},
        {"canonical_smiles": "CCN", "pchembl_value": "7.0", "assay_type": "B"},
        {"canonical_smiles": "CCN", "pchembl_value": "7.2", "assay_type": "B"},
        # binding, two measurements 3.0 apart, standard deviation 2.12, dropped
        {"canonical_smiles": "CCC", "pchembl_value": "5.0", "assay_type": "B"},
        {"canonical_smiles": "CCC", "pchembl_value": "8.0", "assay_type": "B"},
        # functional assay, dropped before anything else
        {"canonical_smiles": "CCF", "pchembl_value": "9.0", "assay_type": "F"},
    ]
    kept, merged, counts = apply_filters(records)
    got = dict(zip(kept["canonical_smiles"], kept["pchembl_value"]))
    check("non-binding assay dropped", "CCF" not in got, f"got {sorted(got)}")
    check("single measurement kept", got.get("CCO") == 6.0, f"got {got.get('CCO')}")
    check("repeats collapse to the median", got.get("CCN") == 7.0, f"got {got.get('CCN')}")
    check(f"standard deviation above {STD_CUTOFF} dropped", "CCC" not in got,
          f"got {sorted(got)}")
    check("counts line up", counts["molecules_after_sd_filter"] == 2
          and counts["molecules_dropped_by_sd_filter"] == 1
          and counts["molecules_measured_once"] == 1, json.dumps(counts))


def test_cache_hash_matches_its_stamp():
    if not CACHE.exists() or not STAMP.exists():
        print("  SKIP  cache or provenance file not on this machine")
        return
    recorded = json.loads(STAMP.read_text()).get("md5")
    actual = hashlib.md5(CACHE.read_bytes()).hexdigest()
    check("cache hash matches the provenance stamp", recorded == actual,
          f"stamp {recorded}, file {actual}")
    n = len(pd.read_csv(CACHE))
    check("cache still holds 1415 molecules", n == 1415, f"got {n}")


def test_refetch_refuses_to_write_the_cache():
    if not CACHE.exists():
        print("  SKIP  cache not on this machine")
        return
    script = Path(__file__).resolve().parent / "verify_herg_provenance.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--cache", str(CACHE)],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    check("re-fetch refuses without the environment variable",
          proc.returncode != 0 and "KIRBY_ALLOW_CHEMBL_FETCH" in (proc.stdout + proc.stderr),
          f"rc={proc.returncode}: {(proc.stdout + proc.stderr)[-200:]}")


def test_saved_raw_pull_reproduces_the_cache():
    raws = sorted(CACHE.parent.glob("chembl_herg_ki.refetch_*.raw.json")) if CACHE.parent.exists() else []
    if not raws or not CACHE.exists():
        print("  SKIP  no saved raw pull beside the cache")
        return
    raw = raws[-1]
    kept, merged, counts = apply_filters(json.loads(raw.read_text()))
    fresh = kept[["canonical_smiles", "pchembl_value"]].copy()
    fresh.columns = ["SMILES", "pChEMBL"]
    cached = pd.read_csv(CACHE)
    same = (fresh.sort_values("SMILES").reset_index(drop=True)
            .equals(cached.sort_values("SMILES").reset_index(drop=True)))
    check(f"{raw.name} reproduces the cache exactly", same,
          f"fresh {len(fresh)} molecules, cache {len(cached)}")


if __name__ == "__main__":
    print("hERG provenance guards")
    test_filters_on_a_made_up_input()
    test_cache_hash_matches_its_stamp()
    test_refetch_refuses_to_write_the_cache()
    test_saved_raw_pull_reproduces_the_cache()
    print(f"\n{len(failures)} failed" if failures else "\nall passed")
    sys.exit(1 if failures else 0)
