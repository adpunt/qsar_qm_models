#!/usr/bin/env python
"""Re-run the hERG ChEMBL query into a SEPARATE file and compare it to the cache.

The working path must not move. `fetch_chembl_herg_ki` in
`<KIRBy>/tests/alternative_data_noise_robustness.py` returns
`tests/data_cache/chembl_herg_ki.csv` (1,415 molecules) before it reaches any
filter, and every hERG result in the study came from that file. This script
never writes to it. It repeats the same REST query and the same three filters
into `chembl_herg_ki.refetch_<UTC date>.csv` beside it, and writes a comparison
JSON saying how many of the 1,415 survive today, what release answered, and
where the inter-assay standard-deviation filter now disagrees.

The query and the filters are copied from that function as it stands on
2026-09-12: binding assays only (`assay_type == "B"`), median pChEMBL per raw
ChEMBL canonical SMILES, and drop any molecule whose pChEMBL values across
assays have a standard deviation above 1.0 log units (a molecule measured once
has no standard deviation and is kept).

Usage:
    KIRBY_ALLOW_CHEMBL_FETCH=1 python scripts/verify_herg_provenance.py \
        --cache /Users/apunt/repos/KIRBy/tests/data_cache/chembl_herg_ki.csv

    # reuse an already-downloaded raw pull instead of hitting the API again
    KIRBY_ALLOW_CHEMBL_FETCH=1 python scripts/verify_herg_provenance.py \
        --raw-json <path written by an earlier run>
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

TARGET_ID = "CHEMBL240"
ACTIVITY_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
STATUS_URL = "https://www.ebi.ac.uk/chembl/api/data/status.json"
STD_CUTOFF = 1.0


def chembl_release():
    r = requests.get(STATUS_URL, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j.get("chembl_db_version", "unknown"), j.get("chembl_release_date", "unknown")


def pull_activities(raw_json=None, limit=1000, sleep=0.5):
    """Every hERG Ki activity ChEMBL will return, unfiltered."""
    if raw_json is not None and Path(raw_json).exists():
        return json.loads(Path(raw_json).read_text())

    records, offset = [], 0
    while True:
        params = {
            "target_chembl_id": TARGET_ID,
            "standard_type": "Ki",
            "pchembl_value__isnull": "false",
            "standard_relation": "=",
            "data_validity_comment__isnull": "true",
            "limit": limit,
            "offset": offset,
            "format": "json",
        }
        resp = requests.get(ACTIVITY_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        acts = data.get("activities", [])
        if not acts:
            break
        for a in acts:
            records.append({
                "canonical_smiles": a.get("canonical_smiles"),
                "pchembl_value": a.get("pchembl_value"),
                "assay_type": a.get("assay_type"),
                "assay_chembl_id": a.get("assay_chembl_id"),
                "molecule_chembl_id": a.get("molecule_chembl_id"),
            })
        print(f"    offset={offset}, {len(records)} activities so far", flush=True)
        if data.get("page_meta", {}).get("next") is None:
            break
        offset += limit
        time.sleep(sleep)
    return records


def apply_filters(records):
    """The three filters, each one's count returned beside its result."""
    df = pd.DataFrame(records)
    counts = {"activities_returned": int(len(df))}

    df = df[df["assay_type"] == "B"].copy()
    counts["activities_after_binding_filter"] = int(len(df))

    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value", "canonical_smiles"])
    counts["activities_with_a_value_and_a_structure"] = int(len(df))

    grouped = df.groupby("canonical_smiles")["pchembl_value"]
    medians = grouped.median().reset_index()
    stds = grouped.std().reset_index().rename(columns={"pchembl_value": "std"})
    n_assays = grouped.size().reset_index().rename(columns={"pchembl_value": "n_assays", 0: "n_assays"})
    merged = medians.merge(stds, on="canonical_smiles").merge(n_assays, on="canonical_smiles")
    counts["molecules_before_sd_filter"] = int(len(merged))

    keep = merged["std"].isna() | (merged["std"] <= STD_CUTOFF)
    counts["molecules_dropped_by_sd_filter"] = int((~keep).sum())
    counts["molecules_measured_once"] = int(merged["std"].isna().sum())
    kept = merged[keep].copy()
    counts["molecules_after_sd_filter"] = int(len(kept))
    return kept, merged, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Users/apunt/repos/KIRBy/tests/data_cache/chembl_herg_ki.csv")
    ap.add_argument("--out-dir", default=None,
                    help="default: the cache's own directory")
    ap.add_argument("--raw-json", default=None,
                    help="reuse a saved raw pull instead of querying ChEMBL")
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.exists():
        sys.exit(f"cache not found: {cache}")
    out_dir = Path(args.out_dir) if args.out_dir else cache.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("KIRBY_ALLOW_CHEMBL_FETCH") != "1" and args.raw_json is None:
        sys.exit("set KIRBY_ALLOW_CHEMBL_FETCH=1 to query ChEMBL over the network")

    stamp = datetime.now(timezone.utc)
    day = stamp.strftime("%Y-%m-%d")
    fresh_csv = out_dir / f"chembl_herg_ki.refetch_{day}.csv"
    raw_path = out_dir / f"chembl_herg_ki.refetch_{day}.raw.json"
    report_path = out_dir / f"chembl_herg_ki.refetch_{day}.comparison.json"
    if fresh_csv.resolve() == cache.resolve():
        sys.exit("refusing to write over the cache")

    release, release_date = chembl_release()
    print(f"  ChEMBL answering today: {release}, dated {release_date}")

    records = pull_activities(args.raw_json)
    if args.raw_json is None:
        raw_path.write_text(json.dumps(records))
    kept, merged, counts = apply_filters(records)

    fresh = kept[["canonical_smiles", "pchembl_value"]].copy()
    fresh.columns = ["SMILES", "pChEMBL"]
    fresh.to_csv(fresh_csv, index=False)

    cached = pd.read_csv(cache)
    label_col = next(c for c in ("pKi", "pChEMBL", "pchembl_value") if c in cached.columns)
    cached = cached.rename(columns={label_col: "pChEMBL"})

    cached_set = set(cached["SMILES"])
    fresh_set = set(fresh["SMILES"])
    survive = cached_set & fresh_set
    gone = sorted(cached_set - fresh_set)
    added = sorted(fresh_set - cached_set)

    # Where a molecule is in both files, does it carry the same number?
    join = cached.merge(fresh, on="SMILES", suffixes=("_cached", "_fresh"))
    join["delta"] = (join["pChEMBL_fresh"] - join["pChEMBL_cached"]).abs()
    changed = join[join["delta"] > 1e-9]

    # Of the cached molecules that are gone, how many the standard-deviation
    # filter removed, versus how many ChEMBL no longer returns at all.
    pre_filter = set(merged["canonical_smiles"])
    dropped_by_sd = [s for s in gone if s in pre_filter]
    not_returned = [s for s in gone if s not in pre_filter]

    report = {
        "written_utc": stamp.isoformat(timespec="seconds"),
        "purpose": "provenance check for the hERG cache; the cache is not modified",
        "cache_file": str(cache),
        "cache_molecules": int(len(cached)),
        "cache_label_column": label_col,
        "fresh_file": str(fresh_csv),
        "fresh_release": release,
        "fresh_release_date": release_date,
        "filter_counts_today": counts,
        "cached_molecules_still_present": len(survive),
        "cached_molecules_absent": len(gone),
        "cached_absent_because_sd_filter_now_removes_them": len(dropped_by_sd),
        "cached_absent_because_chembl_no_longer_returns_them": len(not_returned),
        "molecules_new_since_the_cache": len(added),
        "shared_molecules_with_a_different_label": int(len(changed)),
        "largest_label_change_log_units": float(join["delta"].max()) if len(join) else None,
        "examples_absent": gone[:20],
        "examples_added": added[:20],
        "examples_changed": changed.head(20).to_dict("records"),
    }
    report_path.write_text(json.dumps(report, indent=1))

    print(json.dumps({k: v for k, v in report.items()
                      if not k.startswith("examples")}, indent=1))
    print(f"\n  fresh pull     -> {fresh_csv}")
    print(f"  comparison     -> {report_path}")
    print(f"  cache untouched: {cache}")


if __name__ == "__main__":
    main()
