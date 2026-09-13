#!/usr/bin/env python3
"""Check the ChemBERTa collision count -- the counter, and the file it wrote.

WHAT THIS IS FOR

`scripts/crosscheck_chemberta.py` gate 4 counts how many molecules in a dataset
receive a token sequence some OTHER molecule in the same dataset also receives.
Identical token ids through a frozen encoder mean one identical 384-coordinate
vector, so that count is the rate at which the representation cannot tell two
of the study's molecules apart. The Methods carries the rate per dataset, so
the number has to be reproducible and the file has to be readable by a figure
script rather than retyped.

Five things are checked:

1. The counter itself, on molecules whose answer is known by hand. Chlorobenzene
   and toluene canonicalise to strings the reader turns into identical ids --
   the `l` of `Cl` has no vocabulary entry and is dropped. Benzene and pyridine
   do not collide. Counting a four-molecule set by hand and comparing is what
   catches an off-by-one in the group arithmetic.
2. The arithmetic inside results/chemberta_collisions.json: the affected count
   cannot exceed the molecule count, the percentage has to match the two
   numbers it came from, and the two halves of the split have to sum back. Both
   populations are checked, the pool and the one job.
3. That the file names the datasets the study actually trains ChemBERTa on.
   Gate 4 counted on `KIRBy/data/herg.tab` until 2026-09-12; that is the TDC
   fluidigm classification set and no run in this study reads it. The hERG the
   laboratory pipeline loads is the ChEMBL Ki extract.
4. That the file carries BOTH populations, and that the per-run one is a draw of
   the size a job really takes. The pool rate and the per-run rate differ by
   more than a factor of ten on QM9 -- 2.71% against a median 0.24% -- so a file
   carrying one of them unlabelled is how the wrong number reaches the Methods.
   The draw size is checked against the `--sample-size` default in
   `scripts/process_and_train.py` and in the QM9 job generator, so the recorded
   rate cannot go on claiming 10,000 after a run size changes.
5. That the counter reads back through `load_collision_counts()`, which is what
   a figure or table script calls.

Run:  python scripts/test_chemberta_collisions.py
"""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

try:
    from crosscheck_chemberta import (COLLISIONS_CSV, COLLISIONS_JSON,
                                      MODEL_ID, QM9_MOLECULES_PER_RUN,
                                      READER_CLASS, _count_collisions,
                                      load_collision_counts)
except ImportError as exc:
    # results/ is gitignored, so the written count survives the script being
    # reverted or an older copy being restored. Without this the guard died on
    # an ImportError traceback and read as a broken test rather than as the
    # counter having gone missing under a file that is still on disk.
    print("FAILED -- 1 problem(s):")
    print(f"  - scripts/crosscheck_chemberta.py does not provide what the "
          f"count needs ({exc}). results/chemberta_collisions.json is "
          f"gitignored and is still on disk, so any number read from it now "
          f"came from a version of the counter that is no longer here. "
          f"Restore the script and re-run "
          f"`python scripts/crosscheck_chemberta.py --gates 4`.")
    sys.exit(1)


def test_counter_on_molecules_counted_by_hand(failures):
    from rdkit import Chem, RDLogger
    from transformers import AutoTokenizer
    RDLogger.DisableLog("rdApp.*")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if type(tok).__name__ != READER_CLASS:
        failures.append(f"reader is {type(tok).__name__}, expected {READER_CLASS}")

    def canon(smi):
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))

    # Four molecules. Chlorobenzene and toluene are one pair; benzene and
    # pyridine are separate from everything.
    smiles = [canon(s) for s in ("c1ccccc1Cl", "Cc1ccccc1", "c1ccccc1", "c1ccncc1")]
    if len(set(smiles)) != 4:
        failures.append(f"the four probe molecules are not four distinct "
                        f"canonical strings: {smiles}")
        return

    ids = {s: tuple(tok(s)["input_ids"]) for s in smiles}
    if ids[smiles[0]] != ids[smiles[1]]:
        failures.append(
            "chlorobenzene and toluene no longer share token ids. The reader "
            "has changed; every ChemBERTa result and this whole count change "
            "meaning. Re-measure before touching anything else.")
    if ids[smiles[2]] == ids[smiles[3]]:
        failures.append("benzene and pyridine now share token ids, which they must not")

    counts = _count_collisions(tok, {s: s for s in smiles})
    expected = {
        "distinct_molecules": 4,
        "distinct_token_sequences": 3,
        "token_sequences_serving_more_than_one_molecule": 1,
        "molecules_sharing_a_vector": 2,
        "percent_molecules_sharing_a_vector": 50.0,
        "of_those_already_one_string_before_tokenising": 0,
        "of_those_merged_by_the_reader": 2,
    }
    for key, want in expected.items():
        if counts[key] != want:
            failures.append(f"counter on the four hand-counted molecules: "
                            f"{key} is {counts[key]}, expected {want}")

    # And the other half of the split: two molecules the PIPELINE merges before
    # the reader sees them. Both alanine enantiomers are one string once
    # stereochemistry is stripped, which is what the QM9 side feeds.
    l_ala, d_ala = canon("C[C@H](N)C(=O)O"), canon("C[C@@H](N)C(=O)O")
    stripped = Chem.MolToSmiles(Chem.MolFromSmiles(l_ala), isomericSmiles=False)
    merged = _count_collisions(tok, {l_ala: stripped, d_ala: stripped})
    if merged["of_those_already_one_string_before_tokenising"] != 2:
        failures.append(
            f"the two alanine enantiomers should count as merged BEFORE "
            f"tokenising, and the counter says "
            f"{merged['of_those_already_one_string_before_tokenising']}")
    if merged["of_those_merged_by_the_reader"] != 0:
        failures.append(
            f"the two alanine enantiomers should not be attributed to the "
            f"reader, and the counter attributes "
            f"{merged['of_those_merged_by_the_reader']}")

    # Nothing collides in a set of one.
    alone = _count_collisions(tok, {smiles[2]: smiles[2]})
    if alone["molecules_sharing_a_vector"] != 0:
        failures.append("a single molecule was counted as sharing a vector")


def test_the_written_file_is_self_consistent(failures):
    if not COLLISIONS_JSON.exists():
        failures.append(
            f"{COLLISIONS_JSON} does not exist. Produce it with "
            f"`python scripts/crosscheck_chemberta.py --gates 4`.")
        return
    record = json.loads(COLLISIONS_JSON.read_text())

    for name in ("qm9", "herg_ki"):
        if name not in record["datasets"]:
            failures.append(f"the count has no row for {name}, and the Methods "
                            f"needs one rate per dataset")
    if "herg" in record["datasets"]:
        failures.append(
            "the count has a row named 'herg'. The study's hERG is the ChEMBL "
            "Ki extract the laboratory pipeline loads, spelled herg_ki here; "
            "KIRBy/data/herg.tab is the TDC fluidigm set and no run reads it.")

    # The dataset key alone does not say which file was counted. Repointing the
    # hERG source at KIRBy/data/herg.tab while leaving the key as herg_ki gave a
    # row of zeroes that both gate 4 and this guard accepted, so the file each
    # row came from is pinned here.
    expected_source = {"herg_ki": "chembl_herg_ki.csv",
                       "qm9": "valid_qm9_indices.pth"}
    for name, want in expected_source.items():
        got = record["datasets"].get(name, {}).get("source", "")
        if name in record["datasets"] and want not in got:
            failures.append(
                f"{name} was counted on {got}, which is not the file the study "
                f"trains on ({want})")

    for name, entry in record["datasets"].items():
        if "in_the_pool" not in entry:
            failures.append(
                f"{name}: the row has no 'in_the_pool' count. The file must "
                f"say which population each rate is over -- the pool and the "
                f"molecules one job carries are different numbers.")
            continue
        c = entry["in_the_pool"]
        n, hit = c["distinct_molecules"], c["molecules_sharing_a_vector"]
        if n == 0:
            failures.append(
                f"{name}: no molecules were counted, so there is no rate for "
                f"the Methods to carry. Re-run gate 4 and check the source "
                f"file parsed: {entry.get('source')}")
            continue
        if hit > n:
            failures.append(f"{name}: {hit} molecules share a vector out of {n}")
        if c["distinct_token_sequences"] > n:
            failures.append(f"{name}: more token sequences than molecules")
        pct = round(100.0 * hit / n, 2) if n else 0.0
        if abs(pct - c["percent_molecules_sharing_a_vector"]) > 0.005:
            failures.append(
                f"{name}: the stored percentage "
                f"{c['percent_molecules_sharing_a_vector']} does not come from "
                f"{hit} of {n} (which is {pct})")
        halves = (c["of_those_already_one_string_before_tokenising"]
                  + c["of_those_merged_by_the_reader"])
        if halves != hit:
            failures.append(f"{name}: the split sums to {halves}, not {hit}")
        if entry["smiles_rdkit_could_not_parse"] < 0:
            failures.append(f"{name}: negative unparsed count")
        if entry["rows_read"] < n:
            failures.append(f"{name}: {entry['rows_read']} rows read but {n} "
                            f"distinct molecules counted")

        # The population one job carries.
        if "inside_one_run" not in entry:
            failures.append(
                f"{name}: the row has no 'inside_one_run' count. The pool rate "
                f"is not the rate a fit experiences -- on QM9 a job draws "
                f"{QM9_MOLECULES_PER_RUN} rows of {entry['rows_read']} and two "
                f"molecules share a vector inside that fit only if both are "
                f"drawn.")
            continue
        run = entry["inside_one_run"]
        if not entry.get("how_the_run_draws_its_molecules"):
            failures.append(f"{name}: the row does not say how a run draws its "
                            f"molecules, so the per-run rate cannot be read")
        drawn, med = run["rows_drawn_per_run"], run["distinct_molecules_median"]
        if drawn > entry["rows_read"]:
            failures.append(f"{name}: a run is recorded as drawing {drawn} rows "
                            f"from a pool of {entry['rows_read']}")
        if med > drawn:
            failures.append(f"{name}: {med} distinct molecules from {drawn} "
                            f"drawn rows")
        run_pct = run["percent_molecules_sharing_a_vector_median"]
        expect = round(100.0 * run["molecules_sharing_a_vector_median"] / med, 3)
        if med and abs(run_pct - expect) > 0.01:
            failures.append(
                f"{name}: the per-run percentage {run_pct} does not come from "
                f"{run['molecules_sharing_a_vector_median']} of {med} "
                f"(which is {expect})")
        if not (run["molecules_sharing_a_vector_lowest_draw"]
                <= run["molecules_sharing_a_vector_median"]
                <= run["molecules_sharing_a_vector_highest_draw"]):
            failures.append(f"{name}: the per-run median sits outside the range "
                            f"of the draws")
        if run["simulated"] and run["draws"] < 2:
            failures.append(f"{name}: {run['draws']} draw(s) is not a median "
                            f"across draws")
        if not run["simulated"] and drawn != n:
            failures.append(
                f"{name}: the run is recorded as carrying every molecule but "
                f"draws {drawn} of {n}")
        if run_pct > c["percent_molecules_sharing_a_vector"] + 0.01:
            failures.append(
                f"{name}: the per-run rate {run_pct}% is above the pool rate "
                f"{c['percent_molecules_sharing_a_vector']}%, which a draw from "
                f"that pool cannot be")

    # The QM9 draw size has to be the one a job takes. If --sample-size moves,
    # the recorded per-run rate is a rate for a run size nothing runs.
    qm9_run = record["datasets"].get("qm9", {}).get("inside_one_run", {})
    if qm9_run and qm9_run.get("rows_drawn_per_run") != QM9_MOLECULES_PER_RUN:
        failures.append(
            f"the QM9 per-run rate is over {qm9_run.get('rows_drawn_per_run')} "
            f"rows, and the counter says a job takes {QM9_MOLECULES_PER_RUN}")

    if not COLLISIONS_CSV.exists():
        failures.append(f"{COLLISIONS_CSV} does not exist beside the json")
        return
    header = COLLISIONS_CSV.read_text().split("\n")[0].split(",")
    for column in ("dataset", "pool_distinct_molecules",
                   "pool_molecules_sharing_a_vector",
                   "pool_percent_molecules_sharing_a_vector",
                   "run_rows_drawn_per_run",
                   "run_molecules_sharing_a_vector_median",
                   "run_percent_molecules_sharing_a_vector_median"):
        if column not in header:
            failures.append(f"the csv has no {column} column: {header}")


def test_the_draw_size_is_the_size_a_job_takes(failures):
    """`QM9_MOLECULES_PER_RUN` is a number written in the counter, and it has to
    keep matching the two places a run gets its sample size from."""
    checks = [
        (REPO / "scripts" / "process_and_train.py",
         r'"-n",\s*"--sample-size",\s*type=int,\s*default=(\d+)'),
        (REPO / "slurm_scripts_qm9_rerun" / "generate_scripts.py",
         r"'--sample-size',\s*type=int,\s*default=(\d+)"),
    ]
    for path, pattern in checks:
        if not path.exists():
            failures.append(f"{path} is not there, so the draw size cannot be "
                            f"checked against it")
            continue
        found = re.search(pattern, path.read_text())
        if not found:
            failures.append(
                f"no --sample-size default found in {path.name}; the counter "
                f"assumes a job takes {QM9_MOLECULES_PER_RUN} molecules and "
                f"nothing now confirms it")
            continue
        if int(found.group(1)) != QM9_MOLECULES_PER_RUN:
            failures.append(
                f"{path.name} gives a job {found.group(1)} molecules and the "
                f"counter measured the per-run rate on "
                f"{QM9_MOLECULES_PER_RUN}. Re-run gate 4.")


def test_the_file_reads_back_through_its_loader(failures):
    """A figure or table script calls `load_collision_counts()`, so that path is
    what the guard exercises -- not a hand-written open() of the same file."""
    try:
        loaded = load_collision_counts()
    except Exception as exc:
        failures.append(f"load_collision_counts() raised {type(exc).__name__}: "
                        f"{exc}")
        return
    for name in ("qm9", "herg_ki"):
        try:
            pct = (loaded["datasets"][name]["inside_one_run"]
                   ["percent_molecules_sharing_a_vector_median"])
            pool = (loaded["datasets"][name]["in_the_pool"]
                    ["percent_molecules_sharing_a_vector"])
        except KeyError as missing:
            failures.append(f"load_collision_counts() returns no {missing} for "
                            f"{name}")
            continue
        if not isinstance(pct, (int, float)) or not isinstance(pool, (int, float)):
            failures.append(f"{name}: the rates are not numbers a script can use")


def main():
    failures = []
    print("the counter, on molecules counted by hand")
    test_counter_on_molecules_counted_by_hand(failures)
    print("the written count, results/chemberta_collisions.json")
    test_the_written_file_is_self_consistent(failures)
    print("the draw size against the job's own --sample-size")
    test_the_draw_size_is_the_size_a_job_takes(failures)
    print("the file read back the way a figure script reads it")
    test_the_file_reads_back_through_its_loader(failures)

    if failures:
        print(f"\nFAILED -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    record = json.loads(COLLISIONS_JSON.read_text())
    for name, entry in record["datasets"].items():
        pool, run = entry["in_the_pool"], entry["inside_one_run"]
        print(f"  {name}, the whole pool: "
              f"{pool['molecules_sharing_a_vector']} of "
              f"{pool['distinct_molecules']} molecules share a vector "
              f"({pool['percent_molecules_sharing_a_vector']:.2f}%)")
        if run["simulated"]:
            print(f"  {name}, inside one run of {run['rows_drawn_per_run']} "
                  f"drawn rows: median "
                  f"{run['molecules_sharing_a_vector_median']:.0f} of "
                  f"{run['distinct_molecules_median']:.0f} "
                  f"({run['percent_molecules_sharing_a_vector_median']:.3f}%), "
                  f"{run['draws']} draws")
        else:
            print(f"  {name}, inside one run: the run carries every molecule, "
                  f"so the rate is the pool rate")
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
