#!/usr/bin/env python3
"""
RETIRED. This script does not run; it names the two that replaced it.

WHAT IT USED TO DO
------------------
Scan the results directory, compare it against a protocol written at the top of
the file, and write SLURM scripts for whatever was missing.

WHY IT CANNOT DO THAT ANY MORE
------------------------------
Every part of the protocol it compared against, and every flag it emitted, was
retired between 2026-08-26 and 2026-08-27, and nothing in the file noticed:

  * the six noise conditions it expected -- legacy, value_proportional,
    quantile, threshold, outlier, heteroscedastic -- were deleted in
    noiseInject 1.0.0. No injector can produce any of them, so every cell of its
    gap report was MISSING by construction and nothing that had run was checked
  * its level grid was 0.0 to 0.6 in steps of 0.1. The settled grid is
    0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, and censoring is swept on a different axis
    entirely -- so it could not see a genuinely absent 0.75 or 1.5, because those
    levels were not in the set it compared against
  * `smiles` was among its representations. One-hot SMILES is refused by name in
    process_and_train.py; mol2vec, in the sister list, is deleted outright
  * `gin` and `gcn` are not in the study's model roster
  * the job scripts it wrote passed `--sigma` and `--noise-strategy`, both of
    which process_and_train.py now refuses by name, so every one of them would
    have exited on its first line
  * its SLURM header activated micromamba, which has never worked on this
    cluster; an unactivated job silently ran the system Anaconda

Fixing those in place would have meant a second gap analyser and a second job
generator for one grid. Two generators for one grid is what let this file's
condition names and level grid go stale unnoticed in the first place, so the
work went to the one of each that already exists.

WHERE THE TWO JOBS WENT
-----------------------
  the audit    scripts/verify_anova_complete.py
               Reads the conditions, their level grids, the representations and
               the model roster from the job generator, which reads
               noise_conditions.json. Prints the gaps, and the exact generator
               command that fills them.

  the jobs     slurm_scripts_qm9_rerun/generate_scripts.py
               Owns the flag spelling, the level grid per condition and the array
               shape, and refuses at import if its condition list and
               noise_conditions.json disagree. Follow
               slurm_scripts_qm9_rerun/RUNBOOK.md.

The body of this file is in git history if any of it is wanted back.
"""

import sys

MESSAGE = """
gap_analysis_and_slurm.py is retired and does nothing.

  To audit what is missing:
      python scripts/verify_anova_complete.py <results_dir>

  To generate the jobs that fill it:
      python slurm_scripts_qm9_rerun/generate_scripts.py --help
      then slurm_scripts_qm9_rerun/RUNBOOK.md

Why: the six noise conditions this script expected were deleted, its level grid
stopped at 0.6 against a settled grid reaching 1.5, one of its representations is
refused by name, and the job scripts it wrote passed two flags that
process_and_train.py refuses. See the docstring at the top of this file.
"""


def main():
    print(MESSAGE.strip())
    return 2


if __name__ == "__main__":
    sys.exit(main())
