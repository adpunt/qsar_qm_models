# The audit of 2026-08-26, and what came of it

Closed 2026-08-27. Every one of the 151 candidates carries a verdict.

## The files

| file | what it holds now |
|---|---|
| `unverified.json` | **empty.** Entries were removed as they were closed. |
| `verdicts.json` | all 111 that were unverified, each with `verdict` and `evidence` |
| `confirmed_35.json` | the 35, each with a `recheck_2026_08_27` field |
| `refuted_5.json` | the 5, each with a `recheck_2026_08_27` field |
| `synthesis.md` | the original run's own summary, unchanged |

`_verdict.py` is the script that moved an entry from `unverified.json` into
`verdicts.json`. It is kept so the record can be re-read, not re-run.

## The count

| verdict | count |
|---|---|
| real-fixed | 75 |
| real-open | 14 |
| duplicate | 14 |
| partly-fixed | 4 |
| refuted | 2 |
| not-a-fault | 2 |

Of the 35 already marked confirmed: 27 real and fixed,
7 plain duplicates, 1 still open. Four separate
entries described one ECFP4 fault; two described each graph-model unpack.

Of the 5 marked refuted: three refutations stand, one is half-real, and **one is
misfiled** -- entry 4, the Graph GP constant, is refuted in name while its own
evidence confirms it.

## What every verdict means

- **real-fixed** -- the fault was reproduced by running something, the fix is in,
  and a check fails if the fix is removed.
- **real-open** -- reproduced, not fixed, because fixing it is a decision rather
  than a repair. Listed below.
- **partly-fixed** -- one half closed, the other named.
- **duplicate** -- the same fault as another entry, fixed there.
- **refuted** -- reproduced and found not to be true of the current code.
- **not-a-fault** -- true as described, and not a defect.

## Still open

| entry | verdict | what it is |
|---|---|---|
| 5 | real-open | --calibration-size is accepted, passed down, and never used: conformal calibrates on the whole  |
| 7 | real-open | Optuna-suggested values that are computed and then discarded |
| 10 | partly-fixed | model_defaults values that models.py restates as literals instead of reading — the parity audit |
| 34 | real-open | grouped_shifted is scaled by three different conventions across the three implementations; they |
| 39 | real-open | Off-registry shape x targeting combinations get two different names in the two injectors |
| 47 | real-open | `randomized_smiles` is one-hot encoded against a vocabulary and a maximum length measured on CA |
| 48 | real-open | Two latent record-misalignment routes in the Rust writer, currently unreachable on QM9 |
| 50 | partly-fixed | Temperature calibration exists only on the QM9 side, so the two pipelines' 'uncertainty' column |
| 51 | partly-fixed | Temperature is fitted on CLEAN held-out labels while the model was trained on noisy ones, and T |
| 53 | real-open | The heteroscedastic head's predicted variance is sliced off and discarded at every prediction s |
| 54 | real-open | Two decomposition helpers hard-code aleatoric to a constant broadcast across all molecules |
| 60 | real-open | conformal_hetero writes its uncertainty to a private schema that no downstream reader looks for |
| 69 | real-open | Coverage and uncertainty-error correlation are pooled across every noise level on both sides |
| 71 | real-open | auc_norm is averaged across all six noise conditions and all representations for the headline c |
| 78 | real-open | rmse, mae and the mean-uncertainty columns are in different units on the two sides while sharin |
| 95 | real-open | `chemberta` / `ChemBERTa` is two different pretrained encoders, with different objectives and d |
| 96 | real-open | The `sigma` column holds three different physical quantities across the two writers, and auc_no |
| 108 | partly-fixed | 'Replicate' means an independently reseeded resample-and-split on QM9 and a fixed CV fold on th |

The full reasoning for each is in `verdicts.json`, and the decisions themselves
are set out in `RERUN_PLAN.md` section 2.14.

## What the findings became

`RERUN_PLAN.md` sections 2.10b to 2.14. The checks they left behind are listed in
that section and in the repository README.
