# What is in here

**The answer is `CHOSEN_SETTINGS.md` and `CHOSEN_SETTINGS.json`.** Nothing else
in this folder is the answer, and several files look like it but are not.

To see one table:

    python scripts/write_chosen_settings.py --show qm9
    python scripts/write_chosen_settings.py --show qm9/mlp_bnn_full_variational

The rules behind the tables are `RERUN_PLAN.md` section 5.7RULES.

## The files that matter

| file | what it is |
|---|---|
| `CHOSEN_SETTINGS.md` | sixteen tables: four datasets x four models. One row per candidate setting, one column per representation, the mean of those changes, three filters, a verdict. |
| `CHOSEN_SETTINGS.json` | the chosen setting per dataset and model, or `null` where the default is kept |
| `STATUS.md` | what is running right now, what died, what was restarted. Written by `scripts/watch_local_runs.py`. |

## The raw fits behind them

| pattern | what it is |
|---|---|
| `per_model_c_*.csv` | CLEAN, five representations. **The main experiment.** |
| `per_model_x_*.csv` | NOISE at 0.5, PDV and ChemBERTa only. **One filter column, not the table.** |
| `per_model_*_recovered.csv` | rows rescued from files written under an older column layout |
| `per_model_bayes_*.csv`, `per_model_varia.csv` | QM9 at NOISE 0.5, written before the naming settled. **These have no `level` column**; anything reading them must treat a missing level as 0.5, not 0.0, or the noise run is read as clean. |
| `trials_*.csv` | the random search. Produces the CANDIDATES, never the answer: each representation drew its own twelve settings and scored them only on itself. |
| `best_by_pairing*.json` | the search's per-pairing winners. Inputs, not decisions. |
| `features/`, `split_n*.json` | cached feature matrices and splits |

## Superseded

`superseded/` holds earlier answers and runs made under rules that no longer
apply. It has its own README. Nothing in it should be quoted.

## Representations

Five: PDV, ChemBERTa, ECFP4, MHG-GNN, Sort & Slice. **Avalon was dropped from the
study on 2026-09-01.** Rows already collected for it stay on disk and are not
read.
