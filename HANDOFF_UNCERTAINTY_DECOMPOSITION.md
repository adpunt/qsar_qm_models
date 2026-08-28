# Handoff — wire the uncertainty decomposition in, then run three checks

Written 2026-08-28 at the end of chat I. Everything below was verified against the working tree
that day. Paste the prompt at the bottom.

---

## What is already built and committed

| What | Where | State |
|---|---|---|
| One shared definition of the split, in **variances** throughout | `scripts/uncertainty_decomposition.py` | Built. Gated by `scripts/test_uncertainty_decomposition.py`, 25 checks. One needs scikit-learn 1.6.1 with quantile-forest 1.4.1 and reports as blocked on any other pair |
| Forest branch ends of 5 rather than 1 | `models/model_defaults.py`, spec version 1.4.0 | **Live in both pipelines** — both read this file |
| Twelve corrected evidential pairs | `models/models.py` | Committed. The data term is `beta/(alpha-1)`, confirmed against chemprop's own implementation in `research_archive/f692d614/chemprop_v1_uncertainty_predictor.py:740-800` |
| A variational layer that predicts noise per molecule | `models/models.py`, `VBLLLayer(..., heteroscedastic=True)` | Built. Gated by `scripts/test_heteroscedastic_vbll.py`, 6 checks |
| A Gaussian process that predicts noise per molecule | `models/models.py:7562` | Already existed. Writes both components. Never run by any job |

## What is NOT wired — verified by search on 2026-08-28, all four came back empty

- Nothing imports `scripts/uncertainty_decomposition.py`.
- Nothing can reach `heteroscedastic=True` — there is no command-line switch and no roster entry.
- Neither new model appears in `slurm_scripts_qm9_rerun/generate_scripts.py` or
  `slurm_scripts_uncertainty_rerun/generate_scripts.py`.
- `KIRBy/tests/alternative_data_noise_robustness.py` writes no component columns at all.

**So every number quoted below came from a test harness calling the code directly. None of it is
pipeline output.**

## What was measured, and what it means

All on real QM9 — the HOMO-LUMO gap, nine descriptors from `data/QM9/raw/gdb9.sdf.csv`.

**The forests had no data-noise term at all.** At one molecule per branch end the within-branch
spread is zero, so the whole predictive variance landed in the model term. Both forests measured;
they behave identically to within 0.001 at every setting. Five molecules per branch end costs 0.025
of clean R² and gains 0.006 under noise for the ordinary forest, 0.024 and 0.023 for the quantile
forest.

**The noise-predicting Gaussian process is free and it works.** R² 0.5315 against 0.5318 for the
ordinary one on identical data. Its data-noise term correlates with the true noise size at +0.79
while its model term sits at −0.12 against the same thing.

**🔴 The forest does not separate the two components.** Its data term correlates at +0.84 and its
model term at +0.81 — the same signal reported twice. The Gaussian process separates them; the
forest does not. **This is a finding, and it may mean the forests should not carry a decomposition
column in the paper at all.**

**The zero-noise control is clean for both** — −0.16 for the forest and −0.31 for the Gaussian
process, so neither is tracking molecular features instead of the corruption.

**⚠️ The variational layer is NOT verified to that standard.** Its own check shows the noisier half
of molecules getting a higher data-noise value than the quieter half, by a factor of 1.5. There is
no correlation number, no zero-noise control and no accuracy comparison. It is built and tested; it
is not measured.

**⚠️ Two limits on every correlation above.** Each molecule received one of only two noise amounts,
so +0.79 means the model told two blocks apart rather than ranked molecules. And the correlation is
against the noise SIZE a molecule's region carries, not the exact error drawn for it, because
held-out labels are clean by design.

## The two Bayesian networks proper are untouched

The roster's four Bayesian network families are two with a distribution over the weights and two
variational-last-layer models. Only the variational ones now have a data-noise term. The other two
have none at all. Two routes exist and both are in the literature on disk: give the network a
second output that predicts variance (Kendall & Gal `kg.pdf`, Ryu `ryu2.pdf`, Scalia `scalia.pdf`,
Heid `heid2023.txt:325-350`), or use evidential learning, which is **already implemented in
`models/models.py`** and only ever missed because no job passes the switch for it.

## 🔴 Another session edits this repository at the same time

On 2026-08-28 a concurrent session wrote `models/models.py` from its own copy and destroyed an
uncommitted change that had already passed its tests. **Commit each piece the moment it is green.
Do not hold a working tree.**

---

## The prompt

> **Prompt.** Wire the uncertainty decomposition into both pipelines, then run three checks on it.
> Read `HANDOFF_UNCERTAINTY_DECOMPOSITION.md` and `RERUN_PLAN.md` sections 5.5, 5.5a to 5.5f first.
> The decomposition itself is built and committed — one shared definition in
> `scripts/uncertainty_decomposition.py`, a variational layer that predicts noise per molecule, a
> Gaussian process that does the same, and the forests repaired. **None of it is connected to
> anything.** Do not rebuild it and do not redesign it.
>
> **Another session edits this repository concurrently and has already destroyed one uncommitted
> change that had passed its tests. Commit every piece the moment it is green.**
>
> **Part one — connect it.**
>
> 1. Import `scripts/uncertainty_decomposition.py` in `models/models.py` and in
>    `KIRBy/tests/alternative_data_noise_robustness.py`, and have both pipelines get every split
>    from it. Delete the local definitions it replaces, per the delete list in `RERUN_PLAN.md` 5.5.
>    Everything in that module is a variance; convert to a standard deviation once, at the point of
>    writing, and never add two standard deviations.
> 2. Replace the quantile forest's data-noise term. It currently reports half the 16-to-84 quantile
>    gap and calls that the data term, with no model term at all. Use the forest split in the shared
>    module, which walks trees that are already fitted and costs no retraining. Do the same for the
>    ordinary forest. Leave the reported uncertainty column as the quantile gap so the existing
>    analysis is unaffected, and say in the code why the split's total and that column are two
>    different estimates of the same thing.
> 3. Give the variational layer a command-line switch and a roster entry, in both pipelines. Add the
>    Gaussian process that predicts noise per molecule to both rosters too — it has never run in a
>    job and its behaviour under a real job script is unverified.
> 4. Add the two component columns to the laboratory pipeline's output, which currently has none,
>    and make the merge step accept them without shifting any existing column.
> 5. **Put a column on every row saying whether each component varies per molecule or is one number
>    copied onto every row.** This is the distinction that decides whether a correlation means
>    anything, and no file records it today. Call `assert_matches_support` before every write so a
>    model whose output disagrees with what is claimed for it fails the run rather than writing a
>    column nobody can interpret.
>
> **Part two — three checks. Each ships as a test that fails if the fix it guards is removed.**
>
> 1. **Give every molecule the same amount of noise and check the model finds nothing.** Correlate
>    each model's data-noise term against the same pattern used in the uneven case. There is nothing
>    to find, so a model that still scores high is tracking something else about the molecules and
>    its positive result is an artefact. This is the control that has not been run and it applies to
>    the Gaussian process result as much as to anything else.
> 2. **Give every molecule a different amount of noise and check the model ranks them correctly.**
>    Every correlation measured so far used only two noise amounts, so the models were separating two
>    blocks rather than ranking molecules. Use a graded scale and report the correlation again. If it
>    collapses, say so — that is the real answer.
> 3. **Measure the variational layer the way the Gaussian process was measured.** It has no
>    correlation number, no zero-noise control and no accuracy comparison against the ordinary
>    variational model. Until it has all three it is built but unproven, and it must not be reported
>    beside the Gaussian process as though the two carry the same evidence.
>
> Run all three on real QM9 with the models scored out of fold on scaffold groups, and report each
> correlation beside its zero-noise value, never alone. Keep the results disaggregated — one row per
> model, representation, noise condition and level, never pooled. Update `RERUN_PLAN.md`. Do not
> touch `paper.tex`.
