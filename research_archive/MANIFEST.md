# Research archive — rescued 2026-08-25

**What this is.** Every research artefact from the working sessions of 20–24 August, recovered
from `/private/tmp` and copied into the repository. 583 files, 193 MB. It was sitting in a
temporary directory that the operating system can clear at any time, and none of it was in git.

**Why it exists.** Days of literature work, downloaded papers, extracted text, reference
implementations and analysis outputs were referenced by path in the working documents but had
never been saved anywhere durable. Two of those paths — `scratchpad/ALEA_EPIS_CODE.md` and
`scratchpad/forest_ae.py` — are cited in `immediate_next_steps.md` §D4 as the working code for
the uncertainty decomposition, and neither existed in the repository.

Excluded from the copy: three multi-gigabyte directories of regenerable run output (`v26`,
`faketree*`) and cached model or array files. Everything textual, every paper, every script and
every result table was kept.

---

## The written reviews — the highest-value items

| File | Size | What it is |
|---|---|---|
| `f692d614/ALEA_EPIS_LITERATURE.md` | 36 KB | **The aleatoric/epistemic evidence review.** The formal decomposition and what a model must output; a model-by-model verdict for this paper's roster; what real papers do when only some models support the split; the finding that a rising data-driven component under injected label noise is the *confirmatory* result rather than an anomaly, with the exact prior experiment identified; the community-standard metric set reported alongside; prioritised recommendations; sources actually read, and an honest list of what could not be retrieved |
| `f692d614/ALEA_EPIS_CODE.md` | 30 KB | The implementation research — reference code with source URLs and line numbers |
| `f692d614/ERROR_DISTRIBUTION_LIT.md` | 12 KB | **Is a Gaussian-only noise study missing a real error case?** Whether experimental error is Gaussian in log units; value-dependent error; systematic and non-random error; whether distribution shape matters |
| `f692d614/NOISE_PIPELINE_AUDIT.md` | 18 KB | Audit of the noise injection pipeline |
| `f692d614/RUST_AUDIT.md` | 8 KB | Audit of the Rust injector |
| `f692d614/ins_backup.md` | 44 KB | A backup of the next-steps document |

## Reference implementations

`f692d614/` holds the source files that were read to establish how the decomposition is done in
practice: five Chemprop files (loss functions, uncertainty predictor, calibrator, metrics,
predictors), the estimator, and three files from the Ryu et al. reference implementation
(`ryu_blocks.py`, `ryu_mc_dropout.py`, `ryu_train_cep.py`). `forest_ae.py` is the runnable
quantile-forest decomposition.

## Papers — full text and extracted text

`28450b4e/` holds the noise and measurement-error literature chase: Bentz, Kalliokoski, Kolmar,
Krüger, Landrum, Lange 1989, Hampel, Heid, Frénay 2014, Feng 2007, Gini, Hahn, Dragiev,
Dablander, Lee, and others — each as both the downloaded document and its extracted text.

`f692d614/` holds Heid 2023, Hirschfeld, and two further papers, likewise as document plus text.

## Analysis outputs

`e1d07839/` holds the scripts and result tables from the August audits — dose-matching tables,
the variance-decomposition outputs, the blocker list, and the end-to-end smoke-test outputs.
`28450b4e/` holds the noise-design pilot and range-finding outputs.

---

**Do not delete any of this without checking what still references it.** The working documents
cite several of these files by path.
