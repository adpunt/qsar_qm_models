"""Break each fix, one at a time, and confirm its check goes red.

    python scripts/check_fixes_fail_when_removed.py

A check that passes is worth nothing unless it FAILS when the fix is removed.
This edits the real file, runs the named check, and puts the file back — so it
needs a clean working tree, and it takes a few minutes because several cases
load the real QM9 dataset.

It has already earned itself twice. On its first run, two checks stayed GREEN
with their fix removed:

  - the QM9 split check asserted each split is more than 10% acyclic. Under
    DeepChem's largest-first ordering WITH acyclic singletons, validation comes
    out 100% acyclic and test 82% — which passes a floor. It now asserts each
    split sits within 25 points of the population's share.
  - the sibling-file check was caught by the column rule rather than the name
    rule, so it guarded the name rule not at all; and its fixture then duplicated
    a real row, so the dedup hid the second reading. The fixture now differs on
    the dedup key.

Add a case here whenever a fix lands. The four fields after the name are the
file, the text that must be present, what to replace it with, and the check to
run.
"""
import subprocess, sys, os, shutil, tempfile

QSAR = "/Users/apunt/repos/qsar_qm_models"
KIRBY = "/Users/apunt/repos/KIRBy"
NOISE = "/Users/apunt/repos/NoiseInject"

CASES = [
 ("ECFP4 is Morgan radius 2",
  f"{QSAR}/scripts/process_and_train.py",
  "    bits = np.array(_ECFP4_GENERATOR.GetFingerprint(mol), dtype=np.uint8)",
  "    bits = np.array(Chem.RDKFingerprint(mol), dtype=np.uint8)",
  [sys.executable, f"{QSAR}/scripts/test_ecfp4_identity.py"]),

 ("the BNN carries a KL term",
  f"{QSAR}/models/models.py",
  "def bnn_elbo_criterion(base_criterion, model, n_train):",
  "def bnn_elbo_criterion(base_criterion, model, n_train):\n    return base_criterion  # BROKEN ON PURPOSE",
  [sys.executable, f"{QSAR}/scripts/test_bnn_kl_term.py"]),

 ("the spec is live, not restated",
  f"{QSAR}/models/models.py",
  "\"prior_sigma\": BAYESIAN_DEFAULTS['bnn_prior_sigma'],\n            \"in_features\": \".in_features\",\n            \"out_features\": \".out_features\", \n            \"bias\": \".bias\"",
  "\"prior_sigma\": 0.1,\n            \"in_features\": \".in_features\",\n            \"out_features\": \".out_features\", \n            \"bias\": \".bias\"",
  [sys.executable, f"{QSAR}/scripts/test_spec_is_live.py"]),

 ("no definition is shadowed",
  f"{QSAR}/scripts/utils.py",
  "def calculate_regression_metrics(",
  "def save_results(filepath, *a, **k):\n    raise NotImplementedError\n\n\ndef calculate_regression_metrics(",
  [sys.executable, f"{QSAR}/scripts/test_no_shadowed_definitions.py"]),

 ("the QM9 scaffold split keeps the acyclic molecules",
  f"{QSAR}/scripts/process_and_train.py",
  "    ordered = sorted(groups.values(), key=lambda idx: idx[0])\n    np.random.shuffle(ordered)",
  "    ordered = sorted(groups.values(), key=lambda idx: (-len(idx), idx[0]))",
  [sys.executable, f"{QSAR}/scripts/test_qm9_split_alignment.py"]),

 ("sibling files are not read as results",
  f"{QSAR}/scripts/generate_paper_figures_v2.py",
  "        if f.name.endswith(SIBLING_SUFFIXES):",
  "        if '_uncertainty_values' in f.name:",
  [sys.executable, f"{QSAR}/scripts/test_figure_conditions.py"]),

 ("a results row joins to its manifest",
  f"{QSAR}/scripts/process_and_train.py",
  "    row['iteration'] = iteration\n    row['file_no'] = file_no\n    row['noise_level'] = level",
  "    pass",
  [sys.executable, f"{QSAR}/scripts/test_result_row_condition.py"]),

 ("the generator emits no retired flag",
  f"{QSAR}/slurm_scripts_qm9_rerun/generate_scripts.py",
  "'-m dnn --bayesian-transformation last_layer -u True'",
  "'-m dnn --bayesian-transformation last -u True'",
  [sys.executable, f"{QSAR}/scripts/test_generated_job_flags.py"]),

 ("the CV scaffold key is chirality-blind",
  f"{KIRBY}/tests/alternative_data_noise_robustness.py",
  "        return Chem.MolToSmiles(core, canonical=True, isomericSmiles=False)",
  "        return Chem.MolToSmiles(core, canonical=True)",
  [sys.executable, f"{KIRBY}/tests/smoke/smoke_kirby_splits.py"]),

 ("the early-stopping carve is scaffold-grouped",
  f"{KIRBY}/tests/alternative_data_noise_robustness.py",
  "    carve = GroupShuffleSplit(n_splits=1, test_size=val_frac,\n                              random_state=fold_idx)\n    train_local, val_local = next(\n        carve.split(np.zeros(len(y_train_full)), y_train_full, groups_full))",
  "    _n = len(y_train_full) // 5\n    val_local = np.arange(_n)\n    train_local = np.arange(_n, len(y_train_full))",
  [sys.executable, f"{KIRBY}/tests/smoke/smoke_kirby_splits.py"]),

 ("the target scaler is fitted on the clean labels",
  f"{KIRBY}/tests/alternative_data_noise_robustness.py",
  "        mdl.fit(X_train, _to_model_scale(y_noisy))",
  "        _s2 = StandardScaler().fit(np.asarray(y_noisy, dtype=float).reshape(-1, 1))\n        mdl.fit(X_train, _s2.transform(np.asarray(y_noisy, dtype=float).reshape(-1, 1)).ravel())",
  [sys.executable, f"{KIRBY}/tests/smoke/smoke_kirby_target_scaling.py"]),

 ("the noise pattern matches the injection",
  f"{KIRBY}/tests/alternative_data_noise_robustness.py",
  "    return NoiseInjectorRegression.from_condition(condition, random_state=seed,\n                                                  selection_state=selection)",
  "    return NoiseInjectorRegression.from_condition(condition, random_state=seed)",
  [sys.executable, f"{KIRBY}/tests/smoke/smoke_kirby_target_scaling.py"]),
]


def run(cmd):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=os.path.dirname(cmd[1]))
    return p.returncode, (p.stdout + p.stderr)


ok = True
print(f"{'fix':52} {'check exits':>12}")
print("-" * 70)
for name, path, present, broken, cmd in CASES:
    src = open(path).read()
    if src.count(present) != 1:
        print(f"{name:52} {'SETUP FAIL':>12}  (anchor appears {src.count(present)}x)")
        ok = False
        continue
    open(path, "w").write(src.replace(present, broken))
    try:
        code, out = run(cmd)
    finally:
        open(path, "w").write(src)
    verdict = "RED (good)" if code != 0 else "GREEN (BAD)"
    if code == 0:
        ok = False
    print(f"{name:52} {verdict:>12}")
    if code == 0:
        print("      the check passed with the fix removed -- it guards nothing")

print("-" * 70)
print("every check goes red when its fix is removed" if ok
      else "SOME CHECKS DO NOT GUARD THEIR FIX")
sys.exit(0 if ok else 1)
