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

Since 2026-08-29 every check is also run on the UNTOUCHED tree first, and a check
that is already red there is reported as a DEAD CHECK rather than as a guard.
Without that pass, a test that can never pass -- one importing a function that
was deleted, say -- prints "RED (good)" and is counted as protecting its fix. It
runs each distinct check once and reuses the result, so the cost is roughly one
extra pass over the checks.

Add a case here whenever a fix lands. The four fields after the name are the
file, the text that must be present, what to replace it with, and the check to
run.
"""
import subprocess, sys, os, shutil, tempfile

QSAR = "/Users/apunt/repos/qsar_qm_models"
KIRBY = "/Users/apunt/repos/KIRBy"
NOISE = "/Users/apunt/repos/NoiseInject"

CASES = [
 ("censoring runs its own clean level",
  f"{QSAR}/slurm_scripts_qm9_rerun/generate_scripts.py",
  "NEEDS_OWN_CLEAN_LEVEL = {'censoring'}",
  "NEEDS_OWN_CLEAN_LEVEL = set()",
  [sys.executable, f"{QSAR}/slurm_scripts_qm9_rerun/test_generate_scripts.py",
   "--skip-parser"]),

 ("the two pipelines join on one set of model names",
  f"{QSAR}/model_names.json",
  '"GP-Hetero": "het_gp_rbf",',
  '"GP-Hetero-REMOVED": "het_gp_rbf",',
  [sys.executable, f"{QSAR}/scripts/test_model_names.py"]),

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

 # Break the delivery of a tuned parameter: the forest stops passing the dict to
 # the estimator and falls back to the shared defaults. The value would still be
 # in the file, and the row would still say params_source='tuned'.
 ("a tuned value reaches the model it was written for",
  f"{QSAR}/models/models.py",
  "        model = RandomForestRegressor(random_state=iteration_seed, **params)",
  "        model = RandomForestRegressor(random_state=iteration_seed, **sklearn_params('rf'))",
  [sys.executable, f"{QSAR}/scripts/test_tuned_params_reach_models.py"]),

 # Break the roster the tuned file is checked against: a representation that is
 # not in the study becomes acceptable, which is how the file on disk came to be
 # keyed by pdv, smiles, randomized_smiles and graph.
 ("the tuned file is checked against the real roster",
  f"{QSAR}/models/tuning_rosters.py",
  "MODELS = _GEN.MODELS\nALL_REPS = list(_GEN.ALL_REPS)",
  "MODELS = _GEN.MODELS\nALL_REPS = list(_GEN.ALL_REPS) + ['pdv', 'smiles', 'randomized_smiles', 'graph']",
  [sys.executable, f"{QSAR}/scripts/test_tuning_rosters.py"]),

 # Drop the flag from the generated command, which is where it was until
 # 2026-08-28: every reader of the tuned files sits behind it, so the grid
 # trains on the shared defaults no matter what the sweep wrote.
 #
 # The line moved on 2026-08-28. The template no longer passes the literal: it
 # decides ONCE at task start whether the two tuned files exist and passes
 # $TUNED_FLAG, so a task is entirely tuned or entirely default rather than
 # flipping between noise levels mid-curve. This entry followed it -- pointed at
 # the old literal it reported the fix as already removed, on every run, which is
 # how a guard stops being read.
 ("a generated job script can deliver a tuned value",
  f"{QSAR}/slurm_scripts_qm9_rerun/generate_scripts.py",
  "    $TUNED_FLAG \\\\\n",
  "",
  [sys.executable, f"{QSAR}/scripts/test_tuned_params_reach_the_cluster.py"]),

 # Unpin the base learner's depth, which is where it was until 2026-08-28: the
 # library's default, not ours, and free to move under an upgrade.
 ("NGBoost's base learner is the spec's, not the library's",
  f"{QSAR}/models/model_defaults.py",
  "        'base_max_depth': 3,",
  "        'base_max_depth': 7,",
  [sys.executable, f"{QSAR}/scripts/test_ngboost_stage_selection.py"]),

 # Put the base loss back after the Bayesian transformation, which is where
 # NN-beta had it: the ELBO wrapper is discarded and the Bayesian variants train
 # on plain MSE with no KL term.
 ("NN-beta keeps the loss its transformation wrapped",
  f"{QSAR}/models/models.py",
  # Anchor stops BEFORE the closing bracket on purpose: the optimizer call
  # gained a weight_decay argument on 2026-08-29 and the longer anchor stopped
  # matching the same day, which is a guard silently switching itself off. The
  # short form is still unique in the file and survives arguments being added.
  "    model.to(device)\n\n    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr']",
  "    model.to(device)\n\n    criterion = get_loss_function(loss_name, **loss_kwargs)\n\n    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr']",
  [sys.executable, f"{QSAR}/scripts/test_bnn_criterion_order.py"]),

 # Put the target back where the solved scale goes, in the one condition that
 # bypassed the solver. Gaussian is unaffected, so the check only goes red on the
 # shape axis the registry does not carry.
 ("grouped-shifted takes the solved scale, not the target",
  f"{NOISE}/noiseInject/core.py",
  "            epsilon = self._draw_grouped_shifted(y, solved, groups, **params)",
  "            epsilon = self._draw_grouped_shifted(y, dose, groups, **params)",
  [sys.executable, f"{NOISE}/tests/test_noiseinject.py", "-k", "dose_is_flat"]),

 # Record the delivered amount and never look at it -- which is how the line
 # above stayed wrong through every run in the project.
 ("the Python injector checks what it delivered",
  f"{NOISE}/noiseInject/core.py",
  "        if abs(realised / dose - 1.0) > tol:\n            warnings.warn(",
  "        if False:\n            warnings.warn(",
  [sys.executable, f"{NOISE}/tests/test_noiseinject.py", "-k",
   "delivered_dose_is_checked"]),

 # The Rust half of the same defect: divide the two components down to unit
 # variance while the solver still puts the shape's spread into G, and the
 # pipeline delivers the target divided by that spread.
 ("the two injectors agree on grouped-shifted",
  f"{QSAR}/rust/src/main.rs",
  "                let b = spec.shape.draw(&mut rng);\n                group_offsets.insert(*gid, b);",
  "                let b = spec.shape.draw(&mut rng) / spec.shape.unit_sd();\n                group_offsets.insert(*gid, b);",
  [sys.executable, f"{QSAR}/scripts/test_noise_arms.py"]),

 # Swallow the injector's dose failure the way an ordinary model failure is
 # swallowed: the cell vanishes as one printed line and the job finishes green
 # having reported a noise level it never received. The injector warns rather
 # than raising today, so this guards the route rather than a live failure.
 ("a fatal injection failure reaches the surface",
  f"{KIRBY}/tests/alternative_data_noise_robustness.py",
  "                except (RunIntegrityError, DoseError):",
  "                except (RunIntegrityError,):",
  [sys.executable, f"{KIRBY}/tests/smoke/smoke_kirby_dose_error.py"]),

 # Name a condition the way the two implementations used to name it separately.
 # The old fallback also collapsed 1%, 5% and 10% contamination onto one string,
 # and nothing else in a results row carries the fraction.
 ("one condition has one name on both sides",
  f"{NOISE}/noiseInject/core.py",
  "        return base if self.distribution == 'gaussian' else f\"{base}_{shape}\"",
  "        return f\"{self.strategy}/{self.distribution}\"",
  [sys.executable, f"{QSAR}/scripts/test_condition_names.py"]),

 # The command line took `grouped_wide` while every results row said
 # `grouped_wider`, so the name read off a row killed the run. Rebuilds first --
 # the check runs the shipped binary, so an edited source with a stale binary
 # would pass and prove nothing.
 ("the command line takes the name the rows carry",
  f"{QSAR}/rust/src/main.rs",
  "        \"grouped_wide\" | \"grouped_wider\" => {",
  "        \"grouped_wide\" => {",
  [sys.executable, f"{QSAR}/scripts/test_condition_names.py", "--rebuild"]),

 # The driver's own parser refuses an unlisted choice before the value reaches
 # the injector, so the Rust half of that fix is not enough on its own.
 ("the driver takes the name the rows carry",
  f"{QSAR}/scripts/process_and_train.py",
  "                        choices=[\"uniform\", \"grouped_wide\", \"grouped_wider\",\n"
  "                                 \"grouped_shift\", \"grouped_shifted\", \"outlier\",\n"
  "                                 \"censoring\"],",
  "                        choices=[\"uniform\", \"grouped_wide\", \"grouped_shift\", "
  "\"outlier\", \"censoring\"],",
  [sys.executable, f"{QSAR}/scripts/test_generated_job_flags.py"]),

 # Slice the predicted variance off again, which is what every prediction site
 # used to do. The network still trains on the heteroscedastic loss; only the
 # second output stops reaching the decomposition, and the aleatoric column comes
 # out blank.
 # Re-anchored 2026-08-28. The old anchor named a prediction loop that has since
 # been rewritten, so this reported SETUP FAIL and the guard was UNVERIFIED, not
 # passing. The break is the original defect: slice the second output off and keep
 # the first, so the fitted per-molecule variance reaches no file.
 ("the head's predicted variance reaches the file",
  f"{QSAR}/models/models.py",
  "            output, variance = split_predictive_head(output, loss_name)\n"
  "            means.append(np.asarray(output, dtype=float).reshape(-1))\n"
  "            if variance is not None:\n"
  "                head_vars.append(np.asarray(variance, dtype=float).reshape(-1))",
  "            output, _ = split_predictive_head(output, loss_name)\n"
  "            means.append(np.asarray(output, dtype=float).reshape(-1))",
  [sys.executable, f"{QSAR}/scripts/test_predictive_head.py"]),

 # Leave the error columns in label standard deviations, which is what they were
 # while the same two columns on the experimental side were in log units.
 ("rmse and mae are in the label's own units",
  f"{QSAR}/scripts/utils.py",
  "    _, sd = current_standardisation()\n"
  "    if sd is not None and float(sd) > 0:\n"
  "        sd = float(sd)\n"
  "        mae, mse, rmse = mae * sd, mse * sd * sd, rmse * sd",
  "    pass  # BROKEN ON PURPOSE",
  [sys.executable, f"{QSAR}/scripts/test_metric_units.py"]),

 # Put the dispatch branch back without the refusal, which is the state the file
 # was in before 2026-08-28: `-m conformal_hetero` runs, falls off the end of the
 # elif chain, returns None, and the run exits 0 having written no row.
 ("conformal is out and asking for it stops the run",
  f"{QSAR}/scripts/process_and_train.py",
  "    CONFORMAL_MODELS = ('conformal', 'conformal_hetero')",
  "    CONFORMAL_MODELS = ()",
  [sys.executable, f"{QSAR}/scripts/test_conformal_is_out.py"]),

 # Put the affected molecules back on the level-dependent seed. Who gets damaged
 # then changes at every point of a condition's own degradation curve, and the
 # level-free column the clean-run subtraction rests on describes a different set of
 # molecules at every level (RERUN_PLAN.md 2.26a).
 ("the affected molecules are chosen level-free, in the training script",
  f"{QSAR}/scripts/process_and_train.py",
  "    return shape_seed, int(iteration_seed)",
  "    return shape_seed, int(shape_seed)",
  [sys.executable, f"{QSAR}/scripts/test_injector_wiring.py"]),

 # ...and the same fix in the injector: the selection draw goes back on the stream
 # the caller varies per level.
 ("the affected molecules are chosen level-free, in the injector",
  f"{QSAR}/rust/src/main.rs",
  "    let mut sel_rng = StdRng::seed_from_u64(spec.selection_seed ^ 0x5CA1E);",
  "    let mut sel_rng = StdRng::seed_from_u64(spec.seed ^ 0x5CA1E);",
  ["cargo", "test", "--release", "--test", "noise_gates",
   "the_selection_seed_is_what_decides_who_gets_hit"]),

 # Rename QM9's replicate axis to `fold`, which is how the two axes would merge:
 # one reader then concatenates ten resamples and five overlapping partitions into
 # a single spread and calls it an error bar.
 # Re-anchored 2026-08-28. `"noise_type",` was unique when this was written and now
 # sits in two column lists, so the harness reported SETUP FAIL and the guard was
 # UNVERIFIED. It anchors on the results-row list by name instead, which is the list
 # the check actually reads.
 ("a replicate is QM9's and a fold is the other three datasets'",
  f"{QSAR}/scripts/utils.py",
  'RESULT_COLUMNS = ["sigma", "iteration", "model",',
  'RESULT_COLUMNS = ["sigma", "iteration", "fold", "model",',
  [sys.executable, f"{QSAR}/scripts/test_replicate_is_not_a_fold.py"]),
]


def run(cmd):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    # Every case but one is a script path, and its own directory is the right place
    # to run it from. cargo is the exception: it has to run inside the crate.
    cwd = os.path.join(QSAR, "rust") if cmd[0] == "cargo" else os.path.dirname(cmd[1])
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
    return p.returncode, (p.stdout + p.stderr)


# RED with the fix removed only means something if the check was GREEN with the
# fix present.
#
# scripts/test_predictive_head.py exits 1 on an untouched tree -- it imports
# decompose_uncertainty_sampling from utils, which was deleted and folded into
# scripts/uncertainty_decomposition.py. It is red with the fix and red without
# it, so the moment its anchor was repaired this harness would have printed
# "RED (good)" for a test that cannot pass: a false all-clear, which is the one
# failure mode the harness exists to prevent. It was only visible on 2026-08-28
# because the anchor happened to be rotten as well.
#
# So each check is run on the untouched tree first. Several cases share a check,
# so the result is cached and each distinct command runs once.
CLEAN_RESULTS = {}


def run_clean(cmd):
    key = tuple(cmd)
    if key not in CLEAN_RESULTS:
        CLEAN_RESULTS[key] = run(cmd)
    return CLEAN_RESULTS[key]


# A killed run leaves a broken file behind, and it is invisible.
#
# `finally` puts the file back when the check fails, raises or is interrupted. It
# does NOT run when the process is killed outright, which is what happens when a
# run is stopped from outside. Twice on 2026-08-27 that left `utils.py` carrying a
# save_results that raises NotImplementedError, and `models.py` with the predicted
# variance sliced off again -- neither committed, both live, and nothing said so.
#
# So the original is copied out before the file is touched, and the copy is
# deleted only once the file is back. A copy still sitting here at start-up is a
# previous run that was killed: put the file back and refuse to start, because a
# run that begins from a broken file would restore it to the broken version and
# bake the damage in.
BACKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          ".harness_unrestored")


def _backup_path(path):
    return os.path.join(BACKUP_DIR, path.replace(os.sep, "__"))


def recover_from_a_killed_run():
    if not os.path.isdir(BACKUP_DIR):
        return
    left = sorted(os.listdir(BACKUP_DIR))
    if not left:
        return
    for name in left:
        original = os.path.join(BACKUP_DIR, name)
        target = "/" + name.replace("__", os.sep).lstrip(os.sep)
        shutil.copyfile(original, target)
        print(f"  put back: {target}")
        os.remove(original)
    sys.exit(
        f"\n{len(left)} file(s) were left broken by a run that was killed, and "
        f"have been put back. Check `git diff` on them, then run this again.")


os.makedirs(BACKUP_DIR, exist_ok=True)
recover_from_a_killed_run()

ok = True
print(f"{'fix':52} {'check exits':>12}")
print("-" * 70)
for name, path, present, broken, cmd in CASES:
    src = open(path).read()
    if src.count(present) != 1:
        print(f"{name:52} {'SETUP FAIL':>12}  (anchor appears {src.count(present)}x)")
        ok = False
        continue
    clean_code, clean_out = run_clean(cmd)
    if clean_code != 0:
        print(f"{name:52} {'DEAD CHECK':>12}  (already red with the fix PRESENT)")
        print("      this check cannot pass, so it cannot prove anything about the")
        print("      fix. Last line of its output:")
        print(f"      {clean_out.strip().splitlines()[-1][:150] if clean_out.strip() else '(no output)'}")
        ok = False
        continue
    backup = _backup_path(path)
    with open(backup, "w") as f:
        f.write(src)
    open(path, "w").write(src.replace(present, broken))
    try:
        code, out = run(cmd)
    finally:
        open(path, "w").write(src)
        os.remove(backup)
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
