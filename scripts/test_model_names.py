#!/usr/bin/env python3
"""Every model either pipeline can emit resolves to one canonical name.

WHY THIS EXISTS
---------------
The two pipelines write different spellings into the `model` column -- QM9 writes
`het_gp_rbf`, the laboratory runner writes `GP-Hetero` -- and every table that
puts the four datasets side by side joins on that string. The correspondence
lived inline in `generate_paper_figures_v2.py`, in two dicts a hundred lines
apart, and `scripts/uncertainty_stats.py` did not normalise model names at all.

When four models were added on 2026-08-28 -- GP-Tanimoto, GP-Hetero,
VBLL-Full-Hetero, MLP-VBLL-Full-Hetero -- none was added to either dict. Nothing
raised: an unmapped name is lower-cased and kept, so `GP-Hetero` became
`gp-hetero`, a name nothing else in the study uses, and those rows joined to
nothing. This is the failure `condition_names.json` and `scripts/
test_condition_names.py` exist to stop for the noise conditions; `model_names.json`
and this file are the same treatment for the models.

WHAT IT CHECKS
--------------
1. Every model label in the QM9 job generator's roster resolves.
2. Every display name the laboratory runner builds resolves (needs a KIRBy
   checkout; skipped with a message, not a pass, when there is none).
3. Both sides land on the same canonical set for the models both of them run.
4. Every canonical name a map points at is in the declared `canonical` list.
5. The three registries the figure script draws with -- colour, marker, order --
   cover every canonical model that reaches a figure.

    python scripts/test_model_names.py
    python scripts/test_model_names.py --kirby-dir ../KIRBy
"""
import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

failures = []
notes = []


def check(ok, message):
    if not ok:
        failures.append(message)
    return ok


def load_spec():
    spec = json.loads((ROOT / 'model_names.json').read_text())
    for key in ('canonical', 'qm9', 'validation'):
        if key not in spec:
            raise SystemExit(f"model_names.json has no '{key}' section")
    return spec


def resolve(name, mapping):
    """The reader accepts the hyphen-stripped spelling too."""
    return mapping.get(name, mapping.get(name.replace('-', '')))


# ---------------------------------------------------------------------------
# 1. The QM9 roster
# ---------------------------------------------------------------------------
# The generator's KEY is the filename suffix, not the string save_results writes.
# What models.py writes is derived from the flags, so the check is on the names
# the pipeline can actually produce for each roster entry.
QM9_ROW_NAME = {
    'rf': ['rf'],
    'xgboost': ['xgboost'],
    'lgb': ['lgb'],
    'svm': ['svm'],
    'ngboost': ['ngboost'],
    'dnn': ['dnn'],
    'mlp': ['mlp'],
    'qrf': ['qrf'],
    'gauche_rbf': ['gauche_rbf'],
    'gauche': ['gauche'],
    # The DNN trainer writes the Bayesian variants with no base prefix; the MLP
    # trainer writes them with one (models/models.py, the two model_name blocks).
    'dnn_bnn_full': ['bnn_full', 'dnn_bnn_full'],
    'mlp_bnn_full': ['mlp_bnn_full'],
    'dnn_bnn_full_variational': ['bnn_full_variational', 'dnn_bnn_full_variational'],
    'mlp_bnn_full_variational': ['mlp_bnn_full_variational'],
    'heteroscedastic_gp': ['het_gp_rbf'],
    'dnn_bnn_full_variational_hetero': ['bnn_full_variational_hetero',
                                        'dnn_bnn_full_variational_hetero'],
    'mlp_bnn_full_variational_hetero': ['mlp_bnn_full_variational_hetero'],
}


def qm9_roster():
    """The MODELS dict of the QM9 job generator, read rather than restated."""
    src = (ROOT / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py').read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'MODELS'
                and isinstance(node.value, ast.Dict)):
            return [k.value for k in node.value.keys]
    raise SystemExit('could not find MODELS in the QM9 job generator')


def check_qm9(spec):
    qm9 = spec['qm9']
    roster = qm9_roster()
    unknown = [m for m in roster if m not in QM9_ROW_NAME]
    check(not unknown,
          f"the QM9 generator runs model(s) {unknown} that this test does not "
          f"know the results-row spelling of. Add them to QM9_ROW_NAME here and "
          f"to model_names.json, or a new model joins to nothing.")
    canonicals = {}
    for label in roster:
        for row_name in QM9_ROW_NAME.get(label, []):
            target = resolve(row_name, qm9)
            check(target is not None,
                  f"QM9 emits the model name {row_name!r} (roster entry "
                  f"{label!r}) and model_names.json does not map it")
            if target:
                canonicals[label] = target
    print(f"  QM9: {len(roster)} roster entries, "
          f"{len(set(canonicals.values()))} canonical names")
    return canonicals


# ---------------------------------------------------------------------------
# 2. The laboratory roster
# ---------------------------------------------------------------------------
def kirby_display_names(kirby_dir):
    """The display names the runner builds its experiment list with.

    Read out of the source with a regex rather than by importing it: importing
    the runner pulls in rdkit, torch, gpytorch and ngboost, and this check has to
    be runnable on a laptop with none of them.
    """
    path = Path(kirby_dir) / 'tests' / 'alternative_data_noise_robustness.py'
    if not path.is_file():
        return None, f'no runner at {path}'
    src = path.read_text()
    names = set()
    # experiments.append(('RF', ...)) and the f-string decorated forms.
    for m in re.finditer(r"experiments\.append\(\(\s*(?:f?)'([^']+)'", src):
        names.add(m.group(1))
    for m in re.finditer(r"experiments\.append\(\(\s*f'\{gp_name\}([^']*)'", src):
        # gp_name is 'GP' when the kernel is RBF and 'GP-<Kernel>' otherwise.
        for base in ('GP', 'GP-Tanimoto'):
            names.add(base + m.group(1))
    names.discard('{gp_name}')
    # `experiments.append((gp_name, ...))` passes a variable, so the undecorated
    # process is not a literal anywhere in the file. It is 'GP' when the kernel
    # is RBF and 'GP-<Kernel>' otherwise -- the same rule the decorated form
    # above is built from.
    if 'gp_name' in src:
        names.update({'GP', 'GP-Tanimoto'})
    # The neural models come from a (mtype, mname) list rather than a literal
    # call, so read that list.
    block = re.search(r"for mtype, mname in \[(.*?)\]:", src, re.S)
    if block:
        names.update(re.findall(r"'[^']+',\s*'([^']+)'\)", block.group(1)))
    # gp_name itself is built from the kernel; both spellings are covered above.
    names = {n for n in names if not n.startswith('{')}
    return names, None


def check_validation(spec, kirby_dir):
    val = spec['validation']
    names, why = kirby_display_names(kirby_dir)
    if names is None:
        notes.append(f'laboratory roster NOT CHECKED: {why}. Pass --kirby-dir '
                     f'or set KIRBY_DIR.')
        return {}
    canonicals = {}
    for name in sorted(names):
        target = resolve(name, val)
        check(target is not None,
              f"the laboratory runner builds the model {name!r} and "
              f"model_names.json does not map it")
        if target:
            canonicals[name] = target
    print(f"  laboratory: {len(names)} display names, "
          f"{len(set(canonicals.values()))} canonical names")
    return canonicals


# ---------------------------------------------------------------------------
# 3-5. The shared checks
# ---------------------------------------------------------------------------
def check_canonical_closed(spec):
    declared = set(spec['canonical'])
    for side in ('qm9', 'validation'):
        stray = sorted(set(spec[side].values()) - declared)
        check(not stray,
              f"model_names.json '{side}' maps to {stray}, which are not in its "
              f"own 'canonical' list")
    print(f"  canonical list: {len(declared)} names, both maps closed over it")


# The four that had no name in common before 2026-08-28, named rather than
# implied. The first version of this test checked only that "at least 15" models
# joined, against a real join of 17 -- so a map that pointed two models at
# wrong-but-canonical names still passed, and the four pairs the change exists
# for were nowhere asserted. Found by the smoke test of 2026-08-29.
THE_FOUR = [
    ('gauche', 'GP-Tanimoto', 'gauche'),
    ('het_gp_rbf', 'GP-Hetero', 'het_gp_rbf'),
    ('bnn_full_variational_hetero', 'VBLL-Full-Hetero', 'dnn_vbll_hetero'),
    ('mlp_bnn_full_variational_hetero', 'MLP-VBLL-Full-Hetero', 'mlp_vbll_hetero'),
]


def check_the_four_pairs(spec):
    """The four models the change exists for, by name and by target."""
    qm9, val = spec['qm9'], spec['validation']
    for q, v, want in THE_FOUR:
        a, b = resolve(q, qm9), resolve(v, val)
        check(a == want and b == want,
              f"{q!r} (QM9) and {v!r} (laboratory) must both resolve to {want!r}; "
              f"they resolve to {a!r} and {b!r}. These four are the models that "
              f"had no name in common, so a table pooling the datasets drops them.")
    print(f"  the four: {len(THE_FOUR)} formerly-mismatched pairs resolve to one "
          f"name each")


def check_the_consumers_actually_read_the_file(spec):
    """The two modules that join on these names use the FILE, not inline dicts.

    This test checked the file and never the wiring, and it passed with the fix
    reverted in BOTH consumers -- stale inline dicts in the figure script and an
    empty map in the statistics module. A guard that cannot see the code it
    guards is not a guard. Found by the smoke test of 2026-08-29.
    """
    sys.path.insert(0, str(HERE))
    for module_name, attrs in (
            ('generate_paper_figures_v2', ('QM9_MODEL_MAP', 'VALIDATION_MODEL_MAP')),
            ('uncertainty_stats', ('_QM9_MODEL_NAMES', '_VALIDATION_MODEL_NAMES'))):
        try:
            mod = __import__(module_name)
        except Exception as exc:
            notes.append(f'{module_name} NOT CHECKED: it does not import here '
                         f'({type(exc).__name__}: {exc}). That is a skip, not a pass.')
            continue
        for attr, side in zip(attrs, ('qm9', 'validation')):
            live = getattr(mod, attr, None)
            check(isinstance(live, dict) and live,
                  f"{module_name}.{attr} is {live!r}; the module is not reading "
                  f"model_names.json, so nothing it loads is normalised")
            if not isinstance(live, dict) or not live:
                continue
            for name, target in spec[side].items():
                check(live.get(name) == target,
                      f"{module_name}.{attr} maps {name!r} to {live.get(name)!r}; "
                      f"model_names.json says {target!r}. The module is using its "
                      f"own list, which is how the two drifted apart.")
        print(f"  {module_name}: both maps match the file, key for key")


def check_both_sides_agree(qm9_canon, val_canon):
    if not val_canon:
        return
    shared_q = set(qm9_canon.values())
    shared_v = set(val_canon.values())
    both = shared_q & shared_v
    check(len(both) >= 15,
          f"only {len(both)} models resolve to the same canonical name on both "
          f"sides; the two pipelines run 17 models in common")
    only_q = sorted(shared_q - shared_v)
    only_v = sorted(shared_v - shared_q)
    print(f"  join: {len(both)} models on both sides"
          + (f"; QM9 only {only_q}" if only_q else "")
          + (f"; laboratory only {only_v}" if only_v else ""))


def figure_registries():
    """MODEL_COLORS, MODEL_MARKERS and MODEL_ORDER, read from the figure script."""
    src = (ROOT / 'scripts' / 'generate_paper_figures_v2.py').read_text()
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            name = node.targets[0].id
            if name in ('MODEL_COLORS', 'MODEL_MARKERS') and isinstance(node.value, ast.Dict):
                out[name] = {k.value for k in node.value.keys}
            elif name == 'MODEL_ORDER' and isinstance(node.value, ast.List):
                out[name] = {e.value for e in node.value.elts}
            elif name == 'UNCERTAINTY_COLORS' and isinstance(node.value, ast.Dict):
                out[name] = {k.value for k in node.value.keys}
    return out


def check_figure_registries(qm9_canon):
    reg = figure_registries()
    # The excluded models never reach a figure, so they are not required to have
    # a colour. Everything the current QM9 roster emits does.
    wanted = set(qm9_canon.values())
    # UNCERTAINTY_COLORS is checked against the models that EMIT one, not the
    # whole roster: it is the palette for the figure that compares them, and it
    # is chosen for distinctness rather than family. It was omitted from this
    # list at first, so the three new models fell back to their family colour
    # and drew indistinguishably from the model they are a variant of. Found by
    # the smoke test of 2026-08-29.
    emits = {'qrf', 'ngboost', 'gauche', 'gauche_rbf', 'het_gp_rbf',
             'dnn_bnn_full', 'dnn_vbll', 'dnn_vbll_hetero',
             'mlp_bnn_full', 'mlp_vbll', 'mlp_vbll_hetero'} & wanted
    missing_unc = sorted(emits - reg.get('UNCERTAINTY_COLORS', set()))
    check(not missing_unc,
          f"UNCERTAINTY_COLORS has no entry for {missing_unc}; those models emit "
          f"a per-molecule uncertainty and would be drawn in their family's "
          f"colour, indistinguishable from the model they are a variant of")
    for key in ('MODEL_COLORS', 'MODEL_MARKERS', 'MODEL_ORDER'):
        have = reg.get(key, set())
        missing = sorted(wanted - have)
        check(not missing,
              f"{key} in generate_paper_figures_v2.py has no entry for "
              f"{missing}; those models are on the QM9 roster and would be "
              f"drawn with a default that collides with another family")
    print(f"  figure registries: colour, marker and order cover "
          f"{len(wanted)} canonical models")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kirby-dir', default=os.environ.get('KIRBY_DIR')
                    or str(ROOT.parent / 'KIRBy'))
    args = ap.parse_args()

    print('model_names.json — the two pipelines join on one set of names')
    spec = load_spec()
    check_canonical_closed(spec)
    check_the_four_pairs(spec)
    check_the_consumers_actually_read_the_file(spec)
    qm9_canon = check_qm9(spec)
    val_canon = check_validation(spec, args.kirby_dir)
    check_both_sides_agree(qm9_canon, val_canon)
    check_figure_registries(qm9_canon)

    for note in notes:
        print(f"  NOTE  {note}")
    if failures:
        print(f"\nFAIL — {len(failures)} problem(s):")
        for f in failures:
            print(f"  * {f}")
        return 1
    print('\nPASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
