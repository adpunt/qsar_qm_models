#!/usr/bin/env python
"""Does this interpreter actually have everything the job is about to ask for?

Written for RERUN_PLAN.md section 2.8d. Two Gaussian-process jobs (12822693,
12822694) ran to completion on 2026-08-19 and produced nothing: the interpreter
was missing gpytorch, the experiment list came out empty, five folds looped over
nothing, and the job then crashed reading its own empty output. Eight and six
minutes of queue time to learn that a package was absent.

This script answers the question before the queue does. It names the interpreter
it is speaking for, prints the version of everything the roster needs, and then
CONSTRUCTS each model rather than merely importing its package -- because a
version clash (quantile_forest against a newer scikit-learn, say) imports
cleanly and fails on contact.

Usage
-----
    python check_environment.py                     # the whole QM9 roster
    python check_environment.py --models lgb rf     # only what this job runs
    python check_environment.py --validation        # the KIRBy roster too

Exit status is 0 only if everything asked for can be built.
"""

import argparse
import importlib
import importlib.util
import os
import sys

# The QM9 roster, as the -m flag of process_and_train.py spells it, mapped to
# what has to be importable and constructible for that model to run.
QM9_MODELS = {
    "rf": ("sklearn.ensemble", "RandomForestRegressor"),
    "qrf": ("quantile_forest", "RandomForestQuantileRegressor"),
    "svm": ("sklearn.svm", "SVR"),
    "xgboost": ("xgboost", "XGBRegressor"),
    "lgb": ("lightgbm", "LGBMRegressor"),
    "ngboost": ("ngboost", "NGBRegressor"),
    "gauche": ("gpytorch", None),
    "gauche_rbf": ("gpytorch", None),
    "dnn": ("torch", None),
    "flexible_dnn": ("torch", None),
    "mlp": ("torch", None),
    "dnn_bnn_full": ("torchbnn", None),
    "dnn_bnn_last": ("torchbnn", None),
    "mlp_bnn_full": ("torchbnn", None),
    "mlp_bnn_last": ("torchbnn", None),
    "dnn_vbll": ("torch", None),
    "mlp_vbll": ("torch", None),
    # The job-script spelling for the two variational models
    # (slurm_scripts_qm9_rerun/generate_scripts.py MODELS).
    "dnn_bnn_full_variational": ("torchbnn", None),
    "mlp_bnn_full_variational": ("torchbnn", None),
    "conformal": ("torchcp", None),
    "graph": ("torch_geometric", None),
}

# Packages worth reporting a version for whatever was asked.
VERSION_REPORT = [
    "numpy", "pandas", "scipy", "scikit-learn", "torch", "torch_geometric",
    "rdkit", "xgboost", "lightgbm", "ngboost", "quantile_forest", "gpytorch",
    "gauche", "botorch", "torchbnn", "torchhk", "torchcp",
]

# The compiled companions of torch_geometric. Present-but-unloadable is the one
# failure whose own message does not say what to do about it.
PYG_COMPANIONS = ["torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]


# Loader failures come in two flavours and they need opposite responses.
#   "undefined symbol" / "Symbol not found"  -> the extension was compiled
#       against a different libtorch. The package is wrong for this
#       environment and removing it is the fix.
#   "failed to map segment" / "cannot allocate memory" -> the loader could not
#       mmap the file. The package is FINE; the machine would not give it
#       address space. On a login node this is routine for a CUDA torch build,
#       where libtorch_cuda.so and libcublasLt.so are over a gigabyte between
#       them. Uninstalling here would break a working environment.
_ABI_MARKERS = ("undefined symbol", "symbol not found")
_RESOURCE_MARKERS = (
    "failed to map segment",
    "cannot allocate memory",
    "cannot enable executable stack",
    "cannot allocate version reference table",
)


def classify_loader_error(text):
    """'abi', 'resource', or None if it is neither."""
    low = str(text).lower()
    if any(m in low for m in _RESOURCE_MARKERS):
        return "resource"
    if any(m in low for m in _ABI_MARKERS):
        return "abi"
    return None


def _version(name):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        pass
    try:
        return getattr(importlib.import_module(name), "__version__", "?")
    except Exception:
        return None


def report_versions():
    print(f"interpreter : {sys.executable}")
    print(f"python      : {sys.version.split()[0]}")
    print(f"cwd         : {os.getcwd()}")
    print()
    print("packages")
    for name in VERSION_REPORT:
        v = _version(name) or _version(name.replace("_", "-"))
        print(f"  {name:<20s} {v if v else '-- ABSENT --'}")
    print()


def check_pyg_companions():
    """Can the torch_geometric companion packages load, and if not, why not.

    An ABI mismatch here is fatal in a way that is easy to miss:
    torch_geometric.typing catches it and disables the package, but
    nn/conv/gravnet_conv.py catches only ImportError, and a broken shared object
    raises OSError -- so `import torch_geometric` dies and takes the whole QM9
    pipeline with it. Removing the packages turns the OSError into the
    ImportError gravnet_conv already handles, and nothing here uses their
    operators.

    But a package that cannot be MAPPED is a different thing entirely and must
    not get the same advice. See classify_loader_error.
    """
    abi, resource = [], []
    for name in PYG_COMPANIONS:
        if importlib.util.find_spec(name) is None:
            continue  # absent is fine, and is the state we want
        try:
            importlib.import_module(name)
        except Exception as e:
            kind = classify_loader_error(e)
            entry = (name, f"{type(e).__name__}: {str(e).splitlines()[0]}")
            (resource if kind == "resource" else abi).append(entry)

    if resource:
        print("WARN: torch_geometric companions could not be loaded, but the environment")
        print("      is not the problem -- the loader could not map them into memory.")
        for name, err in resource:
            print(f"      {name}: {err}")
        print()
        print("      Do NOT uninstall anything on the strength of this. Re-run inside a")
        print("      job allocation, where there is address space for a CUDA torch build:")
        print("        srun --account=<acct> --partition=short --mem=32G --pty \\")
        print("             python scripts/check_environment.py")
        print()

    if abi:
        print("FAIL: torch_geometric companion packages are installed but cannot load.")
        print("      They were built against a different libtorch than the installed torch.")
        for name, err in abi:
            print(f"      {name}: {err}")
        print()
        print("      This project uses none of the operators they provide. Remove them:")
        print("        python -m pip uninstall -y " + " ".join(n.replace("_", "-") for n in PYG_COMPANIONS))
        print()

    return not abi


# Models that import and construct cleanly and then fail on contact. The
# quantile forest against a newer scikit-learn is the live one: fit() raises
# "Invalid parameter 'monotonic_cst'" (slurm_scripts_uncertainty_rerun/
# preflight.sh records it). Constructing is not proof; fitting is.
def _fit_qrf():
    from quantile_forest import RandomForestQuantileRegressor
    X, y = _toy()
    m = RandomForestQuantileRegressor(n_estimators=5, random_state=42).fit(X, y)
    q = m.predict(X[:5], quantiles=[0.16, 0.5, 0.84])
    assert q.shape[1] == 3


def _fit_ngboost():
    from ngboost import NGBRegressor
    X, y = _toy()
    m = NGBRegressor(n_estimators=10, verbose=False, random_state=42).fit(X, y)
    assert len(m.pred_dist(X[:5]).scale) == 5


FIT_PROBES = {"qrf": _fit_qrf, "ngboost": _fit_ngboost}


def _toy():
    from sklearn.datasets import make_regression
    return make_regression(n_samples=120, n_features=8, noise=0.3, random_state=0)


def check_declared_requirements(failures, models):
    """Packages whose own declared requirements are not satisfied here.

    pip does not re-check this after the fact, so an environment can sit for
    months with a package installed against a dependency version it does not
    support. That is the state that produced the quantile_forest failure:
    quantile-forest 1.4.1 declares scikit-learn>=1.5 against an installed 1.3.2,
    and it imports and constructs perfectly before failing inside fit().
    """
    try:
        import importlib.metadata as md
        from packaging.requirements import Requirement
        from packaging.version import Version
    except Exception as e:
        print(f"requirement consistency: cannot check ({e})")
        print()
        return

    # Only the packages the requested models actually need. An `rf` job does not
    # care that ngboost declares a scikit-learn it has not got; a job that runs
    # ngboost cares very much.
    # Module root -> the distribution that ships it, where they differ.
    dist_of = {"sklearn": "scikit-learn"}
    relevant = set()
    for m in models:
        if m in QM9_MODELS:
            root = QM9_MODELS[m][0].split(".")[0]
            relevant.add(dist_of.get(root, root))
    relevant |= {"torch", "numpy", "scipy", "scikit-learn"}
    names = [n for n in VERSION_REPORT
             if n in relevant or n.replace("-", "_") in relevant]

    print(f"declared requirements ({', '.join(sorted(names))})")
    bad = []
    for name in names:
        for dist in (name, name.replace("_", "-")):
            try:
                reqs = md.requires(dist) or []
            except md.PackageNotFoundError:
                continue
            for raw in reqs:
                req = Requirement(raw)
                if req.marker is not None and not req.marker.evaluate():
                    continue
                try:
                    have = md.version(req.name)
                except md.PackageNotFoundError:
                    continue
                try:
                    ok = req.specifier.contains(Version(have), prereleases=True)
                except Exception:
                    continue
                if not ok:
                    bad.append(f"{dist} needs {req.name}{req.specifier}, installed {have}")
            break

    if bad:
        for line in bad:
            print(f"  FAIL  {line}")
        failures.append("unsatisfied declared requirements")
    else:
        print("  OK    every declared requirement is satisfied")
    print()


def probe(name, fn, failures, resource_failures=None):
    try:
        fn()
        print(f"  OK    {name}")
    except Exception as e:
        first = str(e).splitlines()[0]
        if classify_loader_error(e) == "resource" and resource_failures is not None:
            # Not a verdict on the environment: the machine would not give the
            # library address space. Counted separately so a login-node run does
            # not read as sixteen broken models.
            print(f"  ---   {name}: could not be mapped into memory ({first})")
            resource_failures.append(name)
            return
        print(f"  FAIL  {name}: {type(e).__name__}: {first}")
        failures.append(name)


def check_qm9(models, failures, resource_failures):
    print("QM9 roster (models/models.py imports every backend unguarded, so this")
    print("also proves the module itself is importable in this interpreter)")

    unknown = [m for m in models if m not in QM9_MODELS]
    if unknown:
        print(f"  FAIL  unknown model name(s): {sorted(unknown)}")
        print(f"        known: {sorted(QM9_MODELS)}")
        failures.append("unknown model names")
        models = [m for m in models if m in QM9_MODELS]

    for model in sorted(set(models)):
        module_name, attr = QM9_MODELS[model]

        def _build(module_name=module_name, attr=attr):
            mod = importlib.import_module(module_name)
            if attr is not None:
                cls = getattr(mod, attr)
                cls()  # constructed, not merely imported

        probe(f"{model:<16s} ({module_name})", _build, failures, resource_failures)

        if model in FIT_PROBES:
            probe(f"{model:<16s} fits", FIT_PROBES[model], failures, resource_failures)
    print()


def check_validation(failures):
    """The KIRBy roster, when that checkout is next to this one."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "KIRBy", "tests"),
        "/data/stat-cadd/scat9264/KIRBy/tests",
        "/data/stat-ecr/scat9264/KIRBy/tests",
    ]
    path = next((p for p in candidates if os.path.isdir(p)), None)
    if path is None:
        print("validation roster: no KIRBy checkout found, skipping")
        print()
        return

    print(f"validation roster ({path})")
    target = os.path.join(path, "alternative_data_noise_robustness.py")
    if not os.path.isfile(target):
        print("  FAIL  alternative_data_noise_robustness.py is missing")
        failures.append("KIRBy pipeline missing")
        print()
        return

    # Read the optional-import flags without executing the module's main path.
    def _flags():
        spec = importlib.util.spec_from_file_location("_kirby_probe", target)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, path)
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path.remove(path)
        off = [n for n in dir(mod)
               if n.startswith("HAS_") and getattr(mod, n) is False]
        if off:
            raise RuntimeError(
                f"optional backends unavailable in this interpreter: {sorted(off)} "
                f"— models needing them would be silently dropped from the run"
            )

    probe("every optional backend importable", _flags, failures)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Only check these models. Default: the whole QM9 roster.")
    ap.add_argument("--validation", action="store_true",
                    help="Also check the KIRBy validation roster.")
    args = ap.parse_args()

    report_versions()

    failures = []
    resource_failures = []
    if not check_pyg_companions():
        failures.append("torch_geometric companions")

    requested = args.models if args.models else sorted(QM9_MODELS)
    check_declared_requirements(failures, requested)

    check_qm9(requested, failures, resource_failures)

    if args.validation:
        check_validation(failures)

    if resource_failures:
        print(f"INCONCLUSIVE: {len(resource_failures)} model(s) could not be loaded because")
        print("the loader ran out of address space, not because anything is missing:")
        print("  " + ", ".join(sorted(resource_failures)))
        print()
        print("This is what a CUDA build of torch does on a LOGIN NODE -- libtorch_cuda.so")
        print("and libcublasLt.so are over a gigabyte between them, and login nodes cap")
        print("memory per user. It says nothing about whether the job would work.")
        print("Re-run inside an allocation, which is where the job scripts run it anyway:")
        print("  srun --account=<acct> --partition=short --mem=32G --pty \\")
        print("       python scripts/check_environment.py")
        print()

    if failures:
        print(f"FAIL: {len(failures)} check(s) failed in {sys.executable}")
        print("Do NOT submit jobs against this interpreter until they are fixed.")
        return 1

    if resource_failures:
        # Deliberately non-zero: nothing is known to be broken, but nothing is
        # confirmed working either, and a preflight must not report a pass it
        # did not observe.
        print(f"NOT PROVEN in {sys.executable} -- rerun in an allocation (see above).")
        return 3

    print(f"OK: everything requested can be constructed in {sys.executable}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
