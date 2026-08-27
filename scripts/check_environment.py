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
    python check_environment.py --validation-models SVM LightGBM   # a KIRBy job's guard
    python check_environment.py --deep --validation # THE PREFLIGHT GATE

The last one is what has to pass before a launch. On top of constructing every
model it imports models/models.py for real, checks that env.yml describes the
interpreter it is speaking for, checks that noiseInject and kirby are
importable, counts the DISTINCT OpenMP runtime files a job would load, and then
runs the two failures that forced the 2026-08-27 environment rebuild -- a
LightGBM fit and a Gaussian-process fit with the boosting libraries already
loaded -- under both of the thread settings the two pipelines use. Both must
pass in the same environment: curing one at the other's expense is the trap.

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
    # The EXCLUDED_MODELS labels from generate_scripts.py. They are off by
    # default (--include-excluded turns them on) but the job template passes the
    # label straight through, so an unknown one would fail the task with
    # "unknown model name" -- a guard blocking a job for the guard's own reason.
    # --audit-roster keeps this list honest.
    "conformal_rf": ("torchcp", None),
    "conformal_qrf": ("torchcp", None),
    "conformal_dnn": ("torchcp", None),
    "dnn_bnn_variational": ("torchbnn", None),
    "mlp_bnn_variational": ("torchbnn", None),
}

# The validation and uncertainty rosters, as KIRBy's --models flag spells them.
#
# These are a SEPARATE namespace from QM9_MODELS: KIRBy says 'LightGBM' where
# process_and_train.py says 'lgb', and 'BNN-Full' where it says 'dnn_bnn_full'.
# Passing a KIRBy label to --models would fail with "unknown model name", which
# is the guard blocking a job for the guard's own reason -- so those two job
# families get --validation-models instead.
#
# Sources, checked rather than guessed: the optional-import blocks at
# alternative_data_noise_robustness.py:253-333 (one try/except per backend,
# each setting a HAS_* flag that silently drops the model when False) and
# UNCERTAINTY_MODELS at :182. VBLL has no external package -- VBLLLayer is
# written out in that file on top of torch -- so it maps to torch.
#
# The value is (modules, attr): EVERY module must import, and attr -- when it is
# not None -- is looked up on the first of them and constructed. GP needs three
# packages, and gauche or botorch missing is exactly the case that turned into a
# silent skip before.
VALIDATION_MODELS = {
    # the validation roster (slurm_scripts_validation_rerun MODELS_ALL)
    "RF": (("sklearn.ensemble",), "RandomForestRegressor"),
    "QRF": (("quantile_forest",), "RandomForestQuantileRegressor"),
    "SVM": (("sklearn.svm",), "SVR"),
    "XGBoost": (("xgboost",), "XGBRegressor"),
    "LightGBM": (("lightgbm",), "LGBMRegressor"),
    "NGBoost": (("ngboost",), "NGBRegressor"),
    "GP": (("gpytorch", "gauche", "botorch"), None),
    "DNN": (("torch",), None),
    # the uncertainty roster's additions (UNCERTAINTY_MODELS)
    "BNN-Full": (("torchbnn", "torchhk"), None),
    "MLP-BNN-Full": (("torchbnn", "torchhk"), None),
    "VBLL-Full": (("torch",), None),
    "MLP-VBLL-Full": (("torch",), None),
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


def check_pyg_companions(resource_failures):
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

    # Recorded where main can see it. Returning only `not abi` meant a companion
    # that could not be MAPPED left no trace: the script printed "OK: everything
    # requested can be constructed" and exited 0, which is exactly the pass-it-
    # did-not-observe that this file is supposed to refuse. Found by the audit.
    resource_failures.extend(f"torch_geometric companion: {name}" for name, _ in resource)

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
    # Both namespaces: KIRBy calls the quantile forest 'QRF' and QM9 calls it
    # 'qrf', and the quantile-forest requirement conflict this check exists for
    # breaks both runs identically.
    relevant = set()
    for m in models:
        if m in QM9_MODELS:
            mods = (QM9_MODELS[m][0],)
        elif m in VALIDATION_MODELS:
            mods = VALIDATION_MODELS[m][0]
        else:
            continue
        for mod in mods:
            root = mod.split(".")[0]
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


def _omp_family(basename):
    """Which OpenMP implementation a filename belongs to, by name alone."""
    if basename.startswith("libiomp"):
        return "Intel (libiomp5)"
    if basename.startswith("libgomp"):
        return "GNU (libgomp)"
    if basename.startswith("libomp"):
        return "LLVM (libomp)"
    return None


def _linked_libraries(binary):
    """(name, resolved path or None) for every shared library `binary` links."""
    import subprocess as _sp
    out = ""
    try:
        if sys.platform == "darwin":
            out = _sp.run(["otool", "-L", binary], capture_output=True,
                          text=True, timeout=30).stdout
        else:
            out = _sp.run(["ldd", binary], capture_output=True,
                          text=True, timeout=30).stdout
    except Exception:
        return []

    found = []
    for raw in out.splitlines():
        if sys.platform == "darwin" and not raw.startswith("\t"):
            continue                          # otool's header: the binary itself
        line = raw.strip().rstrip(":")
        if not line:
            continue
        if "=>" in line:                      # ldd: "libgomp.so.1 => /path (0x..)"
            name, _, rhs = line.partition("=>")
            path = rhs.strip().split(" ")[0]
            found.append((os.path.basename(name.strip()),
                          path if path.startswith("/") else None))
        else:                                 # otool, or ldd's static entries
            token = line.split(" ")[0]
            found.append((os.path.basename(token),
                          token if token.startswith("/") else None))
    return found


def _resolve_lib(name, path, roots):
    """Best real path for a library an extension links, or an unresolved marker.

    macOS records @rpath/libomp.dylib rather than a path, so a linked entry has
    to be looked for in the places the loader would look. Getting this right is
    what separates "three names for one conda file" from "three separate copies
    bundled by three wheels".
    """
    if path and os.path.exists(path):
        return os.path.realpath(path)
    base = os.path.basename(name)
    for d in roots:
        if not d:
            continue
        cand = os.path.join(d, base)
        if os.path.exists(cand):
            return os.path.realpath(cand)
    return f"unresolved:{base}"


def check_threading_runtimes(failures):
    """More than one OpenMP runtime FILE across the installed backends kills jobs.

    With two or more loaded in one process, one library's threads deadlock or
    crash inside the other's barrier, and which model dies depends on which
    package was imported first. Measured 2026-08-27: nothing first -> LightGBM
    crashes; LightGBM first -> the Gaussian process crashes; PyTorch first ->
    LightGBM crashes. No import order saves both (RERUN_PLAN.md 2.8e-bis).
    Neither KMP_DUPLICATE_LIB_OK=TRUE nor OMP_NUM_THREADS=1 cures it.

    It matters because of HOW it fails. In the full pipeline it presented as a
    hang: three hours at zero processor time, one result row, no error. On a
    cluster that consumes the whole allocation and the scheduler kills the
    task, so it writes no rows and no message.

    COUNT FILES, NOT NAMES. The two things a name-based check gets wrong are
    both live here:

      * conda-forge's llvm-openmp ships libgomp.so.1 AND libiomp5.so as
        SYMLINKS to libomp.so. Three names, one file, no conflict -- a
        name-based check fails a perfectly good environment.
      * two PyPI wheels each bundling their own private copy of libomp are two
        distinct files under one name. That is the actual defect, and a
        name-based check passes it.

    So every candidate is resolved to a real path and the distinct paths are
    counted. Where a path cannot be resolved (macOS @rpath entries), the name
    is kept as its own entry rather than assumed to be a duplicate.

    This inspects what is INSTALLED rather than what happens to be imported:
    models/models.py imports every backend at module scope, so every job loads
    every runtime whatever model it was asked for.
    """
    import glob as _glob
    print("threading runtimes (what a job would load, not what is loaded now)")

    # key (a real path, or "unresolved:<name>") -> {"who": set, "family": str}
    by_file = {}
    search_dirs = [os.path.join(sys.prefix, "lib"),
                   os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")]

    def _note(key, who, family):
        e = by_file.setdefault(key, {"who": set(), "family": family})
        e["who"].add(who)

    for pkg in ("torch", "lightgbm", "sklearn", "xgboost", "quantile_forest",
                "gpytorch", "functorch"):
        try:
            spec = importlib.util.find_spec(pkg)
        except Exception:
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        root = list(spec.submodule_search_locations)[0]

        # Copies shipped inside the package. The wheels hide these in
        # .dylibs / .libs, which glob will not descend into unless the dot is
        # named -- the first version of this check missed scikit-learn's for
        # exactly that reason.
        candidates = []
        for sub in ("", ".dylibs", ".libs", "lib"):
            base = os.path.join(root, sub) if sub else root
            candidates += _glob.glob(os.path.join(base, "*.dylib"))
            candidates += _glob.glob(os.path.join(base, "*.so*"))
        # A wheel's bundled copies sit in a sibling "<dist>.libs" directory,
        # NOT inside the package -- and the distribution name is not always the
        # import name (sklearn ships as scikit_learn.libs). Only this package's
        # own directories are scanned: an earlier version globbed all of
        # site-packages and reported every package as bundling every copy.
        parent = os.path.dirname(root)
        dist_dir = {"sklearn": "scikit_learn"}.get(pkg, pkg)
        for stem in {pkg, dist_dir}:
            candidates += _glob.glob(os.path.join(parent, f"{stem}.libs", "*.so*"))
            candidates += _glob.glob(os.path.join(parent, f"{stem}.libs", "*.dylib"))
        for hit in candidates:
            fam = _omp_family(os.path.basename(hit))
            if fam is None:
                continue
            _note(os.path.realpath(hit), f"{pkg} bundles {os.path.basename(hit)}", fam)

        # What its compiled extensions actually LINK against -- the only way to
        # see a runtime that lives outside the package.
        exts = (_glob.glob(os.path.join(root, "*.so"))
                + _glob.glob(os.path.join(root, "*.dylib"))
                + _glob.glob(os.path.join(root, "lib", "*.dylib"))
                + _glob.glob(os.path.join(root, "lib", "*.so")))
        pkg_dirs = [root, os.path.join(root, "lib"),
                    os.path.join(root, ".dylibs"), os.path.join(root, ".libs")]
        for ext in exts[:8]:
            for name, path in _linked_libraries(ext):
                fam = _omp_family(name)
                if fam is None:
                    continue
                _note(_resolve_lib(name, path, pkg_dirs + search_dirs),
                      f"{pkg} links {name}", fam)

    if not by_file:
        print("  OK    no OpenMP runtime detected in any installed backend")
        print()
        return

    if len(by_file) == 1:
        key, e = next(iter(by_file.items()))
        print(f"  OK    one threading runtime: {key}")
        print(f"          {e['family']}")
        for w in sorted(e["who"]):
            print(f"          {w}")
        print()
        return

    print(f"  FAIL  {len(by_file)} DISTINCT OpenMP runtime files are reachable from the "
          f"installed packages:")
    for key, e in sorted(by_file.items()):
        print(f"          {key}   [{e['family']}]")
        for w in sorted(e["who"]):
            print(f"            {w}")
    print("        Any job importing models/models.py loads all of them, because that")
    print("        file imports every backend at module scope. One model will then")
    print("        deadlock or crash inside another's thread barrier, and in this")
    print("        pipeline that presents as a HANG -- the task burns its whole")
    print("        allocation and writes no rows and no error.")
    print("        Pinning threads does NOT cure it (measured, RERUN_PLAN.md 2.8e-bis).")
    print("        Rebuild the environment: SETUP_REBUILD=1 . setup.sh")
    failures.append("multiple threading runtimes")
    print()


# The two failures that forced the rebuild. They must pass in ONE environment:
# curing either at the other's expense is the trap -- an "import lightgbm
# first" fix was committed and reverted on 2026-08-27 for exactly that.
#
# Both run in a SUBPROCESS on purpose. A segfault or a deadlock inside this
# interpreter would take the preflight down with it and report nothing.
_LGB_PROBE = """
import torch, lightgbm as lgb, numpy as np
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=400, n_features=512, random_state=0)
lgb.LGBMRegressor(n_estimators=15, verbose=-1).fit(X, y)
print('LightGBM OK')
"""

# Section 5 of scripts/server_audit.sh, verbatim in behaviour: import the
# boosting libraries first, then fit a plain ExactGP.
_GP_PROBE = """
import lightgbm, xgboost, numpy as np, torch, gpytorch
class G(gpytorch.models.ExactGP):
    def __init__(s, x, y, l):
        super().__init__(x, y, l)
        s.m = gpytorch.means.ConstantMean()
        s.c = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(s, x):
        return gpytorch.distributions.MultivariateNormal(s.m(x), s.c(x))
r = np.random.RandomState(0)
X = r.normal(size=(900, 208)); y = X[:, 0] * 2 + r.normal(scale=.3, size=900)
xt = torch.from_numpy(X); yt = torch.from_numpy(y)
l = gpytorch.likelihoods.GaussianLikelihood(noise=1e-3); m = G(xt, yt, l)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(l, m)
m.train(); l.train()
o = torch.optim.Adam(m.parameters(), lr=0.1)
for _ in range(20):
    o.zero_grad(); (-mll(m(xt), yt)).backward(); o.step()
print('GP OK')
"""


def _run_probe(label, code, env_overrides, failures, timeout=600):
    """Run `code` in a fresh interpreter and classify how it ended."""
    import subprocess as _sp
    env = dict(os.environ)
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    try:
        r = _sp.run([sys.executable, "-c", code], capture_output=True,
                    text=True, timeout=timeout, env=env)
    except _sp.TimeoutExpired:
        # This is the shape that costs a whole allocation: no crash, no output,
        # just threads waiting on a barrier that never completes.
        print(f"  FAIL  {label}: HUNG (no exit after {timeout}s)")
        print("        This is the failure that writes one row and then sits at zero")
        print("        processor time until the scheduler kills the task.")
        failures.append(f"{label} hung")
        return
    if r.returncode == 0:
        print(f"  OK    {label}")
        return
    if r.returncode in (-11, 139, -6, 134):
        print(f"  FAIL  {label}: SEGFAULT (exit {r.returncode}), no Python traceback")
        failures.append(f"{label} segfaulted")
        return
    first = (r.stderr.strip().splitlines() or ["no stderr"])[-1]
    print(f"  FAIL  {label}: exit {r.returncode}: {first}")
    failures.append(label)


def check_hang_probes(failures):
    """The LightGBM fit and the Gaussian-process fit, in one environment.

    The thread settings are named rather than inherited, because the two halves
    of the study differ: QM9 jobs set no thread count at all, the experimental
    module pins both to 4 at import. Testing only the ambient setting is how
    this same question got opposite answers in two audits.

    Use `-u`/None to UNSET. Setting a thread count to the EMPTY STRING is not
    the same as leaving it unset -- some numerical libraries reject an empty
    value outright, which once produced a spurious error and hid the real one.
    """
    print("the two failures that forced the rebuild (each in its own process)")
    unset = {"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None}
    pinned = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    _run_probe("LightGBM fits, no thread count set (how QM9 jobs run)",
               _LGB_PROBE, unset, failures)
    _run_probe("LightGBM fits, both pinned to 4 (how validation jobs run)",
               _LGB_PROBE, pinned, failures)
    _run_probe("Gaussian process fits after lightgbm+xgboost, no thread count set",
               _GP_PROBE, unset, failures)
    _run_probe("Gaussian process fits after lightgbm+xgboost, both pinned to 4",
               _GP_PROBE, pinned, failures)
    print()


def check_loaded_runtimes(failures):
    """Ground truth: what a process that imported every backend actually mapped.

    The static check above reads what is installed. This reads /proc/self/maps
    after importing the backends in the order models/models.py does, which is
    the measurement RERUN_PLAN.md 2.8e-bis reports ("three OpenMP runtimes
    loaded in the one process"). Linux only -- there is no equivalent file on
    macOS, and the cluster is where the answer counts.
    """
    if not sys.platform.startswith("linux"):
        print("loaded threading runtimes: /proc/self/maps is Linux-only, skipping")
        print("  (this check answers the question on the cluster, which is where it counts)")
        print()
        return

    import subprocess as _sp
    code = """
import torch, sklearn, lightgbm, xgboost, gpytorch  # noqa: F401
import os
seen = {}
for line in open('/proc/self/maps'):
    path = line.rstrip().split(' ', 5)[-1].strip()
    if not path.startswith('/'):
        continue
    b = os.path.basename(path)
    if b.startswith(('libomp', 'libiomp', 'libgomp')):
        seen.setdefault(os.path.realpath(path), b)
for k, v in sorted(seen.items()):
    print(f'{k}\\t{v}')
"""
    print("loaded threading runtimes (/proc/self/maps after importing every backend)")
    try:
        r = _sp.run([sys.executable, "-c", code], capture_output=True,
                    text=True, timeout=600)
    except Exception as e:
        print(f"  ---   could not run the probe: {e}")
        print()
        return
    if r.returncode != 0:
        first = (r.stderr.strip().splitlines() or ["no stderr"])[-1]
        print(f"  FAIL  importing every backend failed: exit {r.returncode}: {first}")
        failures.append("backend import for the maps probe")
        print()
        return
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    if len(lines) <= 1:
        for l in lines:
            path, _, name = l.partition("\t")
            print(f"  OK    one runtime mapped: {path} ({name})")
        if not lines:
            print("  OK    no OpenMP runtime mapped")
    else:
        print(f"  FAIL  {len(lines)} distinct OpenMP runtimes mapped into ONE process:")
        for l in lines:
            path, _, name = l.partition("\t")
            print(f"          {path} ({name})")
        failures.append("multiple threading runtimes loaded")
    print()


def check_env_recipe(failures):
    """Is env.yml a truthful record of what is installed?

    It was not, for five months: the file pinned pytorch 2.5.1 while the
    cluster ran a PyPI wheel of 2.3.1+cu121, and pinned gpytorch 1.14 while
    botorch 0.10.0 forced 1.11. Nothing reported either, so "what were these
    results produced under" had no answer. This makes the question a check.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    yml = os.path.join(root, "env.yml")
    if not os.path.isfile(yml):
        print(f"env.yml: not found at {yml}, skipping")
        print()
        return

    try:
        import importlib.metadata as md
    except Exception as e:
        print(f"env.yml: cannot check ({e})")
        print()
        return

    # conda package name -> the distribution that carries its python metadata,
    # where the two differ.
    dist_of = {
        "pytorch": "torch",
        "pytorch_geometric": "torch-geometric",
        "matplotlib-base": "matplotlib",
        "huggingface_hub": "huggingface-hub",
        "typing_extensions": "typing-extensions",
        "rdkit-dev": "rdkit",
    }
    # Pins with no python metadata to compare against.
    skip = {"python", "pip", "boost-cpp", "llvm-openmp", "pybind11",
            "pybind11-global"}

    import re
    pins = []
    for raw in open(yml):
        line = raw.split("#", 1)[0].strip()
        if not line.startswith("- "):
            continue
        spec = line[2:].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==?\s*([0-9][0-9A-Za-z_.\-]*)", spec)
        if not m:
            continue
        name, want = m.group(1), m.group(2)
        if name in skip:
            continue
        pins.append((name, want))

    print(f"env.yml is a truthful record ({len(pins)} pinned packages)")
    bad = []
    for name, want in pins:
        dist = dist_of.get(name, name)
        have = None
        for cand in (dist, dist.replace("_", "-"), dist.replace("-", "_")):
            try:
                have = md.version(cand)
                break
            except md.PackageNotFoundError:
                continue
        if have is None:
            bad.append(f"{name}: pinned {want}, NOT INSTALLED")
            continue
        # A local version tag is exactly the tell that a PyPI wheel replaced a
        # conda package: the live environment read "2.3.1+cu121".
        base = have.split("+")[0]
        if base != want:
            extra = " (a PyPI wheel, not the conda package)" if "+" in have else ""
            bad.append(f"{name}: env.yml pins {want}, installed {have}{extra}")

    if bad:
        for line in bad:
            print(f"  FAIL  {line}")
        print("        env.yml does not describe this interpreter, so it is not a record")
        print("        of what any result was produced under. Rebuild it, or fix the pin")
        print("        and pip-constraints.txt together and say what moved in RERUN_PLAN.md.")
        failures.append("env.yml disagrees with what is installed")
    else:
        print("  OK    every pinned version in env.yml is what is installed")
    print()


def check_project_packages(failures):
    """noiseInject and kirby: both pipelines import them with no sys.path help."""
    print("project packages (the validation pipeline imports these at module scope)")
    for mod, human in (("noiseInject", "the noise injector"),
                       ("kirby", "the validation pipeline's own package")):
        try:
            m = importlib.import_module(mod)
            where = getattr(m, "__file__", "?")
            ver = getattr(m, "__version__", None)
            if ver is None:
                try:
                    import importlib.metadata as md
                    ver = md.version(mod)
                except Exception:
                    ver = "?"
            print(f"  OK    {mod:<12s} {ver}   {where}")
        except Exception as e:
            print(f"  FAIL  {mod}: {type(e).__name__}: {str(e).splitlines()[0]}")
            print(f"        {human} is not importable, so the KIRBy half cannot start.")
            print("        setup.sh installs it editable from the checkout.")
            failures.append(mod)
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
    print("QM9 roster -- each model's backend is imported and the estimator constructed.")
    print("NOTE: this does NOT import models/models.py itself. That import costs about a")
    print("minute, so it is behind --deep and belongs in the preflight, not in every task.")

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


def check_validation_models(models, failures, resource_failures):
    """The per-task guard for the validation and uncertainty jobs.

    The counterpart of check_qm9 for KIRBy's model names. Cheap by design: it
    imports each model's backend and constructs the estimator, and does NOT
    exec alternative_data_noise_robustness.py -- that is check_validation's job
    and it belongs in the preflight, not in 420 array tasks.
    """
    print("validation/uncertainty roster -- each model's backend is imported and, where "
          "there is one, the estimator constructed.")

    unknown = [m for m in models if m not in VALIDATION_MODELS]
    if unknown:
        print(f"  FAIL  unknown model name(s): {sorted(unknown)}")
        print(f"        known: {sorted(VALIDATION_MODELS)}")
        failures.append("unknown validation model names")
        models = [m for m in models if m in VALIDATION_MODELS]

    for model in sorted(set(models)):
        modules, attr = VALIDATION_MODELS[model]

        def _build(modules=modules, attr=attr):
            mods = [importlib.import_module(m) for m in modules]
            if attr is not None:
                getattr(mods[0], attr)()  # constructed, not merely imported

        probe(f"{model:<16s} ({', '.join(modules)})", _build, failures, resource_failures)
    print()


def check_validation(failures, resource_failures=None):
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

    probe("every optional backend importable", _flags, failures, resource_failures)
    print()


def check_models_module(failures, resource_failures=None):
    """Import models/models.py for real.

    Every backend in that file is imported unguarded at module scope, so this
    is the single check that proves the training code can start at all. It is
    NOT part of the per-task guard: it costs about a minute, and 390 tasks
    paying that is seven CPU-hours to learn something one preflight run
    establishes once.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    def _import():
        if root not in sys.path:
            sys.path.insert(0, root)
        importlib.import_module("models.models")

    print("deep check")
    probe("models.models imports", _import, failures, resource_failures)
    print()


def audit_roster():
    """Every label generate_scripts.py can emit must be a key of QM9_MODELS.

    The job template passes the model label straight to --models, so a label
    this file does not know would stop the task with "unknown model name" --
    the guard blocking a job for the guard's own reason rather than for
    anything wrong with the environment.
    """
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    gen = os.path.join(here, "..", "slurm_scripts_qm9_rerun", "generate_scripts.py")
    if not os.path.isfile(gen):
        print(f"FAIL: cannot find {gen}")
        return 1

    text = open(gen).read()
    labels = []
    for block in ("MODELS", "EXCLUDED_MODELS"):
        m = re.search(block + r" = \{(.*?)\n\}", text, re.S)
        if not m:
            print(f"FAIL: cannot parse {block} out of generate_scripts.py")
            return 1
        labels += re.findall(r"^\s*'([a-z0-9_]+)':", m.group(1), re.M)

    unknown = [l for l in labels if l not in QM9_MODELS]
    if unknown:
        print(f"FAIL: the QM9 job generator can emit {len(unknown)} model label(s) this "
              f"probe does not know: {sorted(unknown)}")
        print("      Add them to QM9_MODELS, or the guard will block those jobs for its "
              "own reason.")
        return 1

    print(f"OK: all {len(labels)} QM9 job-generator model labels are known to this probe")

    # The other two job families pass KIRBy's model names to --validation-models,
    # so the same drift is possible there and has to fail the same way.
    other = [
        (os.path.join(here, "..", "slurm_scripts_uncertainty_rerun", "generate_scripts.py"),
         r"^MODELS = \{(.*?)\n\}", r"^\s*'([A-Za-z0-9_-]+)'\s*:"),
        (os.path.join(here, "..", "slurm_scripts_validation_rerun", "generate_scripts.py"),
         r"^MODELS_ALL = \[(.*?)\]", r"'([A-Za-z0-9_-]+)'"),
    ]
    v_total = 0
    for path, block_re, label_re in other:
        if not os.path.isfile(path):
            print(f"SKIP: {os.path.normpath(path)} is not present")
            continue
        m = re.search(block_re, open(path).read(), re.S | re.M)
        if not m:
            print(f"FAIL: cannot parse the model list out of {os.path.normpath(path)}")
            return 1
        v_labels = re.findall(label_re, m.group(1), re.M)
        v_unknown = [l for l in v_labels if l not in VALIDATION_MODELS]
        if v_unknown:
            print(f"FAIL: {os.path.normpath(path)} can emit {len(v_unknown)} model "
                  f"label(s) this probe does not know: {sorted(v_unknown)}")
            print("      Add them to VALIDATION_MODELS.")
            return 1
        v_total += len(v_labels)

    print(f"OK: all {v_total} validation/uncertainty model labels are known to this probe")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Only check these models. Default: the whole QM9 roster.")
    ap.add_argument("--validation-models", nargs="+", default=None, metavar="NAME",
                    help="Check these KIRBy model names instead of the QM9 roster (e.g. "
                         "SVM, LightGBM, BNN-Full). This is what the validation and "
                         "uncertainty job templates call.")
    ap.add_argument("--validation", action="store_true",
                    help="Also check the KIRBy validation roster.")
    ap.add_argument("--deep", action="store_true",
                    help="Also import models/models.py itself, check env.yml against what "
                         "is installed, and RUN the two failures that forced the "
                         "environment rebuild -- the LightGBM fit and the Gaussian-process "
                         "fit after the boosting libraries are loaded. A few minutes. Use "
                         "in the preflight, not in a per-task guard.")
    ap.add_argument("--audit-roster", action="store_true",
                    help="Check that every model the job generator can emit is known here, "
                         "and exit. Nothing is imported.")
    args = ap.parse_args()

    if args.audit_roster:
        return audit_roster()

    report_versions()

    failures = []
    resource_failures = []
    if not check_pyg_companions(resource_failures):
        failures.append("torch_geometric companions")

    if args.validation_models:
        # The validation and uncertainty jobs never touch process_and_train.py,
        # so checking the QM9 roster here would fail them for packages they do
        # not use.
        if args.models:
            print("ERROR: pass --models or --validation-models, not both -- they are "
                  "different namespaces for the same models.")
            return 2
        check_declared_requirements(failures, args.validation_models)
        check_validation_models(args.validation_models, failures, resource_failures)
    else:
        requested = args.models if args.models else sorted(QM9_MODELS)
        check_declared_requirements(failures, requested)
        check_qm9(requested, failures, resource_failures)
    check_threading_runtimes(failures)

    if args.deep:
        check_env_recipe(failures)
        check_project_packages(failures)
        check_models_module(failures, resource_failures)
        # What the process actually mapped, then whether it can survive it.
        # Both halves have to pass HERE, in one environment: curing the
        # LightGBM failure at the Gaussian process's expense is the trap that
        # an "import lightgbm first" fix walked into on 2026-08-27.
        check_loaded_runtimes(failures)
        check_hang_probes(failures)

    if args.validation:
        check_validation(failures, resource_failures)

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
