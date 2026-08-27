import sys, importlib.util, numpy as np
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old = load('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/old_core.py','oldc')
new = load('/Users/apunt/repos/NoiseInject/noiseInject/core.py','newc')
rng = np.random.RandomState(0)
y = rng.normal(3.0, 1.5, 500)
ytest = rng.normal(3.0, 1.5, 120)
for st in ['legacy','quantile','threshold','outlier','hetero','valprop']:
    ok_seq = True
    a = old.NoiseInjectorRegression(strategy=st, random_state=42)
    b = new.NoiseInjectorRegression(strategy=st, random_state=42)
    c = new.NoiseInjectorRegression(strategy=st, random_state=42)
    for s in [0.1,0.5,1.0]:
        ya = a.inject(y, s)
        yb = b.inject(y, s)
        # verbose path on a THIRD injector, interleaved with noise_scale calls
        _ = c.noise_scale(ytest, s, reference=y)
        yc, sc, eps = c.inject_verbose(y, s)
        _ = c.noise_scale(ytest, s, reference=y)
        if not np.array_equal(ya, yb): ok_seq=False; print(f"  {st} s={s}: inject MISMATCH old vs new max|d|={np.abs(ya-yb).max():.3e}")
        if not np.array_equal(ya, yc): ok_seq=False; print(f"  {st} s={s}: inject_verbose MISMATCH max|d|={np.abs(ya-yc).max():.3e}")
        if not np.allclose(yc - eps, y): print(f"  {st} s={s}: eps inconsistent")
    print(f"{st}: {'IDENTICAL' if ok_seq else 'DIFFERS'}")
