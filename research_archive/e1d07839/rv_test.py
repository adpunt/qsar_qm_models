import sys, importlib.util, numpy as np
sys.path.insert(0, '/Users/apunt/repos/NoiseInject')
from noiseInject.core import NoiseInjectorRegression as New

spec = importlib.util.spec_from_file_location("oldcore", "/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/old_core.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
Old = old.NoiseInjectorRegression

STRATS = ['legacy','quantile','threshold','outlier','hetero','valprop']
rs = np.random.RandomState(0)
for name, y in [('centered', rs.normal(0,1,500)),
                ('pki_like', rs.normal(7.0,1.2,500)),
                ('logd_like', rs.normal(1.5,1.5,500))]:
    for s in STRATS:
        # 1) old inject == new inject, same seed, sequential sigmas
        o = Old(strategy=s, random_state=42); n = New(strategy=s, random_state=42)
        ok_seq = True
        for sig in [0.1,0.2,0.5,1.0]:
            a = o.inject(y, sig); b = n.inject(y, sig)
            if not np.array_equal(a,b): ok_seq=False; break
        # 2) inject == inject_verbose for same fresh state
        n1 = New(strategy=s, random_state=7); n2 = New(strategy=s, random_state=7)
        ok_v = True; ok_state = True
        for sig in [0.1,0.3,0.9]:
            a = n1.inject(y, sig)
            b, si, eps = n2.inject_verbose(y, sig)
            if not np.array_equal(a,b): ok_v=False
            if not np.array_equal(b - y, eps): ok_v=False
            st1 = n1.rng.get_state(); st2 = n2.rng.get_state()
            if not (st1[0]==st2[0] and np.array_equal(st1[1],st2[1]) and st1[2:]==st2[2:]): ok_state=False
        print(f"{name:10s} {s:10s} old==new:{ok_seq}  verbose==inject:{ok_v}  rngstate:{ok_state}")
