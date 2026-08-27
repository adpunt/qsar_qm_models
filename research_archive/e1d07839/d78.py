import sys, os, numpy as np, pandas as pd
os.chdir('/Users/apunt/repos/KIRBy/tests')
sys.path.insert(0, '/Users/apunt/repos/KIRBy/tests')
sys.path.insert(0, '/Users/apunt/repos/KIRBy/src')
sys.path.insert(0, '/Users/apunt/repos/NoiseInject')
import importlib.util
spec = importlib.util.spec_from_file_location("pipe", "/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py")
pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)
print("MODULE LOADED", flush=True)
print("NoiseInject file:", __import__('noiseInject').__file__)

df = pipe.download_openadmet()
logd_col = next(c for c in df.columns if 'LogD' in c)
caco_col = next(c for c in df.columns if 'Caco' in c and 'Efflux' in c)
s1,l1 = pipe.load_openadmet_endpoint(df, logd_col, log_transform=False)
s2,l2 = pipe.load_openadmet_endpoint(df, caco_col, log_transform=True)
s3,l3 = pipe.load_chembl_herg()
np.save('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/logd.npy', l1)
np.save('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/caco2.npy', l2)
np.save('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/herg.npy', l3)
for n,l in [('logd',l1),('caco2',l2),('herg',l3)]:
    print(f"{n}: n={len(l)} min={l.min():.4f} max={l.max():.4f} mean={l.mean():.4f} std={l.std():.4f}")
