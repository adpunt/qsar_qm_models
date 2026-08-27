import sys, numpy as np
sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
from noiseInject import NoiseInjectorRegression
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/'
for n in ['logd','caco2','herg']:
    y=np.load(S+n+'.npy')
    h=NoiseInjectorRegression('hetero',0).noise_scale(y,1.0)
    v=NoiseInjectorRegression('valprop',0).noise_scale(y,1.0)
    # count discordant pairs where strict inequality flips
    idx=np.argsort(v, kind='mergesort')
    hs=h[idx]; vs=v[idx]
    bad=0; ex=[]
    for i in range(len(idx)-1):
        if vs[i] < vs[i+1] and hs[i] > hs[i+1]:
            bad+=1
            if len(ex)<3: ex.append((y[idx[i]],y[idx[i+1]],hs[i],hs[i+1],vs[i],vs[i+1]))
    print(n, "strict-order violations (adjacent, v strictly increasing but h decreasing):", bad, ex)
    # ties
    print("   ties: |y| uniq",len(np.unique(np.abs(y))),"h uniq",len(np.unique(h)),"v uniq",len(np.unique(v)))
