DATASETS=['logd','caco2','herg_ki']; REPS=['ECFP4','PDV','SNS','MHG-GNN-pretrained']
FULL=['legacy','outlier','quantile','hetero','threshold','valprop']
def preflight_idx(degen, STRATS):           # verbatim from preflight.sh L175-180
    n_st,n_rep=len(STRATS),len(REPS); idx=[]
    for ds,st in degen:
        d_i,s_i=DATASETS.index(ds),STRATS.index(st)
        for r_i in range(n_rep): idx.append(d_i*(n_st*n_rep)+r_i*n_st+s_i)
    return sorted(idx)
def dispatch(i,STRATS):                      # verbatim from unc_*.sh L61-63
    n_st,n_rep=len(STRATS),len(REPS)
    return (DATASETS[i//(n_st*n_rep)],REPS[(i//n_st)%n_rep],STRATS[i%n_st])
# 1) agreement when nothing is dropped
degen=[('herg_ki','threshold')]
idx=preflight_idx(degen,FULL)
print("full 6-strategy scripts: preflight says skip",idx)
print("  dispatch maps those to:",[dispatch(i,FULL) for i in idx])
print("  agree:",all(dispatch(i,FULL)[0]=='herg_ki' and dispatch(i,FULL)[2]=='threshold' for i in idx))
# 2) after preflight's own advice: --drop-strategies hetero
DROPPED=[s for s in FULL if s!='hetero']
print()
print("after `--drop-strategies hetero` (preflight section 4c recommends this):")
print("  preflight STILL prints (it uses m.STRATEGIES, 6 long):",idx)
print("  dispatch in the regenerated script maps those to:",[dispatch(i,DROPPED) for i in idx])
print("  arms actually skipped:",sorted({(dispatch(i,DROPPED)[0],dispatch(i,DROPPED)[2]) for i in idx}))
print("  correct indices for herg_ki x threshold would be:",preflight_idx(degen,DROPPED))
