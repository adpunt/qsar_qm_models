DATASETS=['logd','caco2','herg_ki']; REPS=['ECFP4','PDV','SNS','MHG-GNN-pretrained']
FULL=['legacy','outlier','quantile','hetero','threshold','valprop']
DROPPED=[s for s in FULL if s!='hetero']
def dispatch(i,S):
    n_st,n_rep=len(S),len(REPS); n=len(DATASETS)*n_rep*n_st
    if i>=n: return f"OUT-OF-RANGE (guard exits 2; n_tasks={n})"
    return (DATASETS[i//(n_st*n_rep)],REPS[(i//n_st)%n_rep],S[i%n_st])
for i in [52,58,64,70]:
    print(f"  idx {i:3d}: 6-strategy script -> {dispatch(i,FULL)}   |  5-strategy script -> {dispatch(i,DROPPED)}")
print("  correct 5-strategy indices for herg_ki x threshold:",
      [2*(5*4)+r*5+DROPPED.index('threshold') for r in range(4)])
