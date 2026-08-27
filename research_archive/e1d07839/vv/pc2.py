import sys, argparse, importlib.util, itertools
P='/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py'
spec=importlib.util.spec_from_file_location('p',P); m=importlib.util.module_from_spec(spec)
sys.modules['p']=m; spec.loader.exec_module(m)
orig=argparse.ArgumentParser.parse_args
class Stop(Exception):
    def __init__(self,ns): self.ns=ns
argparse.ArgumentParser.parse_args=lambda self,a=None,n=None: (_ for _ in ()).throw(Stop(orig(self,a,n)))
DS=['logd','caco2','herg_ki']; REPS=['ECFP4','PDV','SNS','MHG-GNN-pretrained']
ST=['legacy','outlier','quantile','hetero','threshold','valprop']
MODELS=['QRF','NGBoost','GP','BNN-Full','VBLL-Full','MLP-BNN-Full','MLP-VBLL-Full']
ok=bad=0; ns_last=None
for ds,rep,st,mo in itertools.product(DS,REPS,ST,MODELS):
    a=['--datasets',ds,'--models',mo,'--reps',rep,'--strategies',st,
       '--unc-strategies','all','--oof-folds','5','--oof-outer-folds','1']
    if mo=='GP': a+=['--gp-reps',rep,'--gp-kernel','rbf']
    a+=['--results-root','x']
    sys.argv=['x']+a
    try: m.main()
    except Stop as s: ok+=1; ns_last=s.ns
    except SystemExit as e: bad+=1; print("FAIL",e,a)
print(f"full cross product {len(DS)*len(REPS)*len(ST)*len(MODELS)}: ok={ok} bad={bad}")
print("sample namespace:", {k:v for k,v in vars(ns_last).items() if k in
      ('datasets','models','reps','strategies','unc_strategies','oof_folds','oof_outer_folds','gp_reps','gp_kernel','results_root')})
