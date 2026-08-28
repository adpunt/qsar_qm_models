import itertools, subprocess, os, shutil, pathlib, json
S=pathlib.Path("/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/vv")
D="/Users/apunt/repos/qsar_qm_models/slurm_scripts_uncertainty_rerun/generate_scripts.py"
gen=S/"gen"; shutil.rmtree(gen, ignore_errors=True); gen.mkdir(parents=True)
OOF=[[], ["--oof-outer-folds","1"]]
TQ=[[], ["--threshold-quantile","0.1"]]
DR=[[], ["--drop-strategies","hetero"], ["--drop-strategies","hetero","threshold"]]
cases=[]
for n,(a,b,c) in enumerate(itertools.product(OOF,TQ,DR),1):
    od=gen/f"case{n}"; od.mkdir()
    argv=["python3",D,"--out-dir",str(od)]+a+b+c
    r=subprocess.run(argv,capture_output=True,text=True)
    (od/"gen.out").write_text(r.stdout+r.stderr)
    cases.append({"n":n,"argv":argv[3:],"rc":r.returncode,"stdout_head":r.stdout.splitlines()[0] if r.stdout else ""})
    print(n, a+b+c, "rc="+str(r.returncode), "|", (r.stdout.splitlines() or [""])[0])
(gen/"cases.json").write_text(json.dumps(cases,indent=1))
