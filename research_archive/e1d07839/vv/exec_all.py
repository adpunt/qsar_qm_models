import subprocess, os, pathlib, re, itertools, collections, json
S=pathlib.Path("/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/vv")
gen=S/"gen"
prep=str(S/"prep.sh")
DS=['logd','caco2','herg_ki']; REPS=['ECFP4','PDV','SNS','MHG-GNN-pretrained']
ALL=['legacy','outlier','quantile','hetero','threshold','valprop']
DROP={1:[],2:['hetero'],3:['hetero','threshold'],4:[],5:['hetero'],6:['hetero','threshold'],
      7:[],8:['hetero'],9:['hetero','threshold'],10:[],11:['hetero'],12:['hetero','threshold']}
EXPECT_FLAGS={n:(('--oof-outer-folds' in open(gen/f"case{n}"/"unc_gp.sh").read()),
                 ('--threshold-quantile' in open(gen/f"case{n}"/"unc_gp.sh").read())) for n in range(1,13)}
env=dict(os.environ); env['PATH']=f"{S}/bin:/usr/bin:/bin:/usr/sbin:/sbin"
env['SLURM_JOB_PARTITION']='medium'
fails=[]; summary=[]
for n in range(1,13):
    strats=[s for s in ALL if s not in DROP[n]]
    ntasks=3*4*len(strats)
    for sh in sorted((gen/f"case{n}").glob("unc_*.sh")):
        dst=S/"tmp.sh"
        subprocess.run([prep,str(sh),str(dst)],check=True)
        seen={}
        for i in range(ntasks):
            log=S/"tmp.log"
            if log.exists(): log.unlink()
            e=dict(env); e['SLURM_ARRAY_TASK_ID']=str(i); e['STUB_LOG']=str(log)
            r=subprocess.run(["/bin/bash",str(dst)],capture_output=True,text=True,env=e)
            if r.returncode!=0 or r.stderr.strip():
                fails.append((n,sh.name,i,r.returncode,r.stderr.strip()[:200])); continue
            txt=log.read_text() if log.exists() else ""
            if txt.count("---CMD---")!=1:
                fails.append((n,sh.name,i,"python invoked %d times"%txt.count("---CMD---"),"")); continue
            args=[l[4:] for l in txt.splitlines() if l.startswith("ARG:")]
            if args[-2]!='--results-root':
                fails.append((n,sh.name,i,"tail not --results-root",args[-4:])); continue
            oof_f,tq_f=EXPECT_FLAGS[n]
            if oof_f != ('--oof-outer-folds' in args) or tq_f != ('--threshold-quantile' in args):
                fails.append((n,sh.name,i,"flag mismatch",args)); continue
            key=(args[args.index('--datasets')+1],args[args.index('--reps')+1],args[args.index('--strategies')+1])
            if key in seen: fails.append((n,sh.name,i,"DUP with %d"%seen[key],key))
            seen[key]=i
        want=set(itertools.product(DS,REPS,strats))
        if set(seen)!=want:
            fails.append((n,sh.name,"-","coverage",("missing",sorted(want-set(seen)),"extra",sorted(set(seen)-want))))
        summary.append((n,sh.name,ntasks,len(seen)))
print("cases x scripts run:",len(summary),"total task invocations:",sum(s[2] for s in summary))
print("FAILURES:",len(fails))
for f in fails[:30]: print("  ",f)
by=collections.Counter((s[0],s[2]) for s in summary)
print("per-case (case, ntasks) -> #scripts:",dict(by))
print("all seen==ntasks:", all(s[2]==s[3] for s in summary))
