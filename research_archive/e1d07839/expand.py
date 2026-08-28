import re, os, glob, itertools, json
DIRS = ["slurm_scripts_anova","slurm_scripts_missing","slurm_scripts_mol2vec",
        "slurm_scripts_mol2vec_missing","slurm_scripts_mol2vec_investigation",
        "slurm_scripts_vbll","slurm_scripts_full_vbll","slurm_scripts_vbll_valprop_rerun",
        "slurm_scripts_continuous_pdv","slurm_scripts_cpdv_missing",
        "slurm_scripts_gauche_rbf","slurm_scripts_gauche_mhggnn"]
out={}
unres=[]
for d in DIRS:
    paths=set()
    for f in sorted(glob.glob(os.path.join(d,"**","*.sh"),recursive=True)):
        if os.path.basename(f).startswith("submit"): continue
        txt=open(f).read()
        # collect for-loop var -> values  and  ARR=(...) ; VAR=${ARR[...]}
        loops=dict(re.findall(r'for\s+(\w+)\s+in\s+([^\n;]+?)\s*;?\s*do', txt))
        loops={k:v.split() for k,v in loops.items()}
        for k,v in re.findall(r'^(\w+)=\(([^)]*)\)', txt, re.M):
            loops.setdefault(k, v.split())
        for k,v in re.findall(r'^(\w+)="([^"]*)"', txt, re.M):
            if k not in loops and ' ' not in v: loops.setdefault(k,[v])
        # index assignment VAR=${ARR[...]}
        for k,arr in re.findall(r'^(\w+)=\$\{(\w+)\[[^\]]*\]\}', txt, re.M):
            if arr in loops: loops[k]=loops[arr]
        for m in re.finditer(r'-f\s+(\S+\.csv)', txt):
            p=m.group(1)
            vs=set(re.findall(r'\$\{?(\w+)\}?', p))
            vs={v for v in vs if v in loops}
            combos=[dict(zip(vs,c)) for c in itertools.product(*[loops[v] for v in vs])] or [{}]
            for c in combos:
                q=p
                for k,val in c.items():
                    q=q.replace("${%s}"%k,val).replace("$"+k,val)
                if '$' in q or '{' in q: unres.append((f,q))
                paths.add(q)
    out[d]=paths
allp=set().union(*out.values())
for d in DIRS: print(f"{d}: {len(out[d])}")
print("UNION DISTINCT:", len(allp))
print("UNRESOLVED:", len(unres))
for u in unres[:20]: print("  ",u)
open(os.path.join(os.environ['SD'],'expanded.txt'),'w').write("\n".join(sorted(allp)))
