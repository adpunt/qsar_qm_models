import sys, argparse, importlib.util, pathlib, itertools, json
sys.path.insert(0,'/Users/apunt/repos/KIRBy/tests')
P='/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py'
spec=importlib.util.spec_from_file_location('p',P)
m=importlib.util.module_from_spec(spec); sys.modules['p']=m; spec.loader.exec_module(m)
orig=argparse.ArgumentParser.parse_args
class Stop(Exception):
    def __init__(self,ns): self.ns=ns
def patched(self,args=None,namespace=None):
    raise Stop(orig(self,args,namespace))
argparse.ArgumentParser.parse_args=patched
gen=pathlib.Path(S if False else "/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/vv/gen")
bad=0; ok=0
for n in range(1,13):
    for log in sorted((gen).glob(f"../run/*.log")): pass
# rebuild argv straight from the shell run logs for case7 + the drop cases we re-run below
import subprocess, os
prep="/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/vv/prep.sh"
V=pathlib.Path("/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/vv")
env=dict(os.environ); env['PATH']=f"{V}/bin:/usr/bin:/bin"; env['SLURM_JOB_PARTITION']='medium'
seen_argv=set()
for n in range(1,13):
    ntasks={0:72}.get(0)
    for sh in sorted((gen/f"case{n}").glob("unc_*.sh")):
        dst=V/"pc.sh"; subprocess.run([prep,str(sh),str(dst)],check=True)
        txt=sh.read_text(); nst=len(txt.split("STRATS=(")[1].split(")")[0].split())
        for i in (0, 3*4*nst-1, (3*4*nst)//2):
            log=V/"pc.log"
            if log.exists(): log.unlink()
            e=dict(env); e['SLURM_ARRAY_TASK_ID']=str(i); e['STUB_LOG']=str(log)
            subprocess.run(["/bin/bash",str(dst)],capture_output=True,env=e)
            argv=[l[4:] for l in log.read_text().splitlines() if l.startswith("ARG:")][2:]
            seen_argv.add(tuple(argv))
print(f"{len(seen_argv)} distinct emitted argv vectors -> feeding to the REAL parser")
for argv in sorted(seen_argv):
    sys.argv=['alternative_data_noise_robustness.py']+list(argv)
    try:
        m.main()
    except Stop as s:
        ok+=1
    except SystemExit as e:
        bad+=1; print("  PARSE FAIL", e, argv)
    except Exception as e:
        bad+=1; print("  ERR", type(e).__name__, e, argv)
print(f"parsed OK: {ok}   failed: {bad}")
