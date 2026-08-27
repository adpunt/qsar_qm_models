import re, itertools, sys, pathlib
S=pathlib.Path("/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/vv/run")
DS=['logd','caco2','herg_ki']; REPS=['ECFP4','PDV','SNS','MHG-GNN-pretrained']
ST=['legacy','outlier','quantile','hetero','threshold','valprop']
for m in ('gp','qrf'):
    seen={}
    for i in range(72):
        args=[l[4:] for l in (S/f"{m}_{i}.log").read_text().splitlines() if l.startswith("ARG:")]
        d={}
        for k in ('--datasets','--models','--reps','--strategies','--results-root','--oof-folds','--unc-strategies'):
            d[k]=args[args.index(k)+1]
        gp = ('--gp-reps' in args, args[args.index('--gp-reps')+1] if '--gp-reps' in args else None,
              args[args.index('--gp-kernel')+1] if '--gp-kernel' in args else None)
        key=(d['--datasets'],d['--reps'],d['--strategies'])
        if key in seen: print(f"DUP {m}: idx {i} and {seen[key]} both -> {key}")
        seen[key]=i
        exp_slug = d['--reps'].lower().replace('-','')
        want=f"results/uncertainty_rerun/{m}__{d['--datasets']}__{exp_slug}__{d['--strategies']}"
        assert d['--results-root']==want, (i,d['--results-root'],want)
        assert d['--models']==('GP' if m=='gp' else 'QRF'), (i,d['--models'])
        assert d['--unc-strategies']=='all' and d['--oof-folds']=='5'
        if m=='gp':
            assert gp[0] and gp[1]==d['--reps'] and gp[2]=='rbf', (i,gp)
        else:
            assert not gp[0], (i,"QRF unexpectedly has --gp-reps")
    full=set(itertools.product(DS,REPS,ST))
    print(f"{m}: {len(seen)} unique combos; missing={sorted(full-set(seen))}; extra={sorted(set(seen)-full)}")
    # hyphen survival
    hy=[k for k in seen if k[1]=='MHG-GNN-pretrained']
    print(f"{m}: MHG-GNN-pretrained appears {len(hy)} times, rep string preserved verbatim: {set(k[1] for k in hy)}")
