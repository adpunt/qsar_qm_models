"""The 21 September re-run of four networks appended to the old files.

drop_superseded must keep the re-run copy wherever both exist, keep an old row
that has no re-run copy (and say so), and touch no other model. Run:
    python test_superseded_specs.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib_load as L  # noqa: E402

OLD, NEW = '26163cc378cd', '9249926c38dc'
KEY = ['model', 'rep', 'condition', 'sigma', 'replicate']


def row(model, replicate, spec, r2, source):
    return dict(model=model, rep='ecfp4', condition='gaussian', sigma=0.0,
                replicate=replicate, spec_hash=spec, r2=r2,
                params_source=source)


def main():
    df = pd.DataFrame([
        row('dnn', 0, OLD, 0.80, 'default'),   # replaced
        row('dnn', 0, NEW, 0.85, 'tuned'),
        row('dnn', 1, OLD, 0.81, 'default'),   # no re-run copy: kept
        row('dnn_bnn_full', 0, OLD, 0.70, 'tuned'),  # replaced
        row('dnn_bnn_full', 0, NEW, 0.75, 'tuned'),
        row('rf', 0, OLD, 0.83, 'default'),    # not a re-run model: kept
        row('rf', 0, NEW, 0.84, 'default'),    # kept; the duplicate rule decides
    ])
    out = L.drop_superseded(df, KEY, 'test')
    got = sorted(zip(out['model'], out['replicate'], out['spec_hash']))
    want = sorted([('dnn', 0, NEW), ('dnn', 1, OLD), ('dnn_bnn_full', 0, NEW),
                   ('rf', 0, OLD), ('rf', 0, NEW)])
    assert got == want, got
    # The old copy must never be what survives for a re-run key.
    both = out[(out['model'] == 'dnn') & (out['replicate'] == 0)]
    assert list(both['r2']) == [0.85], both
    # No spec_hash column (an assay file that never wrote one): untouched.
    bare = df.drop(columns='spec_hash')
    assert len(L.drop_superseded(bare, KEY, 'test')) == len(bare)
    print('test_superseded_specs: pass')


if __name__ == '__main__':
    main()
