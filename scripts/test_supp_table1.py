#!/usr/bin/env python3
"""Guards on Additional file 1.

    python scripts/test_supp_table1.py

Five checks, in the order they would catch the failure that produced the
hand-typed table this replaces:

1. Every configuration in the job generator's `MODELS` dict has a block in the
   table, and the table has no configuration the generator would not submit.
2. The two settings that are NOT in models/model_defaults.py are still where
   generate_supp_table1.py says they are. Each is looked for as an exact line in
   its source file, and must appear exactly once.
3. The block in additional_files.tex is what the generator emits right now. A
   hand edit inside the markers fails here.
4. The values the hand-typed table got wrong are right: both forests at
   min_samples_leaf 5 and max_features 0.3, and neither at 'sqrt'.
5. Additional file 12, the representation-specific SVM kernel table, is gone,
   and no Tanimoto kernel is claimed for the support vector machine anywhere in
   additional_files.tex. The support vector machine uses an RBF kernel on every
   representation (models/model_defaults.py SKLEARN_DEFAULTS['svm']).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'models'))

import generate_supp_table1 as G     # noqa: E402
import model_defaults as MD          # noqa: E402
import tuning_rosters as R           # noqa: E402

failures = []


def check(name, ok, detail=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + (f'  -- {detail}' if detail else ''))
    if not ok:
        failures.append(name)


body, n_config, tuned_models, n_descriptors = G.build()

# 1 -----------------------------------------------------------------------
check('every configuration in the generator MODELS dict has a block',
      n_config == len(R.MODELS),
      f'{n_config} blocks against {len(R.MODELS)} in MODELS')
missing = [k for k in R.MODELS if f'\\texttt{{{G.tex(k)}}}' not in body]
check('every model key is printed beside its block', not missing,
      ', '.join(missing) if missing else f'{len(R.MODELS)} keys')

# 2 -----------------------------------------------------------------------
for group, entries in G.CODE_LITERALS.items():
    for param, _printed, source, anchor in entries:
        path = os.path.join(_ROOT, source)
        text = open(path).read() if os.path.exists(path) else ''
        n = text.count(anchor)
        check(f'{group}: {param} still in {source}', n == 1,
              f'{n} occurrences of {anchor!r}')

# 2b ----------------------------------------------------------------------
# The clamp the table prints, checked by running the loss rather than by
# reading it: a log variance of 50 must cost exactly what a log variance of 10
# costs, and -50 exactly what -10 costs.
try:
    import torch
    from loss_functions import HeteroscedasticLoss
    loss = HeteroscedasticLoss()
    y = torch.tensor([[0.5]])

    def at(lv):
        return float(loss(torch.tensor([[0.0, lv]]), y))
    check('the variance-head loss clamps log variance at $[-10, 10]$',
          abs(at(50.0) - at(10.0)) < 1e-9 and abs(at(-50.0) - at(-10.0)) < 1e-6,
          f'log var 50 -> {at(50.0):.6f}, 10 -> {at(10.0):.6f}, '
          f'-50 -> {at(-50.0):.6f}, -10 -> {at(-10.0):.6f}')
except ImportError as exc:
    print(f'SKIP  the variance-head loss clamp -- {exc}')

# 3 -----------------------------------------------------------------------
tex_text = open(G.TEX_PATH).read()
has_markers = G.BEGIN_MARK in tex_text and G.END_MARK in tex_text
check('additional_files.tex carries the generated markers', has_markers)
if has_markers:
    start = tex_text.index(G.BEGIN_MARK)
    end = tex_text.index(G.END_MARK) + len(G.END_MARK)
    check('the block in additional_files.tex is what the generator emits',
          tex_text[start:end] == body,
          'run: python scripts/generate_supp_table1.py --write')

# 4 -----------------------------------------------------------------------
for forest in ('rf', 'qrf'):
    d = MD.SKLEARN_DEFAULTS[forest]
    check(f'{forest} min_samples_leaf is 5 in the spec and in the table',
          d['min_samples_leaf'] == 5 and 'min\\_samples\\_leaf & 5' in body,
          f"spec says {d['min_samples_leaf']}")
    check(f'{forest} max_features is 0.3 in the spec and in the table',
          d['max_features'] == 0.3 and 'max\\_features & 0.3' in body,
          f"spec says {d['max_features']!r}")
check("no forest in the table is at max_features 'sqrt'",
      'max\\_features & sqrt' not in body)

# 5 -----------------------------------------------------------------------
check('the support vector machine is RBF on every representation',
      MD.SKLEARN_DEFAULTS['svm']['kernel'] == 'rbf')
check('Additional file 12 is gone from additional_files.tex',
      r'\textbf{Additional file 12}' not in tex_text
      and 'af12:' not in tex_text
      and 'Poly kernel' not in tex_text)
svm_tanimoto = [line for line in tex_text.splitlines()
                if 'Tanimoto' in line and 'SVM' in line]
check('no Tanimoto kernel is claimed for the SVM in additional_files.tex',
      not svm_tanimoto, ' | '.join(svm_tanimoto))

# 5b ----------------------------------------------------------------------
# PDV is the one representation a reader cannot rebuild from a library call: the
# 200 descriptor names are pinned in the pipeline, not taken from RDKit's own
# list, which grows between releases. Table C is the only published record of
# which 200, so it has to match the pipeline name for name.
pipeline_names = G.pdv_descriptor_names()
check('Table C lists every PDV descriptor the pipeline computes',
      len(pipeline_names) == 200
      and all(G.tex(n) in body for n in pipeline_names),
      f'{len(pipeline_names)} names in DEFAULT_DESCRIPTOR_LIST; '
      + ', '.join(n for n in pipeline_names if G.tex(n) not in body)[:120])
check('Table C states the RDKit version that computed the descriptors',
      bool(G.rdkit_version()) and f'RDKit {G.rdkit_version()}' in body,
      str(G.rdkit_version()))

# 6 -----------------------------------------------------------------------
# The PDF is the file that is submitted, and it is a separate build step from
# everything above. Regenerating the block and not rebuilding leaves the .tex
# right and the .pdf wrong, with nothing saying so. The spec hash is printed in
# Table A's caption, so it is enough to read it back out of the PDF.
PDF = os.path.join(_ROOT, 'additional_files.pdf')
try:
    from PyPDF2 import PdfReader
except ImportError:
    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None
if PdfReader is None:
    print('SKIP  additional_files.pdf carries the current spec hash '
          '-- no PyPDF2 or pypdf here')
elif not os.path.exists(PDF):
    check('additional_files.pdf exists', False, PDF)
else:
    import re
    pdf_text = ''.join(p.extract_text() or '' for p in PdfReader(PDF).pages)

    def flatten(s):
        """Letters only, lower case.

        Enough to survive the differences between LaTeX source and extracted
        PDF text -- underscores, line-break hyphens, column separators and
        whitespace all disappear on both sides.
        """
        s = re.sub(r'\\[a-zA-Z]+', ' ', s)
        return ''.join(c for c in s.lower() if c.isalpha())

    # One caption per generated table, each from \caption{ to the line before
    # its \label. Table C (the PDV descriptor names) was added 2026-09-20, so
    # the count moved from two to three; it is asserted rather than counted so
    # that a table silently dropped from the generator fails here.
    captions = [m.group(1) for m in
                re.finditer(r'\\caption\{(.*?)\}\n\\label', body, re.S)]
    check('all three captions were found in the generated block',
          len(captions) == 3, f'{len(captions)} captions')
    flat_pdf = flatten(pdf_text)
    stale = [c[:60] for c in captions if flatten(c) not in flat_pdf]
    fresh = not stale and MD.spec_hash() in pdf_text
    check('additional_files.pdf was rebuilt from the current block', fresh,
          f'{len(PdfReader(PDF).pages)} pages, spec hash {MD.spec_hash()}'
          if fresh else
          'the PDF does not carry the current caption text or spec hash. '
          'Rebuild: pdflatex -output-directory=_build_addfiles '
          'additional_files.tex (twice, for the longtable), then copy '
          '_build_addfiles/additional_files.pdf to additional_files.pdf')

print()
if failures:
    print(f'{len(failures)} FAILED: ' + ', '.join(failures))
    sys.exit(1)
print('all checks passed')
