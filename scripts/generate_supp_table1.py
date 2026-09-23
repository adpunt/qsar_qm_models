#!/usr/bin/env python3
"""Generate the LaTeX for Additional file 1 — the hyperparameter tables.

    python scripts/generate_supp_table1.py            # print the LaTeX
    python scripts/generate_supp_table1.py --write    # splice it into
                                                      # additional_files.tex

WHY THIS SCRIPT EXISTS
----------------------
Additional file 1 was hand-typed. By 2026-09-12 it listed nine models against
a roster of nineteen (HANDOFF.md said eleven; the block holds nine), gave both
forests `min_samples_leaf` 1 where
models/model_defaults.py pins 5 and `max_features` sqrt where it pins 0.3, gave
the Gaussian process a Tanimoto kernel and nothing else, and had no rows for any
Bayesian, variational or variance-head network, for the Gaussian process with a
noise network, or for the epoch cap, the patience and the 100 Monte Carlo passes.

Nothing here is retyped. The roster comes from the `MODELS` dict in
slurm_scripts_qm9_rerun/generate_scripts.py, read through models/tuning_rosters.py
so the table holds exactly the configurations the grid submits. The values come
from models/model_defaults.py, the file both pipelines read. The tuned settings
come from the three files the cluster reads.

WHAT EACH CONFIGURATION'S PARAMETER BLOCKS ARE is derived from that model's own
command line in `MODELS` — `-m rf`, `--kernel rbf`, `--bayesian-transformation
full_variational`, `--heteroscedastic-vbll`, `--loss heteroscedastic` — so a
model added to the generator appears here without this file being edited.

TWO SETTINGS ARE NOT IN THE SHARED SPEC and are listed in CODE_LITERALS below
with the file they live in and an exact line from it. `scripts/test_supp_table1.py`
fails if that line stops being there, so they cannot go stale unnoticed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'models'))

import model_defaults as MD          # noqa: E402
import tuning_rosters as R           # noqa: E402

TEX_PATH = os.path.join(_ROOT, 'additional_files.tex')
BEGIN_MARK = '% BEGIN GENERATED Additional file 1 -- scripts/generate_supp_table1.py'
END_MARK = '% END GENERATED Additional file 1'

QM9_TUNED = os.path.join(_ROOT, 'results', 'master_tuned_hyperparameters.json')
QM9_DECISIONS = os.path.join(_ROOT, 'results', 'hyperparameter_decisions.json')
LAB_TUNED = os.path.join(_ROOT, 'results', 'master_tuned_hyperparameters_lab.json')
MODEL_NAMES = os.path.join(_ROOT, 'model_names.json')


# ---------------------------------------------------------------------------
# The two settings that are NOT in models/model_defaults.py
#
# Every other number in this table is read out of the shared spec. These two are
# literals inside a model's own code, so they are named here together with a
# line from the source that must still be there. scripts/test_supp_table1.py
# greps for each `anchor` and fails if it has moved, which is the only thing
# standing between this table and the hand-typed one it replaced.
# ---------------------------------------------------------------------------
CODE_LITERALS = {
    'heteroscedastic_gp': [
        ('noise network hidden sizes', '[64, 64]',
         'models/models.py', 'nn.Linear(input_dim, 64)'),
        ('noise network output', 'Softplus, plus $10^{-4}$',
         'models/models.py', 'nn.Softplus()  # Ensure positive output'),
        ('Adam learning rate, process', '0.1',
         'models/models.py', "{'params': gp.parameters(), 'lr': 0.1}"),
        ('Adam learning rate, noise network', '0.001',
         'models/models.py', "{'params': noise_net.parameters(), 'lr': 0.001}"),
    ],
    # The anchor carries the comment line above the clamp as well. The clamp
    # itself appears twice in that file -- the second is a multi-domain loss
    # this study does not run -- so the bare line would match the wrong one.
    '--loss heteroscedastic': [
        ('log variance clamp', '$[-10, 10]$',
         'scripts/loss_functions.py',
         '        # Prevent numerical instability\n'
         '        log_var = torch.clamp(log_var, -10, 10)'),
    ],
}

# A plain-English gloss for the parameters whose name does not say what they
# are. These add nothing to the values and change nothing -- they are the
# caption, written per row. Keyed by parameter name; a key that is not in a
# printed block simply never appears.
PARAM_NOTES = {
    'max_depth': 'no limit',                     # printed only when the value is None or -1
    'dist': 'ngboost.distns.Normal',
    'score': 'ngboost.scores.MLE, the log scoring rule',
    'base_max_depth': 'depth of one boosting stage',
    'early_stopping_rounds': 'stages without a better validation log likelihood',
    'use_best_iteration': 'predictions use the best stage, not every stage fitted',
    'max_train_n': 'molecules the process is fitted on; an exact Gaussian '
                   'process is cubic in this. The test set is untouched and the '
                   'number actually fitted is written into every results row',
    'likelihood_noise': 'starting value; it is learned',
    'init_lengthscale_from_data': 'the RBF length scale starts at the median '
                                  'distance between training molecules, not at '
                                  'the library default',
    'lengthscale_probe_n': 'molecules sampled to estimate that median',
    'collapse_fraction': 'a fit whose predictions vary by less than this '
                         'fraction of the training label spread is recorded as '
                         'collapsed rather than scored',
    'fallback_adam_lr': 'used only when the standard fitter refuses the model',
    'fallback_adam_iters': 'used only when the standard fitter refuses the model',
    'single_thread_fit': 'a threading workaround, off since the environment '
                         'check found one runtime',
    'apply_outputscale': '',
}

# Which section of the table a configuration belongs in. Derived from the
# defaults block it draws on, except the support vector machine, which lives in
# SKLEARN_DEFAULTS with the trees and is a kernel method.
KERNEL_BY_NAME = {'svm'}

SECTIONS = ['Tree and boosting models', 'Kernel models', 'Neural networks']


# ---------------------------------------------------------------------------
# LaTeX plumbing
# ---------------------------------------------------------------------------

def tex(s):
    """Escape a plain string for LaTeX."""
    out = []
    for ch in str(s):
        if ch in '_&%#$':
            out.append('\\' + ch)
        elif ch == '~':
            out.append('\\textasciitilde{}')
        elif ch == '^':
            out.append('\\textasciicircum{}')
        else:
            out.append(ch)
    return ''.join(out)


def value(v):
    """One default, as the table prints it.

    A search result such as a learning rate carries the optimiser's full double
    precision -- 0.004285944143830873 -- which is not a number anyone reads. Any
    float whose plain form runs past eight characters is printed to three
    significant figures, and the caption says so. The files keep the full value.
    """
    if v is None:
        return 'None'
    if isinstance(v, bool):
        return 'True' if v else 'False'
    if isinstance(v, float):
        s = repr(v)
        if 'e' in s or 'E' in s:                 # 1e-05 -> 0.00001
            s = f'{v:.10f}'.rstrip('0')
        if len(s.lstrip('-')) > 8:
            s = f'{v:.3g}'
        return f'${s}$' if s.startswith('-') else s
    if isinstance(v, int):
        return f'${v}$' if v < 0 else str(v)
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(value(x) for x in v) + ']'
    return tex(v)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def _display_names():
    """Generator model key -> the name the paper uses for it.

    The alpha/beta family names are the paper's own (paper.tex:216, 224, 284)
    and are BUILT from each model's command line rather than listed: the base
    network gives alpha or beta, the Bayesian transformation gives the prefix,
    and the two noise mechanisms give the suffix. Everything else takes the name
    in model_names.json, which is the settled naming file; where several
    spellings map to one canonical name the first is used.
    """
    names = json.load(open(MODEL_NAMES))
    canonical = {}
    for spelling, canon in names['validation'].items():
        canonical.setdefault(canon, spelling)
    qm9_to_canon = names['qm9']

    out = {}
    for key, entry in R.MODELS.items():
        flags = entry[0]
        base = re.search(r'-m (\S+)', flags)
        base = base.group(1) if base else key
        if base in ('dnn', 'mlp'):
            greek = r'$\alpha$' if base == 'dnn' else r'$\beta$'
            transform = re.search(r'--bayesian-transformation (\S+)', flags)
            transform = transform.group(1) if transform else None
            stem = {None: 'NN-', 'full': 'BNN-', 'full_variational': 'VBLL-'}[transform]
            suffix = ''
            if '--heteroscedastic-vbll' in flags:
                suffix = ' (noise head)'
            elif '--loss heteroscedastic' in flags:
                suffix = ' (variance head)'
            out[key] = stem + greek + suffix
        else:
            canon = qm9_to_canon.get(key, key)
            out[key] = canonical.get(canon, key)
    return out


# ---------------------------------------------------------------------------
# What each configuration's parameters are
# ---------------------------------------------------------------------------

def blocks_for(key, entry):
    """(section, [(parameter, value, note)]) for one configuration.

    Read off that model's own command line in the generator's MODELS dict.
    """
    flags = entry[0]
    base = re.search(r'-m (\S+)', flags)
    base = base.group(1) if base else key
    rows = []

    # --- trees, boosting, support vector machine ---------------------------
    sklearn_key = {'lgb': 'lightgbm'}.get(base, base)
    if sklearn_key in MD.SKLEARN_DEFAULTS:
        for k, v in MD.SKLEARN_DEFAULTS[sklearn_key].items():
            note = PARAM_NOTES.get(k, '')
            if k == 'max_depth' and v not in (None, -1):
                note = ''
            rows.append((k, value(v), note))
        if sklearn_key == 'ngboost':
            rows.append(('ngboost_ensemble_seeds',
                         value(MD.SKLEARN_DEFAULTS['ngboost_ensemble_seeds']),
                         '1 = one fit, no seed ensemble'))
        section = 'Kernel models' if base in KERNEL_BY_NAME else 'Tree and boosting models'
        return section, rows

    # --- Gaussian processes ------------------------------------------------
    if base in ('gauche', 'het_gp'):
        kernel = re.search(r'--kernel (\S+)', flags)
        for k, v in MD.GP_DEFAULTS.items():
            if k == 'kernel' and kernel:
                rows.append((k, tex(kernel.group(1)),
                             'set by this job, not by the shared spec'))
            else:
                rows.append((k, value(v), PARAM_NOTES.get(k, '')))
        if base == 'het_gp':
            for name, val, src, _anchor in CODE_LITERALS['heteroscedastic_gp']:
                rows.append((name, val, f'{src}, not in the shared spec'))
            rows.append(('noise network epochs',
                         value(MD.NEURAL_DEFAULTS['training']['epochs']),
                         'the shared epoch cap'))
        return 'Kernel models', rows

    # --- neural networks ---------------------------------------------------
    if base in ('dnn', 'mlp'):
        for k, v in MD.NEURAL_DEFAULTS[base].items():
            rows.append((k, value(v), ''))
        transform = re.search(r'--bayesian-transformation (\S+)', flags)
        transform = transform.group(1) if transform else None
        if transform == 'full':
            for k in ('bnn_prior_mu', 'bnn_prior_sigma', 'bnn_kl_weight'):
                note = ('1 / number of training molecules'
                        if MD.BAYESIAN_DEFAULTS[k] == 'elbo' else '')
                rows.append((k, value(MD.BAYESIAN_DEFAULTS[k]), note))
        elif transform == 'full_variational':
            for k in ('vbll_prior_mu', 'vbll_prior_sigma',
                      'vbll_init_log_sigma', 'vbll_init_log_noise_var'):
                rows.append((k, value(MD.BAYESIAN_DEFAULTS[k]), ''))
        if '--heteroscedastic-vbll' in flags:
            rows.append(('observation noise', 'predicted per molecule',
                         'a variational noise head on the same features, '
                         'started at the single-number value above'))
        elif '--loss heteroscedastic' in flags:
            rows.append(('output head', 'value and log variance', 'two outputs'))
            rows.append(('loss', 'Gaussian negative log likelihood', ''))
            for name, val, src, _anchor in CODE_LITERALS['--loss heteroscedastic']:
                rows.append((name, val, f'{src}, not in the shared spec'))
        return 'Neural networks', rows

    raise SystemExit(
        f'generate_supp_table1: no parameter block is defined for {key!r} '
        f'({flags!r}). Add one here rather than hand-typing a row into '
        f'additional_files.tex.')


def shared_training_rows():
    """The training settings every neural network in the study shares."""
    return [(k, value(v), note) for k, v, note in (
        (k, v, {'mc_passes': 'stochastic forward passes at inference',
                'epochs': 'cap; early stopping usually reaches it first',
                'patience': 'epochs without a strictly better mean validation loss',
                'restore_best_weights': 'the reported fit is the best epoch, not the last',
                'standardise_targets': 'fitted on training labels alone'}.get(k, ''))
        for k, v in MD.NEURAL_DEFAULTS['training'].items())]


# ---------------------------------------------------------------------------
# Tuned settings
# ---------------------------------------------------------------------------

def tuned_rows():
    """(dataset, model key, parameter, value) for every setting that replaces a
    default, and the list of datasets that have one.

    A setting is chosen once per model per dataset and written under every
    representation, so the six entries are collapsed here — and this raises if
    they are NOT all the same, because that would mean a representation is
    carrying a setting the ranking never gave it.
    """
    rows = []
    decisions = json.load(open(QM9_DECISIONS))
    qm9 = json.load(open(QM9_TUNED))
    lab = json.load(open(LAB_TUNED))

    def collapse(dataset, model, per_rep):
        distinct = {json.dumps(v, sort_keys=True) for v in per_rep.values()}
        if len(distinct) != 1:
            raise SystemExit(
                f'generate_supp_table1: {dataset}/{model} holds '
                f'{len(distinct)} different settings across '
                f'{len(per_rep)} representations. One setting per model per '
                f'dataset is what the tuning rule says, so this table cannot '
                f'print a single row. Fix the tuned file, not this script.')
        return json.loads(distinct.pop()), sorted(per_rep)

    for model, per_rep in sorted(qm9.items()):
        if decisions.get(model) != 'USE_TUNED':
            continue
        params, reps = collapse('qm9', model, per_rep)
        for k, v in sorted(params.items()):
            rows.append(('QM9', model, k, value(v), len(reps)))
    for dataset in sorted(lab):
        for model, per_rep in sorted(lab[dataset].items()):
            params, reps = collapse(dataset, model, per_rep)
            for k, v in sorted(params.items()):
                rows.append((dataset, model, k, value(v), len(reps)))
    return rows


# ---------------------------------------------------------------------------
# The LaTeX
# ---------------------------------------------------------------------------

def main_table(names):
    by_section = {s: [] for s in SECTIONS}
    for key, entry in R.MODELS.items():
        section, rows = blocks_for(key, entry)
        reps = entry[4]
        note = ('' if list(reps) == list(R.ALL_REPS)
                else 'run on ' + ', '.join(tex(r) for r in reps) + ' only')
        by_section[section].append((key, names[key], note, rows))

    n_config = sum(len(v) for v in by_section.values())
    assert n_config == len(R.MODELS), (n_config, len(R.MODELS))

    out = [
        # 5.0cm on the first column, not 4.0: the two longest code names,
        # dnn_bnn_full_variational_hetero and its beta sibling, ran 27pt over a
        # 4.0cm column in typewriter type.
        r'\begin{longtable}{@{}p{5.0cm}p{5.0cm}p{4.6cm}@{}}',
        r'\caption{\textbf{Additional file 1, Table A: default hyperparameters for '
        'every model configuration in the study.'r'}',
        f'Every value is read from \\texttt{{models/model\\_defaults.py}} at spec version '
        f'{tex(MD.SPEC_VERSION)}, spec hash \\texttt{{{tex(MD.spec_hash())}}}, which is the '
        r'file both the QM9 pipeline and the assay pipeline load their settings from. '
        r'Every QM9 results row carries the spec version and the spec hash as two of '
        r'its columns, so a QM9 table can be traced back to the settings that '
        r'produced it; the three assay datasets are written by a second pipeline '
        r'whose results files do not carry those two columns. '
        r'The configuration list is the model roster the job generator submits. '
        r'One setting is used for every representation and for every dataset, '
        r'except where Table B gives a tuned replacement. '
        r'The support vector machine uses an RBF kernel on every representation, '
        r'so no result in this study carries a kernel that changes with the '
        r'representation. Two values are not held in the shared spec and name '
        r'their source file in the third column.}',
        r'\label{af1:hyperparameters}\\',
        r'\toprule',
        r'\textbf{Configuration} & \textbf{Hyperparameter} & \textbf{Default value} \\',
        r'\midrule',
        r'\endfirsthead',
        r'\toprule',
        r'\textbf{Configuration} & \textbf{Hyperparameter} & \textbf{Default value} \\',
        r'\midrule',
        r'\endhead',
    ]

    # The training block, once.
    out.append(r'\multicolumn{3}{@{}l}{\textit{Shared by every neural network below}} \\')
    out.append(r'\midrule')
    first = True
    for k, v, note in shared_training_rows():
        label = r'Training, all networks' if first else ''
        first = False
        val = v + (f' \\newline {{\\footnotesize\\raggedright {tex(note)}}}' if note else '')
        out.append(f'{label} & {tex(k)} & {val}' + r' \\')
    out.append(r'\midrule')

    for section in SECTIONS:
        out.append(r'\multicolumn{3}{@{}l}{\textit{' + section + r'}} \\')
        out.append(r'\midrule')
        for i, (key, label, note, rows) in enumerate(by_section[section]):
            if i:
                out.append(r'\addlinespace')
            head = f'{label} \\newline \\footnotesize\\texttt{{{tex(key)}}}'
            if note:
                head += f' \\newline \\footnotesize {note}'
            for j, (k, v, rownote) in enumerate(rows):
                cell = head if j == 0 else ''
                val = v + (f' \\newline {{\\footnotesize\\raggedright {tex(rownote)}}}' if rownote else '')
                out.append(f'{cell} & {tex(k)} & {val}' + r' \\')
        if section != SECTIONS[-1]:
            out.append(r'\midrule')

    out += [r'\bottomrule', r'\end{longtable}']
    return '\n'.join(out), n_config


def tuned_table(names):
    rows = tuned_rows()
    if not rows:
        return ('% No tuned setting replaces a default today, so Table B is omitted.')
    models = sorted({r[1] for r in rows})
    out = [
        r'\begin{longtable}{@{}llll@{}}',
        r'\caption{\textbf{Additional file 1, Table B: the tuned settings that replace '
        r'a default, and the four datasets they apply to.}',
        r'A setting is chosen once per model per dataset, not per representation, '
        r'and is then used for all six representations: a two-way comparison across '
        r'model and representation cannot be read if each cell carries its own '
        r'hyperparameters. Only the Bayesian and variational networks are tuned; '
        r'every other configuration in Table A trains at the default. A model with '
        r'no row for a dataset kept the default on that dataset. '
        r'Learning rates are printed to three significant figures; the source '
        r'files hold the full value. '
        r'Source files: \texttt{results/master\_tuned\_hyperparameters.json} for QM9 '
        r'and \texttt{results/master\_tuned\_hyperparameters\_lab.json} for the three '
        r'assay datasets.}',
        r'\label{af1:tuned}\\',
        r'\toprule',
        r'\textbf{Dataset} & \textbf{Configuration} & \textbf{Hyperparameter} & '
        r'\textbf{Tuned value} \\',
        r'\midrule',
        r'\endfirsthead',
        r'\toprule',
        r'\textbf{Dataset} & \textbf{Configuration} & \textbf{Hyperparameter} & '
        r'\textbf{Tuned value} \\',
        r'\midrule',
        r'\endhead',
    ]
    label_for = {'qm9': 'QM9', 'QM9': 'QM9', 'herg': r'hERG K$_i$',
                 'caco2': 'Caco-2', 'logd': 'LogD'}
    previous = None
    for dataset, model, k, v, _nreps in rows:
        here = (dataset, model)
        if previous is not None and here != previous:
            out.append(r'\addlinespace')
        cells = ((label_for.get(dataset, tex(dataset)), names.get(model, tex(model)))
                 if here != previous else ('', ''))
        out.append(f'{cells[0]} & {cells[1]} & {tex(k)} & {v}' + r' \\')
        previous = here
    out += [r'\bottomrule', r'\end{longtable}']
    return '\n'.join(out), models


PDV_SOURCE = os.path.join(_ROOT, 'scripts', 'process_and_train.py')
QM9_PROVENANCE = os.path.join(_ROOT, 'data', 'qm9_pool_provenance.json')


def pdv_descriptor_names():
    """The 200 descriptor names PDV is built from, read out of the pipeline.

    PARSED, NOT IMPORTED. `scripts/process_and_train.py` imports torch, RDKit and
    the whole model stack at module level, and this generator has to run on a
    laptop with none of that installed. The list is a plain literal, so reading it
    with a regex is exact and costs nothing.

    NOT RDKit's OWN LIST. `Descriptors._descList` grows between RDKit releases, so
    "the 200 RDKit descriptors" does not name a set. This one is pinned in the
    pipeline and is the only thing that makes PDV reproducible from the paper.
    """
    text = open(PDV_SOURCE).read()
    head, sep, rest = text.partition('DEFAULT_DESCRIPTOR_LIST = [')
    if not sep:
        raise SystemExit(
            f'{PDV_SOURCE} no longer holds DEFAULT_DESCRIPTOR_LIST. Table C is the '
            f'only published record of which descriptors PDV holds, so this is a '
            f'failure rather than an empty table.')
    body, sep, _ = rest.partition(']')
    names = re.findall(r"'([^']+)'", body)
    if len(names) != len(set(names)):
        raise SystemExit('DEFAULT_DESCRIPTOR_LIST holds a duplicate name.')
    return names


def rdkit_version():
    """The RDKit that computed the descriptors, from the QM9 pool provenance.

    Several of the 200 are version-sensitive -- `Ipc` most of all, whose overflow
    behaviour changed between releases -- so a reader cannot reproduce PDV from
    the names alone.
    """
    try:
        with open(QM9_PROVENANCE) as fh:
            return json.load(fh).get('rdkit_version')
    except (OSError, ValueError):
        return None


def descriptor_table(n_columns=4):
    """Additional file 1, Table C: every descriptor name in the PDV vector."""
    names = pdv_descriptor_names()
    version = rdkit_version()
    version_sentence = (
        f'They were computed with RDKit {tex(version)}. '
        if version else
        '% TODO: the RDKit version is not in data/qm9_pool_provenance.json. ')
    rows = -(-len(names) // n_columns)          # ceiling division
    columns = [names[i * rows:(i + 1) * rows] for i in range(n_columns)]
    spec = 'l' * n_columns
    out = [
        r'\begin{longtable}{@{}' + spec + r'@{}}',
        r'\caption{\textbf{Additional file 1, Table C: the ' + str(len(names)) +
        r' descriptors that make up the PDV representation.}',
        r'One entry is one feature of the vector, in the order the pipeline '
        r'computes them. The list is fixed in the code rather than taken from '
        r"RDKit's current default set, which grows between releases, so the same "
        r'' + str(len(names)) + r' features are built on every dataset and in '
        r'every run. ' + version_sentence +
        r'Descriptors that return a non-finite value on a molecule have that value '
        r'replaced by zero before the standardisation constants are computed. '
        r'Source: \texttt{DEFAULT\_DESCRIPTOR\_LIST} in '
        r'\texttt{scripts/process\_and\_train.py}.}',
        r'\label{af1:pdv}\\',
        r'\toprule',
        r'\endfirsthead',
        r'\toprule',
        r'\endhead',
    ]
    for r_i in range(rows):
        cells = [tex(col[r_i]) if r_i < len(col) else '' for col in columns]
        out.append(' & '.join(cells) + r' \\')
    out += [r'\bottomrule', r'\end{longtable}']
    return '\n'.join(out), len(names)


def build():
    names = _display_names()
    table_a, n_config = main_table(names)
    table_b, tuned_models = tuned_table(names)
    table_c, n_descriptors = descriptor_table()
    body = '\n'.join([
        BEGIN_MARK,
        '% Do not edit between these two lines. Regenerate with',
        '%   python scripts/generate_supp_table1.py --write',
        '',
        r'\begin{center}\Large\textbf{Additional file 1}\end{center}',
        '',
        r'{\small',
        table_a,
        r'}',
        '',
        r'\newpage',
        '',
        r'\begin{center}\Large\textbf{Additional file 1, continued}\end{center}',
        '',
        r'{\small',
        table_b,
        r'}',
        '',
        r'\newpage',
        '',
        r'\begin{center}\Large\textbf{Additional file 1, continued}\end{center}',
        '',
        r'{\small',
        table_c,
        r'}',
        '',
        END_MARK,
    ])
    return body, n_config, tuned_models, n_descriptors


def write_into_tex(body):
    text = open(TEX_PATH).read()
    if BEGIN_MARK in text and END_MARK in text:
        head, _, rest = text.partition(BEGIN_MARK)
        _, _, tail = rest.partition(END_MARK)
        new = head + body + tail
    else:
        raise SystemExit(
            f'{TEX_PATH} carries no generated block. Put these two lines around '
            f'the Additional file 1 block first:\n  {BEGIN_MARK}\n  {END_MARK}')
    if new == text:
        print(f'{TEX_PATH} already holds this block; nothing written.')
        return
    open(TEX_PATH, 'w').write(new)
    print(f'wrote the Additional file 1 block into {TEX_PATH}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write', action='store_true',
                    help='splice the block into additional_files.tex')
    args = ap.parse_args()
    body, n_config, tuned_models, n_descriptors = build()
    if args.write:
        write_into_tex(body)
        print(f'{n_config} configurations in Table A, '
              f'{len(R.MODELS)} in the generator MODELS dict')
        print(f'Table B carries tuned settings for: {", ".join(tuned_models)}')
        print(f'Table C lists {n_descriptors} PDV descriptor names')
    else:
        print(body)
