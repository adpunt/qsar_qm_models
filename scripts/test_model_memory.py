#!/usr/bin/env python
"""All three generators ask model_memory.json, and none of them restates it.

WHY THIS EXISTS. The memory request was changed five times between 2026-09-04 and
2026-09-05 -- 128G, 32G, 64G, 128G, 96G -- each time as ONE number for the whole roster,
and each time from evidence that turned out to be somebody else's jobs or a sample of one
representation. The author's reading of the sacct sweep is what settled it: every row in
it that belongs to this study is a neural network, running 38.9 to 61.2 GB, and not one
tree, kernel or deterministic model from this study appears in it at all. So the number
is per model, and it lives in one file.

What this checks:
  - every model in every generator's roster resolves to a tier, through the canonical
    names in model_names.json rather than by string luck
  - the generated .sh carry exactly what the file says, on every generator form
  - nothing is below the author's floor
  - the uncertainty runs keep their own floor, which is the one place the tree tier is
    deliberately not applied
  - no generator has a hard-coded memory literal left in it
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = json.loads((ROOT / 'model_memory.json').read_text())
NAMES = json.loads((ROOT / 'model_names.json').read_text())
FLOOR = int(SPEC['the_floor'])
BY_CANONICAL = {m: tier for tier, s in SPEC['tiers'].items() for m in s['models']}
UNC_FLOOR = SPEC.get('pipeline_overrides', {}).get('uncertainty_runs', {}).get('floor')

QM9 = ROOT / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py'
LAB = ROOT / 'slurm_scripts_validation_rerun' / 'generate_scripts.py'
UNC = ROOT / 'slurm_scripts_uncertainty_rerun' / 'generate_scripts.py'

FORMS = [
    ('QM9 screen', QM9, ['--stage', '0', '--max-hours', '720'], 'qm9'),
    ('QM9 main grid', QM9, ['--stage', '1', '--max-hours', '720'], 'qm9'),
    ('QM9 deep run', QM9, ['--stage', '2', '--max-hours', '720',
                           '--runtime-selection', str(ROOT / 'deep_run_pairs.json')], 'qm9'),
    ('laboratory breadth', LAB, [], 'validation'),
    ('laboratory censoring', LAB, ['--conditions', 'censoring', '--runtime-selection',
                                   str(ROOT / 'censoring_pairs.json')], 'validation'),
    ('uncertainty, the three', UNC, [], 'uncertainty'),
    ('uncertainty, the four that follow', UNC,
     ['--conditions', 'censoring', 'student_t_nu5', 'outlier_p10', 'laplace'],
     'uncertainty'),
]

# How each generator's file names map back to a model name the maps understand.
STRIP = {
    'qm9': lambda n: re.sub(r'^qm9_s\d_', '', n),
    'validation': lambda n: re.sub(r'^val_', '', n),
    'uncertainty': lambda n: re.sub(r'^unc_', '', n),
}


def gb(mem):
    return int(str(mem).upper().rstrip('G'))


def expected(model_file_name, kind):
    """What model_memory.json says this script should ask for, or None if unmatched."""
    stem = STRIP[kind](model_file_name)
    if kind == 'qm9':
        canonical = NAMES['qm9'].get(stem)
    else:
        # The laboratory and uncertainty scripts are named from the runner's spelling,
        # lower-cased with '-' or '_' for the separator. Match case- and separator-blind.
        want = stem.replace('-', '').replace('_', '').lower()
        canonical = next((c for name, c in NAMES['validation'].items()
                          if name.replace('-', '').replace('_', '').lower() == want), None)
    if canonical is None:
        return None
    tier = BY_CANONICAL.get(canonical, SPEC['default'])
    if kind == 'uncertainty' and UNC_FLOOR and gb(tier) < gb(UNC_FLOOR):
        tier = UNC_FLOOR
    return tier


def main():
    failures, checked = [], 0

    # 1. Nothing in the rosters falls through to the default by accident.
    for tier, s in SPEC['tiers'].items():
        for m in s['models']:
            if m not in NAMES['canonical']:
                failures.append(f'model_memory.json tier {tier} names {m!r}, which is '
                                f'not a canonical name in model_names.json')
    covered = set(BY_CANONICAL)
    for m in NAMES['canonical']:
        if m not in covered:
            failures.append(f'{m} is in model_names.json but in no tier, so it would '
                            f'silently take the default {SPEC["default"]}')

    # 2. The floor holds everywhere, including the pipeline override.
    for tier in SPEC['tiers']:
        if gb(tier) < FLOOR:
            failures.append(f'tier {tier} is below the author\'s {FLOOR}G floor')
    if UNC_FLOOR and gb(UNC_FLOOR) < FLOOR:
        failures.append(f'the uncertainty floor {UNC_FLOOR} is below {FLOOR}G')

    # 3. What the generators actually write.
    with tempfile.TemporaryDirectory() as tmp:
        for label, gen, extra, kind in FORMS:
            out = Path(tmp) / re.sub(r'\W+', '_', label)
            out.mkdir(parents=True)
            proc = subprocess.run(
                [sys.executable, str(gen), *extra, '--out-dir', str(out)],
                capture_output=True, text=True)
            if proc.returncode != 0:
                failures.append(f'{label}: generator exited {proc.returncode}\n'
                                f'      {proc.stderr.strip()[-300:]}')
                continue
            for f in sorted(out.glob('*.sh')):
                if f.name in ('submit_all.sh', 'preflight.sh'):
                    continue
                m = re.search(r'^#SBATCH --mem=(\S+)', f.read_text(), re.M)
                if not m:
                    failures.append(f'{label}: {f.name} has no --mem line')
                    continue
                got = m.group(1)
                if gb(got) < FLOOR:
                    failures.append(f'{label}: {f.name} asks {got}, below the {FLOOR}G '
                                    f'floor')
                if f.name == 'smoke_test.sh':
                    continue
                want = expected(f.stem, kind)
                checked += 1
                if want is None:
                    failures.append(f'{label}: cannot map {f.name} to a canonical model '
                                    f'name, so this check has gone blind on it')
                elif got != want:
                    failures.append(f'{label}: {f.name} asks {got}, model_memory.json '
                                    f'says {want}')

    # 4. No generator restates a memory number of its own.
    for gen in (QM9, LAB, UNC):
        text = gen.read_text()
        code = '\n'.join(l for l in text.split('\n') if not l.lstrip().startswith('#'))
        for lit in re.findall(r"'(\d+G)'", code):
            failures.append(f'{gen.name} still carries the literal {lit!r} in code; '
                            f'model_memory.json is where that decision lives')

    if failures:
        print(f'FAIL — {len(failures)} problem(s):\n')
        for f in failures:
            print(f'  - {f}')
        return 1
    tiers = ', '.join(f'{t} = {len(s["models"])} model(s)'
                      for t, s in sorted(SPEC['tiers'].items()))
    print(f'PASS — {checked} generated scripts across {len(FORMS)} generator forms all '
          f'match model_memory.json ({tiers}; uncertainty floor {UNC_FLOOR}), and no '
          f'generator restates a number.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
