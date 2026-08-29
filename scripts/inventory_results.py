#!/usr/bin/env python3
"""
Results Inventory Script
Run this on the server to understand what data you have.

Usage:
    python inventory_results.py [results_dir]

Outputs:
    - inventory_summary.txt (human-readable summary)
    - inventory_detailed.csv (machine-readable for further analysis)
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime

# The settled noise conditions, READ not restated. The six names that used to be
# written into this file -- legacy, value_proportional, quantile, threshold,
# outlier, heteroscedastic -- were deleted in noiseInject 1.0.0, so the report's
# coverage section marked all six NOT FOUND on every run and named none of the
# conditions that actually ran. Longest first, so grouped_wider is matched before
# any shorter name that prefixes it.
_SETTLED_FILE = Path(__file__).resolve().parent.parent / 'noise_conditions.json'
_SETTLED = json.loads(_SETTLED_FILE.read_text())
# Conditions the settled file restricts to a handful of model-and-representation
# pairs (censoring today) are not expected on every pair, so they are listed
# apart -- calling one MISSING because it is absent from most pairs would be the
# same manufactured gap the retired names produced.
_PAIR_SUBSET = {c['name'] for c in _SETTLED['stage_1_full_grid']
                if c.get('scope', {}).get('mode') == 'pair_subset'}
MAIN_GRID_CONDITIONS = [c['name'] for c in _SETTLED['stage_1_full_grid']
                        if c['name'] not in _PAIR_SUBSET]
DEPTH_CONDITIONS = ([c['name'] for c in _SETTLED['stage_2_depth_only']]
                    + sorted(_PAIR_SUBSET))
SETTLED_CONDITION_NAMES = sorted(MAIN_GRID_CONDITIONS + DEPTH_CONDITIONS,
                                 key=len, reverse=True)


def parse_filename(filename):
    """Parse a result filename to extract components."""
    info = {
        'filename': filename,
        'phase': None,
        'subphase': None,
        'model': None,
        'representation': None,
        'noise_strategy': None,
        'target': None,
        'has_uncertainty': '_uncertainty_values' in filename,
        'has_per_epoch': '_per_epoch' in filename,
        'calibration_size': None,
        'sigma': None,
        'file_type': 'unknown'
    }

    # Remove extensions for parsing
    base = filename.replace('_uncertainty_values.csv', '').replace('_per_epoch.csv', '').replace('.csv', '').replace('.json', '')

    # Detect phase
    phase_match = re.match(r'^phase(\d+)([a-z])?_', base)
    if phase_match:
        info['phase'] = int(phase_match.group(1))
        info['subphase'] = phase_match.group(2)
        info['file_type'] = 'phase_result'

    # Phase 0 patterns
    if info['phase'] == 0:
        if '_tuning_' in base:
            info['file_type'] = 'tuning'
            # phase0a_tuning_<model>_<rep>.csv
            match = re.search(r'tuning_(\w+)_(\w+)$', base)
            if match:
                info['model'] = match.group(1)
                info['representation'] = match.group(2)
        elif '_default_' in base:
            info['file_type'] = 'default'
            match = re.search(r'default_(\w+)$', base)
            if match:
                info['model'] = match.group(1)
        elif '_tuned_' in base:
            info['file_type'] = 'tuned'
            match = re.search(r'tuned_(\w+?)(?:_(\w+))?(?:_s([\d.]+))?$', base)
            if match:
                info['model'] = match.group(1)
                info['representation'] = match.group(2)
                info['sigma'] = match.group(3)
        elif '_screen_' in base:
            info['file_type'] = 'screening'
            # phase0c_screen_<model>_<variant>.csv or phase0c_screen_<model>.csv
            match = re.search(r'screen_(.+)$', base)
            if match:
                parts = match.group(1).split('_')
                info['model'] = parts[0]
                if len(parts) > 1:
                    # Could be representation or bayesian variant
                    remainder = '_'.join(parts[1:])
                    if remainder in ['ecfp4', 'pdv', 'sns', 'smiles', 'randomized_smiles', 'graph']:
                        info['representation'] = remainder
                    elif 'bayes' in remainder or 'bnn' in remainder:
                        info['model'] = f"{parts[0]}_{remainder}"
                    else:
                        info['representation'] = remainder
        elif '_mhggnn_' in base:
            info['file_type'] = 'mhggnn'
            info['representation'] = 'mhggnn'

    # Phase 1-2 patterns: phase1a_<rep>_<model>_<variant>.csv
    elif info['phase'] in [1, 2]:
        match = re.search(r'phase\d[a-z]?_(\w+)_(\w+?)(?:_(\w+))?$', base)
        if match:
            info['representation'] = match.group(1)
            info['model'] = match.group(2)
            variant = match.group(3)
            if variant:
                info['model'] = f"{info['model']}_{variant}"

    # Phase 3 patterns (conformal): phase3a_<rep>_conformal_<base>_calib<size>.csv
    elif info['phase'] == 3:
        info['file_type'] = 'conformal'
        match = re.search(r'phase3[a-z]?_(\w+)_conformal_(\w+?)(?:_calib(\d+))?$', base)
        if match:
            info['representation'] = match.group(1)
            info['model'] = f"conformal_{match.group(2)}"
            info['calibration_size'] = match.group(3)

    # Phase 4 patterns: phase4a_<target>_<rep>_<model>_<noise_strategy>.csv
    elif info['phase'] == 4:
        info['file_type'] = 'generalization'
        # Try with target first
        match = re.search(r'phase4[a-z]?_(\w+)_(\w+)_(\w+)_(\w+)$', base)
        if match:
            potential_target = match.group(1)
            # Check if first part is a known target
            known_targets = ['homolumo', 'alpha', 'gibbsenergy', 'hlm', 'rlm', 'mu']
            if potential_target in known_targets:
                info['target'] = potential_target
                info['representation'] = match.group(2)
                info['model'] = match.group(3)
                info['noise_strategy'] = match.group(4)
            else:
                # No target specified, it's rep_model_strategy
                info['representation'] = potential_target
                info['model'] = match.group(2)
                info['noise_strategy'] = match.group(3)
        else:
            # Try simpler pattern: phase4a_<rep>_<model>_<strategy>.csv
            match = re.search(r'phase4[a-z]?_(\w+)_(\w+)_(\w+)$', base)
            if match:
                info['representation'] = match.group(1)
                info['model'] = match.group(2)
                info['noise_strategy'] = match.group(3)

    # The current QM9 grid: anova_<condition>_<rep>_<model>.csv, the name the
    # job template writes (slurm_scripts_qm9_rerun/generate_scripts.py). Nothing
    # here recognised it, so an inventory of a real run reported every file as
    # 'other' and the noise-condition section of the report came out empty --
    # which reads as "no noise runs exist" rather than "this parser is old".
    #
    # The condition is matched against the settled names, longest first, because
    # several of them contain underscores (grouped_wider, student_t_nu5) and a
    # split on '_' cuts them in half.
    if not phase_match and base.startswith('anova_'):
        rest = base[len('anova_'):]
        for condition in SETTLED_CONDITION_NAMES:
            if rest.startswith(condition + '_'):
                tail = rest[len(condition) + 1:]
                if '_' in tail:
                    rep, model = tail.split('_', 1)
                    info['file_type'] = 'noise_grid'
                    info['noise_strategy'] = condition
                    info['representation'] = rep
                    info['model'] = model
                break

    # Non-phase files
    if not phase_match and info['file_type'] == 'unknown':
        # Check for common patterns
        if 'scaffold' in base.lower():
            info['file_type'] = 'scaffold'
        elif 'tuning' in base.lower():
            info['file_type'] = 'tuning'
        elif 'loss_' in base:
            info['file_type'] = 'loss_experiment'
        elif 'noise_' in base:
            info['file_type'] = 'noise_experiment'
        elif 'shap' in base.lower():
            info['file_type'] = 'shap'
        elif 'bayesian' in base.lower():
            info['file_type'] = 'bayesian'
        elif 'distribution' in base.lower():
            info['file_type'] = 'distribution'
        elif 'sanity' in base.lower():
            info['file_type'] = 'sanity_check'
        else:
            info['file_type'] = 'other'

    return info


def get_file_size(filepath):
    """Get file size in MB."""
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except:
        return 0


def scan_results(results_dir):
    """Scan results directory and categorize all files."""
    results_dir = Path(results_dir)

    all_files = []

    # Scan CSV files
    for f in results_dir.glob("*.csv"):
        info = parse_filename(f.name)
        info['size_mb'] = get_file_size(f)
        info['full_path'] = str(f)
        all_files.append(info)

    # Scan JSON files
    for f in results_dir.glob("*.json"):
        info = parse_filename(f.name)
        info['size_mb'] = get_file_size(f)
        info['full_path'] = str(f)
        all_files.append(info)

    # Scan subdirectories
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]

    return all_files, subdirs


def generate_report(all_files, subdirs, results_dir):
    """Generate inventory report."""

    lines = []
    lines.append("=" * 80)
    lines.append("RESULTS INVENTORY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Results directory: {results_dir}")
    lines.append("=" * 80)

    # Overall statistics
    total_files = len(all_files)
    total_size = sum(f['size_mb'] for f in all_files)
    uncertainty_files = [f for f in all_files if f['has_uncertainty']]
    uncertainty_size = sum(f['size_mb'] for f in uncertainty_files)

    lines.append(f"\n{'OVERALL STATISTICS':=^80}")
    lines.append(f"Total CSV/JSON files: {total_files}")
    lines.append(f"Total size: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    lines.append(f"Uncertainty value files: {len(uncertainty_files)} ({uncertainty_size:.1f} MB)")
    lines.append(f"Subdirectories: {len(subdirs)}")

    # Files by phase
    lines.append(f"\n{'FILES BY PHASE':=^80}")
    phase_counts = defaultdict(lambda: {'count': 0, 'size': 0, 'uncertainty': 0})
    for f in all_files:
        phase = f['phase'] if f['phase'] is not None else 'non-phase'
        subphase = f['subphase'] or ''
        key = f"Phase {phase}{subphase}" if phase != 'non-phase' else 'Non-phase files'
        phase_counts[key]['count'] += 1
        phase_counts[key]['size'] += f['size_mb']
        if f['has_uncertainty']:
            phase_counts[key]['uncertainty'] += 1

    for phase in sorted(phase_counts.keys()):
        data = phase_counts[phase]
        lines.append(f"  {phase}: {data['count']} files ({data['size']:.1f} MB), {data['uncertainty']} with uncertainty")

    # Phase 0 breakdown
    lines.append(f"\n{'PHASE 0 - SCREENING & TUNING':=^80}")
    phase0_files = [f for f in all_files if f['phase'] == 0]

    # Models found in phase 0
    phase0_models = set()
    phase0_reps = set()
    for f in phase0_files:
        if f['model']:
            phase0_models.add(f['model'])
        if f['representation']:
            phase0_reps.add(f['representation'])

    lines.append(f"Models found: {len(phase0_models)}")
    sorted_models = sorted(m for m in phase0_models if m)
    lines.append(f"  {', '.join(sorted_models[:20])}")
    if len(sorted_models) > 20:
        lines.append(f"  ... and {len(sorted_models) - 20} more")
    lines.append(f"Representations found: {', '.join(sorted(r for r in phase0_reps if r))}")

    # Screening files specifically
    screening = [f for f in phase0_files if f['file_type'] == 'screening']
    lines.append(f"Screening files (phase0c): {len(screening)}")

    # Check for MHGGNN
    mhggnn_files = [f for f in all_files if f['representation'] == 'mhggnn' or 'mhggnn' in f['filename'].lower()]
    lines.append(f"\nMHGGNN files: {len(mhggnn_files)}")
    if mhggnn_files:
        for f in mhggnn_files[:5]:
            lines.append(f"  - {f['filename']}")

    # Phase 1-2 breakdown (Bayesian/Uncertainty)
    lines.append(f"\n{'PHASE 1-2 - BAYESIAN & UNCERTAINTY':=^80}")
    phase12_files = [f for f in all_files if f['phase'] in [1, 2]]

    # Group by model type
    bayesian_variants = defaultdict(list)
    for f in phase12_files:
        if f['model']:
            bayesian_variants[f['model']].append(f)

    lines.append(f"Total Phase 1-2 files: {len(phase12_files)}")
    lines.append("Model variants found:")
    for model in sorted(k for k in bayesian_variants.keys() if k):
        files = bayesian_variants[model]
        uncertainty_count = sum(1 for f in files if f['has_uncertainty'])
        lines.append(f"  {model}: {len(files)} files ({uncertainty_count} with uncertainty)")

    # Phase 3 breakdown (Conformal)
    lines.append(f"\n{'PHASE 3 - CONFORMAL PREDICTION':=^80}")
    phase3_files = [f for f in all_files if f['phase'] == 3]
    lines.append(f"Total Phase 3 files: {len(phase3_files)}")

    # Group by base model and calibration size
    conformal_configs = defaultdict(list)
    for f in phase3_files:
        key = (f['model'], f['representation'], f['calibration_size'])
        conformal_configs[key].append(f)

    lines.append("Configurations:")
    for (model, rep, calib), files in sorted(conformal_configs.items(), key=lambda x: (x[0][0] or '', x[0][1] or '', x[0][2] or '')):
        uncertainty_count = sum(1 for f in files if f['has_uncertainty'])
        calib_str = f"calib{calib}" if calib else "default"
        lines.append(f"  {model}/{rep}/{calib_str}: {len(files)} files ({uncertainty_count} with uncertainty)")

    # Noise conditions: the current anova_<condition>_<rep>_<model> grid AND the
    # old phase-4 files, counted together. Looking only at phase 4 meant a
    # results directory holding nothing but a finished current run reported no
    # noise runs at all.
    lines.append(f"\n{'NOISE CONDITIONS & GENERALIZATION':=^80}")
    phase4_files = [f for f in all_files
                    if f['phase'] == 4 or f['file_type'] == 'noise_grid']
    lines.append(f"Total noise-condition files: {len(phase4_files)}")

    noise_strategies = set(f['noise_strategy'] for f in phase4_files if f['noise_strategy'])
    lines.append(f"Noise conditions: {', '.join(sorted(noise_strategies)) if noise_strategies else 'None found'}")

    # Targets found
    targets = set(f['target'] for f in phase4_files if f['target'])
    lines.append(f"Targets: {', '.join(sorted(targets)) if targets else 'Default (homolumo)'}")

    # Models and reps in phase 4
    phase4_models = set(f['model'] for f in phase4_files if f['model'])
    phase4_reps = set(f['representation'] for f in phase4_files if f['representation'])
    lines.append(f"Models: {', '.join(sorted(m for m in phase4_models if m))}")
    lines.append(f"Representations: {', '.join(sorted(r for r in phase4_reps if r))}")

    # Detailed coverage matrix
    lines.append("\nCoverage matrix (noise condition x model x rep):")
    for strategy in sorted(noise_strategies):
        strategy_files = [f for f in phase4_files if f['noise_strategy'] == strategy]
        uncertainty_count = sum(1 for f in strategy_files if f['has_uncertainty'])
        lines.append(f"  {strategy}: {len(strategy_files)} files ({uncertainty_count} with uncertainty)")

    # Uncertainty files summary
    lines.append(f"\n{'UNCERTAINTY FILES SUMMARY':=^80}")
    lines.append(f"Total uncertainty value files: {len(uncertainty_files)}")
    lines.append(f"Total size: {uncertainty_size:.1f} MB ({uncertainty_size/1024:.2f} GB)")

    # Group by phase
    uncertainty_by_phase = defaultdict(list)
    for f in uncertainty_files:
        phase = f['phase'] if f['phase'] is not None else 'non-phase'
        uncertainty_by_phase[phase].append(f)

    lines.append("By phase:")
    for phase in sorted(uncertainty_by_phase.keys(), key=lambda x: (isinstance(x, str), x)):
        files = uncertainty_by_phase[phase]
        size = sum(f['size_mb'] for f in files)
        lines.append(f"  Phase {phase}: {len(files)} files ({size:.1f} MB)")

    # What's needed for the paper (gap analysis)
    lines.append(f"\n{'GAP ANALYSIS - WHAT YOU MIGHT NEED':=^80}")

    # Check for MHGGNN at high sigma
    lines.append("\n1. MHGGNN data:")
    if not mhggnn_files:
        lines.append("   ❌ NO MHGGNN FILES FOUND - Need to generate")
    else:
        lines.append(f"   ✓ Found {len(mhggnn_files)} MHGGNN files")

    # Coverage against the settled conditions, read from noise_conditions.json.
    # The main grid is expected on every pair; the depth-only conditions run on a
    # narrower selection, so they are listed apart and their absence is not a gap.
    lines.append("\n2. Noise condition coverage (main grid):")
    for condition in MAIN_GRID_CONDITIONS:
        if condition in noise_strategies:
            count = len([f for f in phase4_files if f['noise_strategy'] == condition])
            lines.append(f"   FOUND    {condition}: {count} files")
        else:
            lines.append(f"   MISSING  {condition}")
    lines.append("\n   run on a named subset of pairs, so absence is not a gap:")
    for condition in DEPTH_CONDITIONS:
        count = len([f for f in phase4_files if f['noise_strategy'] == condition])
        lines.append(f"   {'FOUND   ' if count else 'absent  '} {condition}: {count} files")
    unrecognised = sorted(noise_strategies - set(MAIN_GRID_CONDITIONS)
                          - set(DEPTH_CONDITIONS))
    if unrecognised:
        lines.append("\n   conditions no settled list names (retired, or new and "
                     "unrecorded):")
        for condition in unrecognised:
            count = len([f for f in phase4_files if f['noise_strategy'] == condition])
            lines.append(f"   ?        {condition}: {count} files")

    # Check for BNN variants
    lines.append("\n3. Bayesian Neural Network variants:")
    bnn_variants = ['bnn_full', 'bnn_last', 'bnn_variational', 'bayesian_full', 'bayesian_last']
    found_bnn = []
    for f in all_files:
        if f['model']:
            for variant in bnn_variants:
                if variant in f['model'].lower():
                    found_bnn.append(f['model'])
    found_bnn = set(found_bnn)
    if found_bnn:
        lines.append(f"   ✓ Found: {', '.join(sorted(b for b in found_bnn if b))}")
    else:
        lines.append("   ❌ No BNN variants found")

    # Check for conformal with multiple calibration sizes
    lines.append("\n4. Conformal prediction calibration sizes:")
    calib_sizes = set(f['calibration_size'] for f in phase3_files if f['calibration_size'])
    if calib_sizes:
        lines.append(f"   ✓ Found calibration sizes: {', '.join(sorted(c for c in calib_sizes if c))}")
    else:
        lines.append("   ⚠️ No calibration size variants found (only default)")

    # Subdirectories
    lines.append(f"\n{'SUBDIRECTORIES':=^80}")
    for subdir in sorted(subdirs):
        try:
            file_count = len(list(subdir.glob("*")))
            lines.append(f"  {subdir.name}/: {file_count} items")
        except:
            lines.append(f"  {subdir.name}/: (unable to count)")

    # Large files warning
    lines.append(f"\n{'LARGE FILES (>50 MB)':=^80}")
    large_files = [f for f in all_files if f['size_mb'] > 50]
    if large_files:
        for f in sorted(large_files, key=lambda x: -x['size_mb'])[:20]:
            lines.append(f"  {f['filename']}: {f['size_mb']:.1f} MB")
    else:
        lines.append("  No files > 50 MB found")

    # Recommendations
    lines.append(f"\n{'RECOMMENDATIONS':=^80}")
    lines.append("""
Based on this inventory, consider:

1. For analysis scripts: You probably only need the main CSV files (not _uncertainty_values)
   for most figures. Only download uncertainty files for specific analyses.

2. Large uncertainty files: Consider running uncertainty-specific analysis ON the server
   and only downloading the summary outputs.

3. Missing data: Check the gap analysis above and generate any missing experiments.

4. Old data cleanup: Files without 'phase' prefix might be from old experiments.
   Review the 'Non-phase files' section.
""")

    return '\n'.join(lines)


def generate_csv(all_files, output_path):
    """Generate detailed CSV for further analysis."""
    import csv

    fieldnames = ['filename', 'phase', 'subphase', 'file_type', 'model', 'representation',
                  'noise_strategy', 'target', 'calibration_size', 'has_uncertainty',
                  'has_per_epoch', 'size_mb']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for file_info in all_files:
            row = {k: file_info.get(k, '') for k in fieldnames}
            writer.writerow(row)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"

    if not os.path.isdir(results_dir):
        # Try relative to script location
        script_dir = Path(__file__).parent
        results_dir = script_dir.parent / "results"
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            sys.exit(1)

    print(f"Scanning {results_dir}...")
    all_files, subdirs = scan_results(results_dir)

    print(f"Found {len(all_files)} files, generating report...")
    report = generate_report(all_files, subdirs, results_dir)

    # Save report
    output_dir = Path(results_dir)
    report_path = output_dir / "inventory_summary.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved summary to: {report_path}")

    # Save detailed CSV
    csv_path = output_dir / "inventory_detailed.csv"
    generate_csv(all_files, csv_path)
    print(f"Saved detailed CSV to: {csv_path}")

    # Print report to console too
    print("\n" + report)


if __name__ == "__main__":
    main()
