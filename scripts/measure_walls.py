#!/usr/bin/env python
"""What the jobs ACTUALLY used, and the scontrol lines to re-request it.

    python scripts/measure_walls.py                 # report
    python scripts/measure_walls.py --emit-scontrol # + the commands to apply it

WHY THIS EXISTS, AND WHY IT IS URGENT RATHER THAN TIDY.

Every wall clock in this study is a guess with a wide margin, and several were never
timed on ARC at all. That was harmless while nothing was queued. It is not harmless now:
a 22-day request cannot backfill. SLURM fits pending work into the gaps it can predict,
and a job asking for three weeks fits almost no gap, so it waits for a drain that may
never come while 4-hour jobs run past it. The walls are the queue problem.

Cutting a TimeLimit on a PENDING job is allowed to the owner and keeps the submit time,
so it costs no queue position -- unlike cancel-and-resubmit. Raising one is not, so the
margins below are deliberately generous: a job killed at the wall has no partial credit.

MEMORY IS THE SAME QUESTION AND WAS SETTLED ON ONE DATA POINT. model_memory.json rests on
a single 61.2 GB peak from an old run, because that was all anybody had. There are now
hundreds of completed tasks from THIS pipeline, so the tiers can be read off rather than
argued about.

WHAT IT REFUSES TO DO. It never proposes a wall or a memory BELOW what was measured, it
never proposes memory under the author's 64G floor, and it says how many tasks each
figure rests on -- a median over two tasks is not a measurement.
"""
import argparse
import re
import subprocess
import sys
from collections import defaultdict

GROUPS = [
    ('QM9 screen', 12971601, 12971619),
    ('laboratory breadth', 12971620, 12971638),
    ('QM9 main grid', 12980573, 12980591),
    ('QM9 deep run', 12986314, 12986332),
    ('QM9 censoring', 12986333, 12986351),
    ('laboratory depth', 12986352, 12986370),
    ('laboratory censoring', 12986371, 12986389),
    ('uncertainty, the three', 12986390, 12986395),
    ('uncertainty, the four', 12986396, 12986401),
]
BAD = ('FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'CANCELLED')


def secs(text):
    if not text or text in ('INVALID', 'UNKNOWN', 'Partition_Limit', 'UNLIMITED'):
        return None
    d, _, rest = text.partition('-')
    if not rest:
        d, rest = '0', text
    try:
        parts = [int(x) for x in rest.split(':')]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(d) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def gb(text):
    """MaxRSS / ReqMem -> GB. SLURM suffixes K, M, G, T; ReqMem may carry n or c."""
    if not text:
        return None
    m = re.match(r'^([\d.]+)\s*([KMGT]?)', text.strip())
    if not m or not m.group(1):
        return None
    v = float(m.group(1))
    return {'K': v / 1048576, 'M': v / 1024, 'G': v, 'T': v * 1024, '': v / 1048576}[m.group(2)]


def run_sacct(ids, since, fields, extra):
    try:
        p = subprocess.run(['sacct', '-M', 'arc', '-S', since, '-j', ids, '-n', '-P',
                            f'--format={fields}', *extra],
                           capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit('no sacct on this machine -- run this on the cluster.')
    return p.stdout.splitlines() if p.returncode == 0 else []


def hhmmss(s):
    d, r = divmod(int(s), 86400)
    h, r = divmod(r, 3600)
    m, _ = divmod(r, 60)
    return f'{d}-{h:02d}:{m:02d}:00' if d else f'{h}:{m:02d}:00'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', default='2026-09-01')
    ap.add_argument('--wall-margin', type=float, default=2.0,
                    help='multiply the LONGEST completed task by this (default 2.0). '
                         'A killed job has no partial credit.')
    ap.add_argument('--mem-margin', type=float, default=1.6)
    ap.add_argument('--floor-gb', type=int, default=64, help="the author's 64G floor")
    ap.add_argument('--min-tasks', type=int, default=3,
                    help='do not propose anything from fewer completed tasks than this')
    ap.add_argument('--emit-scontrol', action='store_true')
    cli = ap.parse_args()

    per = defaultdict(lambda: dict(elapsed=[], rss=[], req_t=None, req_m=None,
                                   jobs=set(), bad=defaultdict(list)))
    for name, first, last in GROUPS:
        ids = ','.join(str(j) for j in range(first, last + 1))
        rss = {}
        for line in run_sacct(ids, cli.since, 'JobID,MaxRSS', []):
            p = line.split('|')
            if len(p) == 2 and '.' in p[0] and p[1]:
                base = p[0].rsplit('.', 1)[0]
                v = gb(p[1])
                if v is not None:
                    rss[base] = max(rss.get(base, 0), v)
        for line in run_sacct(ids, cli.since,
                              'JobID,JobName,State,Elapsed,Timelimit,ReqMem', ['-X']):
            p = line.split('|')
            if len(p) < 6 or '_' not in p[0] or '[' in p[0]:
                continue
            jid, jname, state, elapsed, limit, reqmem = p[:6]
            k = (name, jname)
            d = per[k]
            d['jobs'].add(jid.split('_')[0])
            d['req_t'] = d['req_t'] or secs(limit)
            d['req_m'] = d['req_m'] or gb(reqmem)
            st = state.split()[0]
            if st == 'COMPLETED':
                s = secs(elapsed)
                if s:
                    d['elapsed'].append(s)
                if jid in rss:
                    d['rss'].append(rss[jid])
            elif st in BAD:
                d['bad'][st].append(jid)

    print(f"{'group / job':46s} {'ok':>4s} {'longest':>10s} {'asked':>11s} "
          f"{'peak GB':>8s} {'asked':>7s}")
    print('-' * 96)
    scontrol = []
    for (group, jname), d in sorted(per.items()):
        req_m_gb = d['req_m']
        n = len(d['elapsed'])
        longest = max(d['elapsed']) if d['elapsed'] else None
        peak = max(d['rss']) if d['rss'] else None
        print(f"{group[:22]:22s} {jname[:23]:23s} {n:4d} "
              f"{(hhmmss(longest) if longest else '--'):>10s} "
              f"{(hhmmss(d['req_t']) if d['req_t'] else '--'):>11s} "
              f"{(f'{peak:.1f}' if peak else '--'):>8s} "
              f"{(f'{req_m_gb:.0f}' if req_m_gb else '--'):>7s}")
        for st, jids in sorted(d['bad'].items()):
            print(f"{'':46s} !! {len(jids):d} {st}: {', '.join(jids[:4])}"
                  + (' ...' if len(jids) > 4 else ''))
        if n < cli.min_tasks or not longest:
            continue
        want_t = int(longest * cli.wall_margin) + 3600
        want_m = max(cli.floor_gb,
                     int((peak or 0) * cli.mem_margin / 16 + 1) * 16) if peak else None
        if d['req_t'] and want_t < d['req_t'] * 0.75:
            for j in sorted(d['jobs']):
                scontrol.append(f"scontrol update JobId={j} TimeLimit={hhmmss(want_t)}"
                                f"   # {jname}: longest of {n} was {hhmmss(longest)}")
        if want_m and d['req_m'] and want_m < d['req_m'] * 0.9:
            for j in sorted(d['jobs']):
                scontrol.append(f"scontrol update JobId={j} MinMemoryNode={want_m * 1024}"
                                f"   # {jname}: peak of {n} was {peak:.1f} GB")

    print(f"\n  Walls proposed at {cli.wall_margin}x the LONGEST completed task plus an "
          f"hour;\n  memory at {cli.mem_margin}x the peak, rounded up to 16 GB, never "
          f"below {cli.floor_gb}G.\n  Nothing is proposed from fewer than "
          f"{cli.min_tasks} completed tasks.")
    if scontrol:
        print(f"\n  {len(scontrol)} change(s) worth making. Cutting a TimeLimit on a "
              f"PENDING job keeps\n  its submit time, so it costs no queue position -- "
              f"and a smaller request is\n  what lets SLURM backfill it at all.")
        if cli.emit_scontrol:
            print()
            for line in scontrol:
                print(f"  {line}")
        else:
            print(f"  Re-run with --emit-scontrol to print them.")
    else:
        print("\n  Nothing worth changing on the evidence so far.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
