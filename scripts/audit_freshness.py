"""Check that every result file the manuscript quotes is newer than the references it
depends on, and say which ones are legitimately older.

Two stale-data incidents happened without this check: robustness_check.py was left out of
the rerun list because it was filed mentally as a 'core' rather than a 'downstream'
analysis, and the healthy reference's eval output is named differently from the other two,
so an older file was picked up silently. Timestamps catch both; memory did not.
"""
import os, re, glob, sys, time
ROOT = '/home/holiday01/2026_cm'
REFS = {'programs.npz': 'healthy (never rebuilt)',
        'programs_aml_malignant_v3.npz': 'AML malignant',
        'programs_aml_all_v3.npz': 'AML all cells'}
def mt(p):
    return os.path.getmtime(p) if os.path.exists(p) else None
newest_aml = max(mt(f'{ROOT}/data/programs/{k}') or 0
                 for k in REFS if k != 'programs.npz')
print(f'newest AML reference: {time.strftime("%H:%M", time.localtime(newest_aml))}\n')
rows = []
# _superseded holds outputs from earlier reference versions, kept for the record
for f in sorted(glob.glob(f'{ROOT}/results/*.csv') + glob.glob(f'{ROOT}/results/*.json')):
    base = os.path.basename(f)
    src = None
    for s in glob.glob(f'{ROOT}/scripts/*.py'):
        txt = open(s).read()
        if base.replace('.csv', '').replace('.json', '') in txt and 'results/' in txt:
            src = os.path.basename(s); break
    uses_aml = bool(src and re.search(r'programs_aml', open(f'{ROOT}/scripts/{src}').read()))
    t = mt(f)
    stale = uses_aml and t < newest_aml
    rows.append((base, src or '?', 'AML+healthy' if uses_aml else 'healthy only',
                 time.strftime('%H:%M', time.localtime(t)), 'STALE' if stale else 'ok'))
w = max(len(r[0]) for r in rows)
print(f'{"result":{w}s}  {"written by":26s} {"depends on":12s} {"mtime":6s} {"":5s}')
for r in sorted(rows, key=lambda x: (x[4] != 'STALE', x[0])):
    print(f'{r[0]:{w}s}  {r[1]:26s} {r[2]:12s} {r[3]:6s} {r[4]}')
n = sum(1 for r in rows if r[4] == 'STALE')
print(f'\n{n} file(s) depend on an AML reference and predate the newest rebuild')
sys.exit(1 if n else 0)
