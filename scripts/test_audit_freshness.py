"""Fault injection for audit_freshness.py.

The audit passing on good data says nothing about whether it rejects bad data. Each case
rewrites the audit in memory, runs the rewritten copy in a subprocess, and requires a
non-zero exit. Nothing on disk is modified.

The last case is differential rather than absolute. It forces one claimant of an ambiguous
result to look AML-dependent and makes that result old, then compares the current code,
which takes the dependency of every claimant, with a copy that keeps only the first
claimant. The result must be reported stale by the first and missed by the second;
otherwise the union over claimants is not doing anything.
"""
import os, subprocess, sys, tempfile

PY = '/home/holiday01/miniconda/envs/drug_sc/bin/python'
AUDIT = '/home/holiday01/2026_cm/scripts/audit_freshness.py'
SRC = open(AUDIT).read()

MT = "def mt(p):\n    return os.path.getmtime(p) if os.path.exists(p) else None"
WRITERS = "    for f in written:\n        writers.setdefault(f, []).append(b)"
DEPLINE = "        if re.search(r'programs_aml', open(f'{ROOT}/scripts/{script}').read()):"
LOOP = "    per = []\n    for script in scripts:"


def age(target):
    return (MT, f"def mt(p):\n    t = os.path.getmtime(p) if os.path.exists(p) else None\n"
                f"    if t and p.endswith('{target}'): t = 0.0\n    return t")


def run(mod):
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(mod); path = f.name
    r = subprocess.run([PY, '-I', path], capture_output=True, text=True)
    os.unlink(path)
    stale = [l.split()[0] for l in r.stdout.splitlines() if l.rstrip().endswith('STALE')]
    unver = [l.split()[0] for l in r.stdout.splitlines() if l.rstrip().endswith('UNVERIFIED')]
    return r.returncode, stale, unver


CASES = [
    ('an AML-dependent result predates the newest reference',
     [age('robustness_check.csv')]),
    ('a result loses its producer',
     [(WRITERS, "    for f in written:\n        if f == 'budget_sweep.csv': continue\n"
                "        writers.setdefault(f, []).append(b)")]),
]

if __name__ == '__main__':
    ok = True
    rc, stale, unver = run(SRC)
    print(f'positive control, unmodified audit: exit={rc}, '
          f'{len(stale)} stale, {len(unver)} unverified')
    if rc != 0:
        ok = False; print('  FAIL: the audit does not pass on the real tree')

    print('\ninjected failures, each must be rejected')
    for name, subs in CASES:
        mod = SRC
        for a, b in subs:
            assert a in mod, name
            mod = mod.replace(a, b)
        rc, stale, unver = run(mod)
        caught = rc != 0
        ok &= caught
        print(f'  [{"caught " if caught else "MISSED "}] {name}')
        print(f'              stale={stale} unverified={unver}')

    print('\ndifferential case, the union over ambiguous claimants')
    base = SRC.replace(*age('probe_per_drug.csv')).replace(
        DEPLINE, "        if script == 'probe_directions.py' or "
                 "re.search(r'programs_aml', open(f'{ROOT}/scripts/{script}').read()):")
    _, union, _ = run(base)
    _, first, _ = run(base.replace(LOOP, "    per = []\n    for script in scripts[:1]:"))
    gained = sorted(set(union) - set(first))
    print(f'  union over all claimants : {union}')
    print(f'  first claimant only      : {first}')
    print(f'  caught only by the union : {gained}')
    if 'probe_per_drug.csv' not in gained:
        ok = False
        print('  FAIL: keeping one claimant loses nothing, so the union is not doing work')

    print(f'\n{"all cases behave as required" if ok else "SOME CASES FAILED"}')
    sys.exit(0 if ok else 1)
