"""Fault injection for check_reported_numbers.py.

The checker passing on the real manuscript says nothing about which edits it would reject.
Each case copies the manuscript text into a temporary directory, points a copy of the
checker at that directory, changes one number there and requires a non-zero exit. Nothing
in the manuscript or in results/ is modified.

The cases are the edits that a value transcribed into the checker cannot see: a sign that
flips, a figure replaced by another result's figure, an exponent that changes, an integer
in the same sentence, a value that collides with a penalty, and a comment added to the
checker itself. A number only present as a literal in the checker is not checked, so the
last case must fail even though the literal is there.
"""
import os, re, shutil, subprocess, sys, tempfile

PY = '/home/holiday01/miniconda/envs/drug_sc/bin/python'
ROOT = '/home/holiday01/2026_cm'
M = f'{ROOT}/manuscript'
CHECKER = f'{ROOT}/scripts/check_reported_numbers.py'
SRC = ['abstract_v3.md', 'introduction.md', 'methods.md', 'results.md', 'discussion.md',
       'figure_legends.md', 'supplementary_methods.md', 'table_notes.py',
       'Table1_cohort.csv', 'Table2_enrichment.csv']


def run(edits=(), checker_edit=None):
    """Run the checker against a copy of the manuscript; edits are (file, old, new)."""
    d = tempfile.mkdtemp(prefix='chk_numbers_')
    try:
        for f in SRC:
            shutil.copy(f'{M}/{f}', f'{d}/{f}')
        for f, a, b in edits:
            t = open(f'{d}/{f}').read()
            assert t.count(a) == 1, (f, a, t.count(a))
            open(f'{d}/{f}', 'w').write(t.replace(a, b))
        src = open(CHECKER).read().replace(f"'{M}", f"'{d}").replace(f'{{M}}', d)
        assert f"'{M}" not in src, 'the manuscript path is not fully redirected'
        if checker_edit:
            src += checker_edit
        path = f'{d}/checker.py'
        open(path, 'w').write(src)
        r = subprocess.run([PY, '-I', path], capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr
    finally:
        shutil.rmtree(d)


CASES = [
    ('a value replaced by one that appears nowhere',
     [('results.md', '−0.065', '−0.98765')], None),
    ('the sign of a correlation flipped',
     [('results.md', '−0.065', '+0.065')], None),
    ('a value replaced by another result of this study',
     [('results.md', 'ρ = −0.065', 'ρ = −0.919')], None),
    ('a value replaced by one of the penalties',
     [('results.md', '−0.065', '−0.001')], None),
    ('an exponent changed in a scientific-notation P value',
     [('results.md', '2.8 × 10⁻⁷', '2.8 × 10⁻³')], None),
    ('an integer in the same sentence changed',
     [('results.md', 'the 227 specimens', 'the 228 specimens')], None),
    ('the last digit of a group mean changed',
     [('results.md', 'led by +0.0412', 'led by +0.0413')], None),
    ('a bound loosened until it no longer binds',
     [('results.md', 'within 0.014 of each other', 'within 0.14 of each other')], None),
    ('a claim sentence rewritten so that no pattern matches',
     [('results.md', 'at Spearman ρ = −0.065',
       'at a Spearman correlation of −0.065')], None),
    ('a value replaced by one that appears nowhere, with the value added to the checker '
     'as a comment',
     [('results.md', '−0.065', '−0.98765')], '\n# unused 0.98765\n'),
]

if __name__ == '__main__':
    ok = True
    rc, out = run()
    line = [l for l in out.splitlines() if l.startswith('total:')]
    print(f'positive control, unmodified manuscript: exit={rc} | {line[0] if line else out[-200:]}')
    if rc != 0:
        ok = False
        print('  FAIL: the checker does not pass on the real manuscript')
    cov = [l for l in out.splitlines() if 'read out of the text by a claim' in l]
    if cov:
        print(' ', cov[0].strip())

    print('\ninjected edits, each must be rejected')
    for name, edits, cedit in CASES:
        rc, out = run(edits, cedit)
        caught = rc != 0
        ok &= caught
        print(f'  [{"caught " if caught else "MISSED "}] {name}')
        if not caught:
            print(f'    exit={rc}; tail of output:\n{out[-400:]}')
        else:
            why = [l.strip() for l in out.splitlines()
                   if 'manuscript=' in l or 'uncovered' in l or 'matched' in l]
            if why:
                print(f'              {why[0][:100]}')

    print(f'\n{"all cases behave as required" if ok else "SOME CASES FAILED"}')
    sys.exit(0 if ok else 1)
