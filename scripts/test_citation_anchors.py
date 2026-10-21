"""Fault injection for the citations the two manuscript builders insert.

Both builders place a citation by matching a sentence fragment in the markdown, so an
edit to that sentence drops the citation with nothing else to show for it. Each case
copies the manuscript sources and both builders into a temporary directory, injects one
fault there, and requires a non-zero exit carrying the expected reason. Nothing in the
manuscript is modified.

The differential case is the regression this file exists for. Asking only whether the
citation string reached the document passes whenever a second anchor cites the same
reference, so a lost anchor is caught only by recording each anchor on its own.
"""
import os, shutil, subprocess, sys, tempfile

PY = '/home/holiday01/miniconda/envs/drug_sc/bin/python'
M = '/home/holiday01/2026_cm/manuscript'
SRC = ['abstract_v3.md', 'introduction.md', 'methods.md', 'results.md', 'discussion.md',
       'figure_legends.md', 'statements.md', 'table_notes.py', 'references.bib',
       'Table1_cohort.csv', 'Table2_enrichment.csv', 'build_tex.py', 'build_docx.py']

TEX, DOC = 'build_tex.py', 'build_docx.py'

# anchors whose reference stays cited somewhere else, so only per-anchor bookkeeping sees them
NG = ('introduction.md', 'in independent clinical cohorts', 'in independent patient cohorts')
LAMBO = ('introduction.md', 'that persist through treatment', 'that endure through treatment')
# an anchor that is the only citation of its reference, which either check would reject
STEIGER = ('methods.md', 'with a variable in common', 'with one variable in common')
SPARE = ('references.bib', '@article{Steiger1980,',
         '@misc{Spare2099, title={Never cited}, year={2099}}\n@article{Steiger1980,')
KEY_TEX = ('discussion.md', '## Discussion', '## Discussion\n\nA stray \\citep{NotInBib} key.')
KEY_DOC = ('discussion.md', '## Discussion', '## Discussion\n\nA stray [@NotInBib] key.')
# the panel letters of every figure legend are written (**a**), so the order of the two
# markdown substitutions decides whether the PDF prints the asterisks
ORDER = (TEX,
         "    t=re.sub(r'\\*\\*(.+?)\\*\\*', r'\\\\textbf{\\1}', t, flags=re.S)\n"
         "    t=re.sub(r'\\*([A-Za-z])\\*', r'\\\\textit{\\1}', t)",
         "    t=re.sub(r'\\*([A-Za-z])\\*', r'\\\\textit{\\1}', t)\n"
         "    t=re.sub(r'\\*\\*(.+?)\\*\\*', r'\\\\textbf{\\1}', t, flags=re.S)")
OLD_CHECK = (TEX, 'missing=[e[0] for e in CITE if e[0] not in MATCHED]',
             'missing=[e[0] for e in CITE if e[1] not in TEX]')


def run(builder, edits=()):
    """Build in a throwaway copy of the manuscript; edits are (file, old, new), once each."""
    d = tempfile.mkdtemp(prefix='cite_anchors_')
    try:
        for f in SRC:
            shutil.copy(f'{M}/{f}', f'{d}/{f}')
        os.symlink(f'{M}/figures', f'{d}/figures')
        for f, a, b in edits:
            t = open(f'{d}/{f}').read()
            assert t.count(a) >= 1, (f, a)
            open(f'{d}/{f}', 'w').write(t.replace(a, b, 1))
        r = subprocess.run([PY, '-I', builder], cwd=d, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr
    finally:
        shutil.rmtree(d)


CASES = [
    ('the Ng anchor in the Introduction, with Ng still cited in Methods',
     [NG], [TEX, DOC], 'ANCHOR NOT FOUND'),
    ('the Lambo anchor in the Introduction, with Lambo still cited in Methods',
     [LAMBO], [TEX, DOC], 'ANCHOR NOT FOUND'),
    ('the Steiger anchor, the only citation of that reference',
     [STEIGER], [TEX, DOC], 'ANCHOR NOT FOUND'),
    ('a bibliography entry that no anchor cites',
     [SPARE], [TEX, DOC], 'BIB ENTRY NEVER CITED'),
    ('a citation key in the text that the bibliography does not define',
     [KEY_TEX], [TEX], 'CITED BUT NOT IN BIB'),
    ('a citation key in the text that the bibliography does not define',
     [KEY_DOC], [DOC], 'CITED BUT NOT IN BIB'),
    ('the markdown substitutions in the order that leaves (**a**) as (*a*)',
     [ORDER], [TEX], 'MARKDOWN ASTERISK LEFT IN THE TEX'),
]

if __name__ == '__main__':
    ok = True
    print('positive control, unmodified sources')
    for b in (TEX, DOC):
        rc, out = run(b)
        line = [l for l in out.splitlines() if 'citation' in l]
        print(f'  {b}: exit={rc} | {line[0] if line else out.splitlines()[-1]}')
        if rc != 0:
            ok = False; print('    FAIL: the builder does not run on the real manuscript')

    print('\ninjected faults, each must be rejected with its reason')
    for name, edits, builders, want in CASES:
        for b in builders:
            rc, out = run(b, edits)
            caught = rc != 0 and want in out
            ok &= caught
            print(f'  [{"caught " if caught else "MISSED "}] {b}: {name}')
            if not caught:
                print(f'    exit={rc}, expected {want!r} in output:\n{out[:400]}')

    # Lambo is cited by two anchors through the same \citep, which is the collision the
    # citation-string check cannot see; the two Ng anchors now differ in \citep against \citet
    print('\ndifferential case, per-anchor bookkeeping against the citation-string check')
    rc_new, _ = run(TEX, [LAMBO])
    rc_old, out_old = run(TEX, [LAMBO, OLD_CHECK])
    print(f'  current check: exit={rc_new}')
    print(f'  citation-string check only: exit={rc_old}')
    if not (rc_new != 0 and rc_old == 0):
        ok = False
        print('  FAIL: the two checks behave alike, so this case proves nothing')

    print(f'\n{"all cases behave as required" if ok else "SOME CASES FAILED"}')
    sys.exit(0 if ok else 1)
