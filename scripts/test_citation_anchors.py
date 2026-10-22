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
LAMBO = ('introduction.md', 'towards more primitive states at relapse',
         'towards less primitive states at relapse')
# an anchor that is the only citation of its reference, which either check would reject
STEIGER = ('methods.md', 'with a variable in common', 'with one variable in common')
SPARE = ('references.bib', '@article{Steiger1980,',
         '@misc{Spare2099, title={Never cited}, year={2099}}\n@article{Steiger1980,')
KEY_TEX = (TEX, '"\\\\citep{Steiger1980}"', '"\\\\citep{Steiger1999}"')
RAW_TEX = ('discussion.md', '## Discussion', '## Discussion\n\nA stray \\citep{NotInBib} key.')
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
# the Results report the numbers and cite nothing, so an anchor loose enough to match a
# sentence there has to be rejected rather than quietly placed
LOOSE_TEX = (TEX, '"with the corrected resampled t-test"', '"corrected resampled t-test"')
LOOSE_DOC = (DOC, '"with the corrected resampled t-test"', '"corrected resampled t-test"')
# braces are grouping characters and a backtick is a left quote in LaTeX, so the converter
# has to escape the one and convert the other before any command braces are inserted
CODESPAN = (TEX, """    t=re.sub(r'`([^`]+)`',
             lambda m: r'\\texttt{%s}' % m.group(1).replace('_', r'\\_\\allowbreak{}'), t)""",
            '    pass')


def run(builder, edits=(), read=None):
    """Build in a throwaway copy of the manuscript; edits are (file, old, new), once each.
    With read, also return that file's content from the copy."""
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
        got = open(f'{d}/{read}').read() if read and os.path.exists(f'{d}/{read}') else ''
        return r.returncode, r.stdout + r.stderr, got
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
    ('an anchor naming a key the bibliography does not define',
     [KEY_TEX], [TEX], 'CITED BUT NOT IN BIB'),
    ('LaTeX written into the markdown, which neither build converts',
     [RAW_TEX], [TEX], 'BACKSLASH IN THE MARKDOWN SOURCE'),
    ('a citation key in the text that the bibliography does not define',
     [KEY_DOC], [DOC], 'CITED BUT NOT IN BIB'),
    ('an anchor loose enough to place a citation in the Results',
     [LOOSE_TEX], [TEX], 'CITATIONS PLACED IN THE RESULTS'),
    ('an anchor loose enough to place a citation in the Results',
     [LOOSE_DOC], [DOC], 'CITATIONS PLACED IN THE RESULTS'),
    ('the markdown substitutions in the order that leaves (**a**) as (*a*)',
     [ORDER], [TEX], 'UNCONVERTED MARKDOWN LEFT IN THE TEX'),
    ('the code spans left as backticks, which LaTeX sets as quotation marks',
     [CODESPAN], [TEX], 'UNCONVERTED MARKDOWN LEFT IN THE TEX'),
]

if __name__ == '__main__':
    ok = True
    print('positive control, unmodified sources')
    for b in (TEX, DOC):
        rc, out, _ = run(b)
        line = [l for l in out.splitlines() if 'citation' in l]
        print(f'  {b}: exit={rc} | {line[0] if line else out.splitlines()[-1]}')
        if rc != 0:
            ok = False; print('    FAIL: the builder does not run on the real manuscript')

    print('\ninjected faults, each must be rejected with its reason')
    for name, edits, builders, want in CASES:
        for b in builders:
            rc, out, _ = run(b, edits)
            caught = rc != 0 and want in out
            ok &= caught
            print(f'  [{"caught " if caught else "MISSED "}] {b}: {name}')
            if not caught:
                print(f'    exit={rc}, expected {want!r} in output:\n{out[:400]}')

    # Lambo is cited by two anchors through the same \citep, which is the collision the
    # citation-string check cannot see; the two Ng anchors now differ in \citep against \citet
    print('\ndifferential case, per-anchor bookkeeping against the citation-string check')
    rc_new, _, _ = run(TEX, [LAMBO])
    rc_old, _, _ = run(TEX, [LAMBO, OLD_CHECK])
    print(f'  current check: exit={rc_new}')
    print(f'  citation-string check only: exit={rc_old}')
    if not (rc_new != 0 and rc_old == 0):
        ok = False
        print('  FAIL: the two checks behave alike, so this case proves nothing')

    # a swallowed brace leaves no markdown behind, so it is checked on the assembled TEX
    print('\nconversions checked on the output rather than by the guard')
    _, _, tex = run(TEX, read='manuscript.tex')
    for frag, why in ((r'\{0.001, 0.01, 0.1, 1\}', 'the penalty set keeps its braces'),
                      (r'\texttt{beataml\_\allowbreak{}waves1to4',
                       'the file name is monospace and may wrap at its underscores')):
        hit = frag in tex
        ok &= hit
        print(f'  [{"ok     " if hit else "MISSING"}] {why}')

    print(f'\n{"all cases behave as required" if ok else "SOME CASES FAILED"}')
    sys.exit(0 if ok else 1)
