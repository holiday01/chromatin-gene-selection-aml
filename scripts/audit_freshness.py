"""Check that every result file the manuscript quotes is newer than the references it
depends on, and say which ones cannot be checked at all.

Two stale-data incidents happened without this check: robustness_check.py was left out of
the rerun list because it was filed mentally as a 'core' rather than a 'downstream'
analysis, and the healthy reference's eval output is named differently from the other two,
so an older file was picked up silently.

The producer of each result is found by parsing each script and reading the string
arguments of its write calls, which resolves paths assembled from f-strings and spread over
several lines. A result whose producer cannot be identified is reported as UNVERIFIED and
counted as a failure, because an unknown producer carries an unknown dependency and must
not be allowed to pass as healthy-only.
"""
import ast, os, glob, re, sys, time
ROOT = '/home/holiday01/2026_cm'
REFS = {'programs.npz': 'healthy (never rebuilt)',
        'programs_aml_malignant_v3.npz': 'AML malignant',
        'programs_aml_all_v3.npz': 'AML all cells'}
WRITE_ATTRS = {'to_csv', 'to_excel', 'savefig', 'savetxt', 'savez', 'savez_compressed',
               'to_parquet', 'write_image', 'dump'}
WRITE_FUNCS = {'ExcelWriter'}

def mt(p):
    return os.path.getmtime(p) if os.path.exists(p) else None

newest_aml = max(mt(f'{ROOT}/data/programs/{k}') or 0 for k in REFS if k != 'programs.npz')
print(f'newest AML reference: {time.strftime("%H:%M", time.localtime(newest_aml))}\n')

def literals(node):
    """Every string a call could be passing, with f-string holes turned into globs."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append(n.value)
        elif isinstance(n, ast.JoinedStr):
            s = ''.join(p.value if isinstance(p, ast.Constant) else '*' for p in n.values)
            out.append(s)
    return out

RESULT = re.compile(r'^([A-Za-z0-9_.*-]+\.(?:csv|json|xlsx))$')
existing = {os.path.basename(p) for p in glob.glob(f'{ROOT}/results/*')}

def expand(pat):
    """A path pattern to the result files it names. Only the last path segment is matched,
    because the directory is often held in a variable the parser cannot resolve."""
    m = RESULT.match(pat.rstrip('/').split('/')[-1])
    if not m: return []
    g = m.group(1)
    if '*' not in g:
        return [g] if g in existing else []
    rx = re.compile('^' + '.*'.join(re.escape(p) for p in g.split('*')) + '$')
    return [f for f in existing if rx.match(f)]

def is_write(call):
    f = call.func
    if isinstance(f, ast.Attribute) and f.attr in WRITE_ATTRS: return True
    if isinstance(f, ast.Name) and f.id in WRITE_FUNCS: return True
    if isinstance(f, ast.Name) and f.id == 'open':
        return any(isinstance(a, ast.Constant) and isinstance(a.value, str) and 'w' in a.value
                   for a in call.args[1:]) or any(
            k.arg == 'mode' and isinstance(k.value, ast.Constant) and 'w' in str(k.value.value)
            for k in call.keywords)
    return False

writers, reads = {}, {}
for s in sorted(glob.glob(f'{ROOT}/scripts/*.py')):
    b = os.path.basename(s)
    src = open(s).read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        print(f'  !! {b} does not parse; its outputs cannot be attributed')
        continue
    written = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and is_write(n):
            for lit in literals(n):
                written.update(expand(lit))
    for lit in literals(tree):
        for f in expand(lit):
            if f not in written:
                reads.setdefault(b, set()).add(f)
    for f in written:
        writers.setdefault(f, b)

def aml_dep(script, out=None, seen=()):
    """A script depends on an AML reference if it loads one, or reads a result that does.
    An unresolved read leaves the dependency unknown rather than absent.

    Several scripts take the reference as a command-line argument and name their outputs
    after it (tag = basename of the npz), so for those the output filename carries the
    reference identity and is used instead of the script source."""
    if script in seen or not os.path.exists(f'{ROOT}/scripts/{script}'):
        return None
    if out and 'aml' in out.lower():
        return True
    if re.search(r'programs_aml', open(f'{ROOT}/scripts/{script}').read()):
        return True
    got = [aml_dep(writers[r], r, seen + (script,)) if r in writers else None
           for r in sorted(reads.get(script, ()))]
    if any(g is True for g in got): return True
    if any(g is None for g in got): return None
    return False

rows = []
for f in sorted(glob.glob(f'{ROOT}/results/*.csv') + glob.glob(f'{ROOT}/results/*.json')
                + glob.glob(f'{ROOT}/results/*.xlsx')):
    base = os.path.basename(f)
    src = writers.get(base)
    dep = aml_dep(src, base) if src else None
    t = mt(f)
    if dep is None:
        flag, lbl = 'UNVERIFIED', 'unknown'
    elif dep and t < newest_aml:
        flag, lbl = 'STALE', 'AML+healthy'
    else:
        flag, lbl = 'ok', 'AML+healthy' if dep else 'healthy only'
    rows.append((base, src or '?', lbl, time.strftime('%H:%M', time.localtime(t)), flag))

w = max(len(r[0]) for r in rows)
print(f'{"result":{w}s}  {"written by":28s} {"depends on":12s} {"mtime":6s}')
order = {'STALE': 0, 'UNVERIFIED': 1, 'ok': 2}
for r in sorted(rows, key=lambda x: (order[x[4]], x[0])):
    print(f'{r[0]:{w}s}  {r[1]:28s} {r[2]:12s} {r[3]:6s} {r[4]}')
stale = [r for r in rows if r[4] == 'STALE']
unver = [r for r in rows if r[4] == 'UNVERIFIED']
print(f'\n{len(stale)} file(s) depend on an AML reference and predate the newest rebuild')
print(f'{len(unver)} file(s) have no identified producer and an unknown dependency')
sys.exit(1 if stale or unver else 0)
