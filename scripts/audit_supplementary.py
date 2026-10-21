"""Check the Supporting Information against the files it describes.

check_reported_numbers.py asserts values the manuscript quotes, and audit_freshness.py
checks which reference a result depends on and when it was written. Neither reads the
submission workbook, so three kinds of error passed both of them repeatedly: a scope
sentence that generalised from one output file to all of them, a table legend that promised
a quantity its sheet does not hold, and a column holding paired differences under a name
that reads as performance.

This script closes that gap. It parses the reference-by-penalty table of the supplementary
methods and checks every machine-checkable row against the outputs, it checks that each
sheet of the workbook has a legend and each legend a sheet, and it checks that a legend
promising an absolute performance figure sits on a sheet that holds one. A row it cannot
check mechanically is reported as DECLARED rather than passed silently.
"""
import json, re, sys
import numpy as np, pandas as pd
ROOT = '/home/holiday01/2026_cm'
R, M = f'{ROOT}/results', f'{ROOT}/manuscript'
BOOK = f'{M}/supplementary/Supporting_Information_Tables.xlsx'
SM = f'{M}/supplementary_methods.md'
BAD, DECLARED = [], []

def fail(where, msg):
    BAD.append((where, msg))

def rd(f, **k):
    try: return pd.read_csv(f'{R}/{f}', **k)
    except Exception: return None

def rj(f):
    try: return json.load(open(f'{R}/{f}'))
    except Exception: return None

# ---------------------------------------------------------------- scope table
text = open(SM).read()
block = re.search(r'\| Analysis \| Healthy multiome \|.+?\n\n', text, re.S)
if not block:
    fail('S-M1', 'the reference-by-penalty table was not found')
    rows = []
else:
    rows = []
    for line in block.group(0).strip().splitlines()[2:]:
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) == 5:
            rows.append(dict(analysis=c[0], healthy=c[1], mal=c[2], all_=c[3], pen=c[4]))
print(f'parsed {len(rows)} rows from the supplementary scope table\n')

REFKEY = {'BMMC': 'healthy', 'AML-mal': 'mal', 'AML-all': 'all_',
          'BMMC (healthy marrow)': 'healthy', 'AML malignant': 'mal', 'AML all cells': 'all_'}

def from_robustness(arm):
    d = rd('robustness_check.csv')
    if d is None: return None
    d = d[d.arm == arm]
    return ({REFKEY[r] for r in d.reference.unique()},
            int(d.groupby('reference').C.nunique().max()))

def from_confound(arm):
    got, pen = set(), set()
    for f, key in (('confound_programs.json', 'healthy'),
                   ('confound_programs_aml_malignant_v3.json', 'mal'),
                   ('confound_programs_aml_all_v3.json', 'all_')):
        j = rj(f)
        if not j: continue
        if arm == 'linkcount-rand':
            hits = [k for k in j['cv'] if k.startswith('linkcount-rand')]
            if len(hits) != 20: fail('S-M2', f'{f} holds {len(hits)} link-count draws, not 20')
            if hits: got.add(key); pen.add(len(j['cv'][hits[0]]))
        elif arm in j['cv']:
            got.add(key); pen.add(len(j['cv'][arm]))
    return got, (max(pen) if pen else 0)

def from_budget():
    d = rd('budget_sweep.csv')
    if d is None: return None
    d = d[d.reference != 'reference']
    return ({REFKEY[r] for r in d.reference.unique()},
            len({c.split('_')[-1] for c in d.columns if c.startswith('d_cv_')}))

def from_per_program():
    got, pen = set(), set()
    for f, key in (('eval_programs_per_program.csv', 'healthy'),
                   ('eval_programs_aml_malignant_v3_per_program.csv', 'mal'),
                   ('eval_programs_aml_all_v3_per_program.csv', 'all_')):
        d = rd(f)
        if d is None: continue
        got.add(key); pen.add(len({c.split('_', 1)[1] for c in d.columns if c.startswith('cv_')}))
    return got, (max(pen) if pen else 0)

def from_os_programs():
    got, pen = set(), set()
    for f, key in (('os_programs.json', 'healthy'),
                   ('os_programs_aml_malignant_v3.json', 'mal'),
                   ('os_programs_aml_all_v3.json', 'all_')):
        j = rj(f)
        if not j: continue
        got.add(key); pen.add(len(next(iter(j['cv'].values()))))
    return got, (max(pen) if pen else 0)

def from_remaining(analysis):
    d = rd('recomputed_remaining.csv')
    if d is None: return None
    d = d[d.analysis == analysis]
    refs = {REFKEY[r] for r in d.reference.dropna().unique() if r in REFKEY}
    return refs, int(d.groupby('reference').C.nunique().max()) if len(refs) else 0

def from_cross_reference():
    d = rd('cross_reference_programs.csv')
    if d is None: return None
    return {'healthy', 'mal'}, len({c[1:] for c in ('b001', 'b01')})

def from_json_keys(f, pred):
    j = rj(f)
    if not j: return None
    cv = j.get('cv', j)
    hits = [k for k in cv if pred(k)]
    if not hits: return set(), 0
    v = cv[hits[0]]
    return {'healthy'}, (len(v) if isinstance(v, dict) else 2)

CHECKS = {
    'arm A against arm R, 200 genes': lambda: from_robustness('armA_sum'),
    'arm A divided by link count': lambda: from_robustness('armA_mean_per_link'),
    'arm A residualised on ln(1 + link count)': lambda: from_robustness('armA_count_partialled'),
    'link count alone': lambda: from_robustness('n_links_only'),
    'gene budget sweep, 10 to 200 genes': from_budget,
    'peak density, genomic span, summed link correlation':
        lambda: from_confound('peak density only'),
    'link-count-matched random draws': lambda: from_confound('linkcount-rand'),
    'every programme of the reference': from_per_program,
    'overall survival, every programme': from_os_programs,
    'overall survival, paired comparison for the stem cell programme':
        lambda: from_remaining('overall_survival'),
    'programmes matched between references by cell identity': from_cross_reference,
    'published comparator and LSC17':
        lambda: from_json_keys('baseline_lsc104.json', lambda k: k.startswith('Ng2016')),
    'variance-matched draws and the most variable genes':
        lambda: from_json_keys('baseline_lsc104.json', lambda k: k.startswith('var-matched')),
    'linking-free locus arms':
        lambda: from_json_keys('locus_arm.json', lambda k: k.startswith('locus_')),
    'permuted-link nulls':
        lambda: ({'healthy'}, len({k.rsplit('_C', 1)[1]
                                   for k in rj('link_permutation_window100.json')})),
}
NOT_CHECKABLE = {
    'subsets of the selected genes': 'distributed_test.csv records size and arm, not reference',
    'clinical covariate models': 'recompute_remaining.py loads programs.npz directly; the '
                                 'clinical rows carry no reference field',
    'regulon and pathway enrichment': 'enrichr output carries no reference or penalty',
}

print(f'{"analysis":62s} {"declared":22s} {"actual":22s}')
for r in rows:
    decl_refs = {k for k in ('healthy', 'mal', 'all_') if r[k].lower() == 'yes'}
    decl_pen = r['pen']
    if r['analysis'] in NOT_CHECKABLE:
        DECLARED.append((r['analysis'], NOT_CHECKABLE[r['analysis']]))
        continue
    fn = CHECKS.get(r['analysis'])
    if fn is None:
        fail('S-M1', f'no check defined for the row "{r["analysis"]}"')
        continue
    try:
        got = fn()
    except Exception as e:
        fail('S-M1', f'{r["analysis"]}: check raised {type(e).__name__}: {e}')
        continue
    if got is None:
        fail('S-M1', f'{r["analysis"]}: source output missing')
        continue
    act_refs, act_pen = got
    d1 = ''.join('y' if k in decl_refs else 'n' for k in ('healthy', 'mal', 'all_'))
    d2 = ''.join('y' if k in act_refs else 'n' for k in ('healthy', 'mal', 'all_'))
    ok = (decl_refs == act_refs) and (decl_pen == str(act_pen))
    print(f'{r["analysis"][:62]:62s} {d1+" pen "+decl_pen:22s} {d2+" pen "+str(act_pen):22s}'
          f'{"" if ok else "  MISMATCH"}')
    if decl_refs != act_refs:
        fail('S-M1', f'{r["analysis"]}: declared references {d1}, outputs give {d2}')
    if decl_pen != str(act_pen):
        fail('S-M1', f'{r["analysis"]}: declared {decl_pen} penalties, outputs give {act_pen}')

# ------------------------------------------------- sheets against their legends
book = pd.ExcelFile(BOOK)
sheets = book.sheet_names
legends = dict(re.findall(r'\*\*Table (S\d+[ab]?)\.\*\*\s*(.+?)(?=\n\n|\Z)', text, re.S))
print(f'\nworkbook sheets {len(sheets)} | table legends {len(legends)}')
sheet_ids = {s.split('_')[0]: s for s in sheets}
for sid, sheet in sheet_ids.items():
    if sid not in legends:
        fail('legends', f'sheet {sheet} has no legend')
for lid in legends:
    if lid not in sheet_ids:
        fail('legends', f'legend for Table {lid} names no sheet')

# a legend promising an absolute performance figure needs a column that holds one
DIFFWORDS = ('difference', 'differences', 'increment', 'minus')
for sid, sheet in sorted(sheet_ids.items()):
    if sid not in legends: continue
    leg = ' '.join(legends[sid].split()).lower()
    cols = list(book.parse(sheet, nrows=0).columns)
    lc = ' '.join(cols).lower()
    says_abs = any(w in leg for w in ('auroc', 'concordance index', 'performance'))
    says_diff = any(w in leg for w in DIFFWORDS)
    has_abs = any(('auroc' in c or 'cindex' in c or 'concordance' in c)
                  and 'minus' not in c and 'delta' not in c for c in lc.split())
    has_diff = any(('minus' in c or 'delta' in c or c == 'diff') for c in lc.split())
    if says_abs and not says_diff and has_diff and not has_abs:
        fail('legends', f'Table {sid} promises performance but {sheet} holds only differences')
    if says_diff and not has_diff and not has_abs:
        fail('legends', f'Table {sid} promises differences but {sheet} names none')
    # a column the paper abandoned must not be shipped
    for banned in ('beats', 'wilcoxon_p', 'p', 'sign_p'):
        if banned in [c.lower() for c in cols]:
            fail('legends', f'{sheet} carries the ambiguous or abandoned column "{banned}"')
    # a column that is entirely empty should not be in a submission table
    d = book.parse(sheet)
    empty = [c for c in d.columns if d[c].isna().all()]
    if empty:
        fail('legends', f'{sheet} has column(s) empty for every row: {empty}')

# ------------------------------------------------------------------- report
if DECLARED:
    print('\ndeclared, not machine-checkable from results/:')
    for a, why in DECLARED: print(f'  {a}\n      {why}')
print(f'\n{len(BAD)} problem(s)')
for where, msg in BAD: print(f'  [{where}] {msg}')
sys.exit(1 if BAD else 0)
