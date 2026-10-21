"""Check the Supporting Information against the files it describes.

check_reported_numbers.py asserts values the manuscript quotes, and audit_freshness.py
checks which reference a result depends on and when it was written. Neither reads the
submission workbook, so three kinds of error passed both of them repeatedly: a scope
sentence that generalised from one output file to all of them, a table legend that promised
a quantity its sheet does not hold, and a column holding paired differences under a name
that reads as performance.

This script parses the reference-by-penalty table of the supplementary methods and checks
every machine-checkable row against the outputs, checks that each sheet of the workbook has
a legend and each legend a sheet, and checks that a legend promising a performance figure
sits on a sheet that holds one. A row it cannot check mechanically is reported as DECLARED
rather than passed silently.

Every check returns the penalties it found per reference, not a count pooled over
references, so one reference missing a penalty cannot be masked by another. A row naming
several analyses is checked for each of them. test_audit_supplementary.py injects the
failures these rules exist to catch and requires a non-zero exit for each.
"""
import json, re, sys
import numpy as np, pandas as pd

ROOT = '/home/holiday01/2026_cm'
R, M = f'{ROOT}/results', f'{ROOT}/manuscript'
BOOK = f'{M}/supplementary/Supporting_Information_Tables.xlsx'
SM = f'{M}/supplementary_methods.md'
REFS = ('healthy', 'mal', 'all_')
REFKEY = {'BMMC': 'healthy', 'AML-mal': 'mal', 'AML-all': 'all_',
          'BMMC (healthy marrow)': 'healthy', 'AML malignant': 'mal',
          'AML all cells': 'all_'}


def rd(f, **k):
    """Read a result CSV. Replaced by the fault-injection test."""
    return pd.read_csv(f'{R}/{f}', **k)


def rj(f):
    """Read a result JSON. Replaced by the fault-injection test."""
    return json.load(open(f'{R}/{f}'))


def book():
    """Open the submission workbook. Replaced by the fault-injection test."""
    return pd.ExcelFile(BOOK)


def _try(fn, *a):
    try:
        return fn(*a)
    except FileNotFoundError:
        return None
    except Exception:
        return None


# --------------------------------------------------------------------- coverage
# Each function returns {reference: set of penalties} for the analysis it names, or None if
# its source is unreadable. An empty set for a reference means the reference appears with no
# usable penalty, which is a failure, not an absence.

def cov_robustness(arm):
    d = _try(rd, 'robustness_check.csv')
    if d is None: return None
    d = d[d.arm == arm]
    if not len(d): return {}
    return {REFKEY[r]: set(g.C.dropna().unique())
            for r, g in d.groupby('reference') if r in REFKEY}


def cov_confound(arm):
    out = {}
    for f, key in (('confound_programs.json', 'healthy'),
                   ('confound_programs_aml_malignant_v3.json', 'mal'),
                   ('confound_programs_aml_all_v3.json', 'all_')):
        j = _try(rj, f)
        if j is None: continue
        cv = j.get('cv', {})
        if arm == 'linkcount-rand':
            hits = sorted(k for k in cv if k.startswith('linkcount-rand'))
            if len(hits) != 20:
                return {'__error__': f'{f} holds {len(hits)} link-count draws, not 20'}
            # every draw must carry the same penalties, not just the first
            pens = {frozenset(cv[k]) for k in hits}
            if len(pens) != 1:
                return {'__error__': f'{f} link-count draws differ in their penalties'}
            out[key] = set(next(iter(pens)))
        elif arm in cv:
            out[key] = set(cv[arm])
        else:
            out[key] = set()
    return out


def cov_budget():
    d = _try(rd, 'budget_sweep.csv')
    if d is None: return None
    d = d[d.reference != 'reference']
    pen = sorted({c.split('d_cv_')[1] for c in d.columns if c.startswith('d_cv_')})
    out = {}
    for r, g in d.groupby('reference'):
        if r not in REFKEY: continue
        out[REFKEY[r]] = {p for p in pen
                          if pd.to_numeric(g[f'd_cv_{p}'], errors='coerce').notna().any()}
    return out


def cov_per_program():
    out = {}
    for f, key in (('eval_programs_per_program.csv', 'healthy'),
                   ('eval_programs_aml_malignant_v3_per_program.csv', 'mal'),
                   ('eval_programs_aml_all_v3_per_program.csv', 'all_')):
        d = _try(rd, f)
        if d is None: continue
        out[key] = {c.split('cv_')[1] for c in d.columns
                    if c.startswith('cv_') and d[c].notna().any()}
    return out


def cov_os_programs():
    out = {}
    for f, key in (('os_programs.json', 'healthy'),
                   ('os_programs_aml_malignant_v3.json', 'mal'),
                   ('os_programs_aml_all_v3.json', 'all_')):
        j = _try(rj, f)
        if j is None: continue
        pens = {frozenset(v) for v in j.get('cv', {}).values()}
        out[key] = set(next(iter(pens))) if len(pens) == 1 else set()
    return out


def cov_remaining(analysis):
    d = _try(rd, 'recomputed_remaining.csv')
    if d is None: return None
    d = d[d.analysis == analysis]
    return {REFKEY[r]: set(g.C.dropna().unique())
            for r, g in d.groupby('reference') if r in REFKEY}


def cov_cross_reference():
    """Read the columns, do not assume them. b* are the healthy reference, a* the
    malignant-cell one, and the suffix names the penalty."""
    d = _try(rd, 'cross_reference_programs.csv')
    if d is None: return None
    SUF = {'001': '0.01', '01': '0.1'}
    out = {}
    for pre, key in (('b', 'healthy'), ('a', 'mal')):
        pens = set()
        for suf, pen in SUF.items():
            col = f'{pre}{suf}'
            if col in d.columns and pd.to_numeric(d[col], errors='coerce').notna().any():
                pens.add(pen)
        if pens or any(c.startswith(pre) for c in d.columns):
            out[key] = pens
    return out


def cov_json_arms(f, prefix):
    j = _try(rj, f)
    if j is None: return None
    cv = j.get('cv', j)
    hits = sorted(k for k in cv if k.startswith(prefix))
    if not hits: return {'healthy': set()}
    pens = {frozenset(cv[k]) for k in hits if isinstance(cv[k], dict)}
    if len(pens) != 1:
        return {'__error__': f'{f} arms starting "{prefix}" differ in their penalties'}
    return {'healthy': set(next(iter(pens)))}


def cov_permutation():
    out = set()
    for f in ('link_permutation_window100.json',):
        j = _try(rj, f)
        if j is None: return None
        for k in j:
            if '_C' in k: out.add(k.rsplit('_C', 1)[1])
    return {'healthy': out}


# A row may name several analyses. Every one of them is checked, and the row passes only if
# all of them agree with the declaration.
CHECKS = {
    'arm A against arm R, 200 genes': [lambda: cov_robustness('armA_sum')],
    'arm A divided by link count': [lambda: cov_robustness('armA_mean_per_link')],
    'arm A residualised on ln(1 + link count)': [
        lambda: cov_robustness('armA_count_partialled')],
    # the four-penalty paired comparison the main text reports
    'link count alone': [lambda: cov_robustness('n_links_only')],
    'gene budget sweep, 10 to 200 genes': [cov_budget],
    # the two-penalty connectivity panel, where link count is one member
    'peak density, genomic span, summed link correlation, link count': [
        lambda: cov_confound('peak density only'),
        lambda: cov_confound('gene span only'),
        lambda: cov_confound('sum_r only'),
        lambda: cov_confound('n_links only')],
    'link-count-matched random draws': [lambda: cov_confound('linkcount-rand')],
    'every programme of the reference': [cov_per_program],
    'overall survival, every programme': [cov_os_programs],
    'overall survival, paired comparison for the stem cell programme': [
        lambda: cov_remaining('overall_survival')],
    'programmes matched between references by cell identity': [cov_cross_reference],
    'published comparator and LSC17': [
        lambda: cov_json_arms('baseline_lsc104.json', 'Ng2016'),
        lambda: cov_json_arms('baseline_lsc17.json', 'LSC17')],
    'variance-matched draws and the most variable genes': [
        lambda: cov_json_arms('baseline_lsc104.json', 'var-matched'),
        lambda: cov_json_arms('baseline_lsc104.json', 'top-')],
    'linking-free locus arms': [lambda: cov_json_arms('locus_arm.json', 'locus_')],
    'permuted-link nulls': [cov_permutation],
}
# The penalty values each row declares, not merely how many. A set of the right size with a
# wrong value in it passed the earlier version, which compared len() alone. Classification
# uses C and the Cox models use alpha, so the two grids are declared separately and never
# pooled. Values are compared as floats after normalisation, because the sources store them
# as JSON keys, CSV column suffixes and numeric columns.
PENALTIES = {
    'arm A against arm R, 200 genes': {0.001, 0.01, 0.1, 1},
    'arm A divided by link count': {0.001, 0.01, 0.1, 1},
    'arm A residualised on ln(1 + link count)': {0.001, 0.01, 0.1, 1},
    'link count alone': {0.001, 0.01, 0.1, 1},
    'gene budget sweep, 10 to 200 genes': {0.01, 0.1},
    'peak density, genomic span, summed link correlation, link count': {0.01, 0.1},
    'link-count-matched random draws': {0.01, 0.1},
    'every programme of the reference': {0.01, 0.1},
    'overall survival, every programme': {1.0, 0.1},
    'overall survival, paired comparison for the stem cell programme': {100.0, 10.0, 1.0, 0.1},
    'programmes matched between references by cell identity': {0.01, 0.1},
    'published comparator and LSC17': {0.01, 0.1},
    'variance-matched draws and the most variable genes': {0.01, 0.1},
    'linking-free locus arms': {0.01, 0.1},
    'permuted-link nulls': {0.01, 0.1},
}
NOT_CHECKABLE = {
    'subsets of the selected genes': 'distributed_test.csv records size and arm, not reference',
    'clinical covariate models': 'recompute_remaining.py loads programs.npz directly; the '
                                 'clinical rows carry no reference field',
    'regulon and pathway enrichment': 'enrichr output carries no reference or penalty',
}

# Which columns of each sheet hold an absolute performance figure and which hold a paired
# difference. Stated per sheet rather than guessed, so a sheet that loses its performance
# columns fails whether or not it still has difference columns.
# The exact columns of each sheet, declared rather than inferred. Checking that a sheet has
# "some column holding a difference" is a weaker property than the schema it is meant to
# have: a sheet could lose one arm's performance column, or gain a column, and still satisfy
# it. Set equality against this manifest catches a removal, an addition and a rename, and it
# subsumes the ban on shipping a column the paper abandoned or a name ambiguous between two
# statistics. Changing a column here is deliberate; the audit then records the submission's
# schema. A sheet with no declaration fails.
SHEET_COLUMNS = {
    'S1': ['reference', 'arm', 'rank', 'gene', 'selection_score', 'n_linked_peaks', 'in_Ng2016_LSC104', 'in_Ng2016_LSC17'],
    'S2': ['reference', 'program', 'lineage', 'A_minus_R_cv_C0.01', 'A_minus_R_xwave_C0.01', 'A_minus_R_cv_C0.1', 'A_minus_R_xwave_C0.1'],
    'S3': ['tag', 'window', 'min_r', 'top_m', 'n_links', 'eligible', 'nA', 'nR', 'armA_cv_AUROC_C0.001', 'armA_cv_AUROC_C0.01', 'armA_cv_AUROC_C0.1', 'armA_cv_AUROC_C1', 'armR_cv_AUROC_C0.001', 'armR_cv_AUROC_C0.01', 'armR_cv_AUROC_C0.1', 'armR_cv_AUROC_C1', 'variance_matched_cv_AUROC_C0.001', 'variance_matched_cv_AUROC_C0.01', 'variance_matched_cv_AUROC_C0.1', 'variance_matched_cv_AUROC_C1'],
    'S4': ['reference', 'k', 'n_A', 'A_cv_AUROC_C0.01', 'R_cv_AUROC_C0.01', 'A_minus_R_cv_C0.01', 'A_minus_R_xwave_C0.01', 'A_cv_AUROC_C0.1', 'R_cv_AUROC_C0.1', 'A_minus_R_cv_C0.1', 'A_minus_R_xwave_C0.1'],
    'S5': ['reference', 'measure', 'arm', 'n', 'n_links', 'mean_r', 'span_kb', 'peak_density', 'cv_AUROC_C0.01', 'cv_AUROC_C0.1', 'xwave_AUROC_C0.01', 'xwave_AUROC_C0.1'],
    'S6a': ['null', 'setting', 'n_permutations', 'real_cv_AUROC', 'null_cv_AUROC_mean', 'null_cv_AUROC_sd', 'empirical_p_cv', 'real_xwave_AUROC', 'null_xwave_AUROC_mean', 'n_perms_real_higher_xwave', 'gene_overlap_median', 'gene_overlap_min', 'gene_overlap_max', 'distinct_peaks_real', 'distinct_peaks_permuted_median'],
    'S6b': ['arm', 'gene_overlap_with_armA', 'cv_AUROC_C0.01', 'cv_AUROC_C0.1', 'xwave_AUROC_C0.01', 'xwave_AUROC_C0.1'],
    'S7a': ['gene_set', 'library', 'term', 'enrichr_p', 'odds_ratio', 'enrichr_combined_score', 'enrichr_q_BH', 'n_overlap', 'genes'],
    'S7b': ['reference', 'n_ref_in_pool', 'gene_set', 'n_in_gene_set', 'n_overlap', 'n_overlap_expected', 'odds_ratio', 'fisher_one_sided_p', 'genes'],
    'S8a': ['symbol', 'entrez', 'log2fc', 'signature'],
    'S8b': ['table_symbol', 'entrez', 'signature', 'current', 'beataml'],
    'S9a': ['reference', 'arm', 'endpoint', 'n', 'events', 'cv_cindex_alpha1.0', 'cv_cindex_alpha0.1', 'xwave_cindex_alpha1.0', 'xwave_cindex_alpha0.1'],
    'S9b': ['reference', 'alpha', 'A_minus_R_cindex', 'corrected_t', 'corrected_p'],
    'S10': ['C', 'quantity', 'AUROC', 'delta_AUROC', 'comparison', 'corrected_t', 'corrected_p'],
    'S11a': ['reference_and_penalty', 'n_programmes', 'n_programmes_favouring_armA', 'mean_A_minus_R', 'median_A_minus_R', 'sign_test_p_over_programmes', 'wilcoxon_signedrank_p_over_programmes'],
    'S11b': ['healthy_programme', 'malignant_programme', 'healthy_A_minus_R_C0.01', 'malignant_A_minus_R_C0.01', 'healthy_A_minus_R_C0.1', 'malignant_A_minus_R_C0.1', 'mean_over_four_cv_cells', 'positive_in_all_four', 'myeloid_stem_progenitor'],
    'S12a': ['step', 'n'],
    'S12b': ['variable', 'denominator', 'excluded', 'retained'],
}


def check_scope(text, bad, declared):
    blk = re.search(r'\| Analysis \| Healthy multiome \|.+?\n\n', text, re.S)
    if not blk:
        bad.append(('S-M1', 'the reference-by-penalty table was not found'))
        return []
    rows = []
    for line in blk.group(0).strip().splitlines()[2:]:
        c = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(c) == 5:
            rows.append(dict(analysis=c[0], healthy=c[1], mal=c[2], all_=c[3], pen=c[4]))
    printed = []
    for r in rows:
        decl_refs = {k for k in REFS if r[k].lower() == 'yes'}
        if r['analysis'] in NOT_CHECKABLE:
            declared.append((r['analysis'], NOT_CHECKABLE[r['analysis']]))
            continue
        fns = CHECKS.get(r['analysis'])
        if fns is None:
            bad.append(('S-M1', f'no check defined for the row "{r["analysis"]}"'))
            continue
        try:
            want_pen = int(r['pen'])
        except ValueError:
            bad.append(('S-M1', f'{r["analysis"]}: penalty column is not a number'))
            continue
        want_set = PENALTIES.get(r['analysis'])
        if want_set is None:
            bad.append(('S-M1', f'{r["analysis"]}: no declared penalty set'))
        elif len(want_set) != want_pen:
            bad.append(('S-M1', f'{r["analysis"]}: the table says {want_pen} penalties but '
                                f'the declared set holds {len(want_set)}'))
        shown = None
        for i, fn in enumerate(fns):
            got = fn()
            if got is None:
                bad.append(('S-M1', f'{r["analysis"]}: source output {i} missing'))
                continue
            if '__error__' in got:
                bad.append(('S-M1', f'{r["analysis"]}: {got["__error__"]}'))
                continue
            act_refs = set(got)
            if act_refs != decl_refs:
                bad.append(('S-M1', f'{r["analysis"]} (source {i}): declared references '
                                    f'{sorted(decl_refs)}, outputs give {sorted(act_refs)}'))
            for k in sorted(act_refs & decl_refs):
                act_set = set()
                for v in got[k]:
                    try:
                        act_set.add(float(v))
                    except (TypeError, ValueError):
                        bad.append(('S-M1', f'{r["analysis"]} (source {i}): {k} carries a '
                                            f'penalty that is not a number, {v!r}'))
                if want_set is not None and act_set != want_set:
                    bad.append(('S-M1', f'{r["analysis"]} (source {i}): {k} declares '
                                        f'penalties {sorted(want_set)}, outputs give '
                                        f'{sorted(act_set)}'))
            if shown is None: shown = act_refs
        d1 = ''.join('y' if k in decl_refs else 'n' for k in REFS)
        d2 = ''.join('y' if shown and k in shown else 'n' for k in REFS)
        printed.append((r['analysis'], d1, d2, r['pen']))
    return printed


def check_legends(text, bad):
    bk = book()
    sheets = bk.sheet_names
    legends = dict(re.findall(r'\*\*Table (S\d+[ab]?)\.\*\*\s*(.+?)(?=\n\n|\Z)', text, re.S))
    sheet_ids = {s.split('_')[0]: s for s in sheets}
    for sid, sheet in sheet_ids.items():
        if sid not in legends:
            bad.append(('legends', f'sheet {sheet} has no legend'))
    for lid in legends:
        if lid not in sheet_ids:
            bad.append(('legends', f'legend for Table {lid} names no sheet'))
    for sid, sheet in sorted(sheet_ids.items()):
        if sid not in legends: continue
        d = bk.parse(sheet)
        cols = [str(c) for c in d.columns]
        want = SHEET_COLUMNS.get(sid)
        if want is None:
            bad.append(('legends', f'sheet {sheet} has no declared columns'))
        else:
            missing = [c for c in want if c not in cols]
            extra = [c for c in cols if c not in want]
            if missing:
                bad.append(('legends', f'{sheet} is missing declared column(s) {missing}'))
            if extra:
                bad.append(('legends', f'{sheet} carries undeclared column(s) {extra}'))
            if not missing and not extra and cols != list(want):
                bad.append(('legends', f'{sheet} has the declared columns in another order'))
        empty = [c for c in d.columns if d[c].isna().all()]
        if empty:
            bad.append(('legends', f'{sheet} has column(s) empty for every row: {empty}'))
        if not len(d):
            bad.append(('legends', f'{sheet} holds no rows'))
    return len(sheets), len(legends)


def main(verbose=True):
    bad, declared = [], []
    text = open(SM).read()
    printed = check_scope(text, bad, declared)
    if verbose:
        print(f'{"analysis":62s} {"declared":14s} {"actual":14s}')
        for a, d1, d2, pen in printed:
            print(f'{a[:62]:62s} {d1 + " pen " + pen:14s} {d2:14s}')
    try:
        ns, nl = check_legends(text, bad)
        if verbose: print(f'\nworkbook sheets {ns} | table legends {nl}')
    except Exception as e:
        bad.append(('legends', f'the workbook could not be read: {type(e).__name__}: {e}'))
    if verbose:
        if declared:
            print('\ndeclared, not machine-checkable from results/:')
            for a, why in declared: print(f'  {a}\n      {why}')
        print(f'\n{len(bad)} problem(s)')
        for where, msg in bad: print(f'  [{where}] {msg}')
    return bad


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
