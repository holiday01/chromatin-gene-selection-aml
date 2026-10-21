"""Fault injection for audit_supplementary.py.

An audit that passes on good data has not been shown to reject bad data. These cases were
found by injecting failures into the readers in memory and checking that the audit returns
a non-zero problem count. Nothing on disk is modified; every case replaces
audit_supplementary.rd, .rj or .book for the duration of one call.

Each case names the false negative it exists to prevent, so a later rewrite that
reintroduces one is caught here rather than in review.
"""
import copy, importlib.util, io, json, sys
import pandas as pd

ROOT = '/home/holiday01/2026_cm'
# python -I keeps the script's own directory off sys.path, so load the audit by path
_spec = importlib.util.spec_from_file_location('audit_supplementary',
                                               f'{ROOT}/scripts/audit_supplementary.py')
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
_rd, _rj, _book = A.rd, A.rj, A.book
CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn)); return fn
    return deco


def patch_json(filename, mutate):
    """Serve a mutated copy of one result JSON, the rest untouched."""
    def rj(f):
        j = _rj(f)
        return mutate(copy.deepcopy(j)) if f == filename else j
    return rj


def patch_csv(filename, mutate):
    def rd(f, **k):
        d = _rd(f, **k)
        return mutate(d.copy()) if f == filename else d
    return rd


def patch_sheet(sheet, mutate):
    """Serve the workbook with one sheet mutated, via an in-memory rewrite."""
    def book():
        src = _book()
        frames = {s: src.parse(s) for s in src.sheet_names}
        frames[sheet] = mutate(frames[sheet])
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            for s, d in frames.items():
                d.to_excel(w, sheet_name=s, index=False)
        buf.seek(0)
        return pd.ExcelFile(buf)
    return book


# ---------------------------------------------------------------------- cases

@case('one reference loses a penalty while the others keep both')
def c1():
    def m(j):
        j['cv']['peak density only'].pop('0.1', None)
        return j
    A.rj = patch_json('confound_programs_aml_all_v3.json', m)


@case('one member of a merged row disappears entirely')
def c2():
    def m(j):
        j['cv'].pop('gene span only', None)
        return j
    A.rj = patch_json('confound_programs_aml_all_v3.json', m)


@case('the cross-reference table loses its malignant-cell columns')
def c3():
    A.rd = patch_csv('cross_reference_programs.csv',
                     lambda d: d.drop(columns=['a001', 'a01']))


@case('a sheet loses every performance column')
def c4():
    A.book = patch_sheet('S5_connectivity_audit',
                         lambda d: d.drop(columns=[c for c in d.columns
                                                   if 'AUROC' in c]))


@case('a sheet loses its difference columns')
def c5():
    A.book = patch_sheet('S2_per_program',
                         lambda d: d.drop(columns=[c for c in d.columns
                                                   if 'minus' in c]))


@case('an abandoned column comes back')
def c6():
    def m(d):
        d['beats'] = 1
        return d
    A.book = patch_sheet('S6a_permuted_link_nulls', m)


@case('a link-count draw is missing from one reference')
def c7():
    def m(j):
        j['cv'].pop('linkcount-rand #19', None)
        return j
    A.rj = patch_json('confound_programs_aml_malignant_v3.json', m)


@case('the survival comparison loses a penalty in one reference')
def c8():
    def m(d):
        bad = (d.analysis == 'overall_survival') & (d.reference == 'AML-all') & (d.C == 0.1)
        return d[~bad]
    A.rd = patch_csv('recomputed_remaining.csv', m)


@case('a sheet becomes empty')
def c9():
    A.book = patch_sheet('S12a_inclusion_flow', lambda d: d.iloc[0:0])


@case('a source output goes missing')
def c10():
    def rj(f):
        if f == 'locus_arm.json': raise FileNotFoundError(f)
        return _rj(f)
    A.rj = rj


# positive control: good data must still pass
def control():
    A.rd, A.rj, A.book = _rd, _rj, _book
    return A.main(verbose=False)


if __name__ == '__main__':
    print('positive control, unmodified data')
    base = control()
    print(f'  {len(base)} problem(s){"" if not base else " -- " + str(base)}')
    ok = not base
    if not ok:
        print('  FAIL: the audit does not pass on the real data')

    print('\ninjected failures, each must be rejected')
    for name, setup in CASES:
        A.rd, A.rj, A.book = _rd, _rj, _book
        setup()
        try:
            bad = A.main(verbose=False)
        except Exception as e:
            bad = [('raised', f'{type(e).__name__}: {e}')]
        A.rd, A.rj, A.book = _rd, _rj, _book
        caught = bool(bad)
        ok &= caught
        print(f'  [{"caught " if caught else "MISSED "}] {name}')
        if caught:
            print(f'              {bad[0][1][:110]}')
    print(f'\n{"all cases behave as required" if ok else "SOME CASES FAILED"}')
    sys.exit(0 if ok else 1)
