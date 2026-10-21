"""Checks on the exact label-assignment test behind the compartment comparison.

The same seventeen per-programme values are redistributed over the two groups, so two
assignments that are equal in theory can differ in the last bits of their floats. A bare
`>=` then counts one of them and not the other, which moves the P value by one unit of
1/680 without any change to the data. cross_reference.py enumerates every assignment and
counts a gap within TIE_TOL of the observed one as equal to it.

The cases below require that the tail count does not depend on which float path produced
the statistic, that the tolerance is orders of magnitude below the spacing of genuinely
different assignments, and that the result agrees with SciPy's own exact permutation test.
"""
import itertools, sys
import numpy as np, pandas as pd
from scipy.stats import permutation_test

ROOT = '/home/holiday01/2026_cm'
TIE_TOL = 1e-12
DEFS = (('myeloid stem/progenitor', ['HSC', 'G/M prog', 'ID2-hi myeloid prog']),
        ('progenitor, lymphoid included',
         ['HSC', 'G/M prog', 'ID2-hi myeloid prog', 'Lymph prog']))

M = pd.read_csv(f'{ROOT}/results/cross_reference_programs.csv')
CP = pd.read_csv(f'{ROOT}/results/cross_reference_compartment.csv').set_index('definition')


def gaps_for(values, k):
    """The group-mean difference of every assignment of k labels to the programmes."""
    n = len(values); tot = values.sum()
    return np.array([(lambda q: q / k - (tot - q) / (n - k))(values[list(c)].sum())
                     for c in itertools.combinations(range(n), k)])


def tail(values, stem, tol):
    k = int(stem.sum())
    obs = values[stem].mean() - values[~stem].mean()
    g = gaps_for(values, k)
    return int((g >= obs - tol).sum()), len(g), g, obs


if __name__ == '__main__':
    ok = True
    csv_v = M.mean4.values
    src_v = M[['b001', 'a001', 'b01', 'a01']].mean(axis=1).values   # how the producer builds it
    print(f'the two float paths to the same statistic differ by at most '
          f'{np.abs(csv_v - src_v).max():.3g}')

    for lbl, members in DEFS:
        stem = M.bmmc.isin(members).values
        k = int(stem.sum())
        print(f'\n{lbl} ({k} programmes)')

        # 1. the tail count must not depend on which float path produced the statistic
        n_csv, tot_, g, obs = tail(csv_v, stem, TIE_TOL)
        n_src, _, _, _ = tail(src_v, stem, TIE_TOL)
        same = n_csv == n_src
        ok &= same
        print(f'  [{"ok     " if same else "FAIL   "}] tail count with the tolerance: '
              f'{n_csv} from the stored column, {n_src} from the recomputed one')

        # 2. the differential case: a bare comparison does depend on it
        b_csv = int((gaps_for(csv_v, k) >= csv_v[stem].mean() - csv_v[~stem].mean()).sum())
        b_src = int((gaps_for(src_v, k) >= src_v[stem].mean() - src_v[~stem].mean()).sum())
        print(f'  [differential] without the tolerance: {b_csv} and {b_src}'
              + ('  (this definition happens to tie either way)' if b_csv == b_src else ''))

        # 3. the tolerance must sit far below the spacing of different assignments
        d = np.abs(g - obs)
        ties, near = int((d <= TIE_TOL).sum()), float(d[d > TIE_TOL].min())
        head = ties == 1 and near / TIE_TOL > 1e6
        ok &= head
        print(f'  [{"ok     " if head else "FAIL   "}] {ties} assignment tied at {TIE_TOL:g}, '
              f'nearest different gap {near:.3g}, which is {near/TIE_TOL:.0e} times the '
              f'tolerance')

        # 4. agreement with SciPy's own exact permutation test
        r = permutation_test((csv_v[stem], csv_v[~stem]),
                             lambda x, y, axis=0: np.mean(x, axis=axis) - np.mean(y, axis=axis),
                             permutation_type='independent', n_resamples=np.inf,
                             alternative='greater', vectorized=True)
        mine = n_csv / tot_
        agree = abs(r.pvalue - mine) < 1e-9
        ok &= agree
        print(f'  [{"ok     " if agree else "FAIL   "}] exact P {mine:.7f}, '
              f'SciPy {r.pvalue:.7f}')

        # 5. the stored output must be that count over that many assignments
        row = CP.loc[lbl]
        stored = (int(row.n_assignments) == tot_ and int(row.tail_count) == n_csv
                  and abs(float(row.p_one_sided) - mine) < 1e-12)
        ok &= stored
        print(f'  [{"ok     " if stored else "FAIL   "}] stored output: '
              f'{int(row.tail_count)}/{int(row.n_assignments)} = {float(row.p_one_sided):.7f}')

    print(f'\n{"all cases behave as required" if ok else "SOME CASES FAILED"}')
    sys.exit(0 if ok else 1)
