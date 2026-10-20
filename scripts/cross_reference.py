"""Does the selection-rule difference hold outside the haematopoietic stem cell programme,
and does it land on the same programmes in two reference constructions?

HSC entered the manuscript because it ranked near the top in the healthy reference, so
"HSC also works in the AML reference" is a selected comparison. This script evaluates every
programme of both references and asks two unselected questions: how often the chromatin
rule leads over the whole set of programmes, and whether the per-programme differences
agree between references over programmes matched by cell identity.

Four properties of the design bound what the output can support.
  1. Programmes are not independent units. They share the same Beat AML patients, the same
     induction outcomes and the same cross-validation folds, and their gene sets overlap.
     The sign test and Wilcoxon reported here treat programmes as exchangeable units and do
     not account for that dependence, so they are descriptive, not confirmatory.
  2. Every cell here is cross-validated. The two references x two penalties give four CV
     estimates; the held-out wave split is not part of this analysis.
  3. Switching reference does not remove the selection of HSC, because both references are
     scored on the same cohort and the same endpoint.
  4. The stem/progenitor contrast is a grouping of three named programmes, reported with
     its membership and with the sensitivity of the grouping definition, because no
     permutation over programme labels can establish exchangeability between lineages.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, wilcoxon, binomtest
ROOT = '/home/holiday01/2026_cm'
N_PERM = 200000

# cell identity correspondence between the two reference nomenclatures, written from the
# cell type names alone, before looking at any A-minus-R value
PAIRS = [('HSC', 'HSC'), ('G/M prog', 'GMP'), ('ID2-hi myeloid prog', 'Progenitor'),
         ('Lymph prog', 'CLP'), ('CD14+ Mono', 'Monocytes'), ('CD16+ Mono', 'CD16.Monocytes'),
         ('Proerythroblast', 'Early.Erythrocyte'), ('Normoblast', 'Late.Erythrocyte'),
         ('Naive CD20+ B', 'B.Cell'), ('Transitional B', 'Pre.B.Cell'),
         ('CD4+ T naive', 'CD4.Naive'), ('CD4+ T activated', 'CD4.Memory'),
         ('CD8+ T naive', 'CD8.Naive'), ('CD8+ T', 'CD8.Effector'), ('NK', 'NK'),
         ('pDC', 'pDC'), ('cDC2', 'cDC')]

B = pd.read_csv(f'{ROOT}/results/per_program.csv')          # BMMC, from per_program.py
A = pd.read_csv(f'{ROOT}/results/eval_programs_aml_malignant_v3_per_program.csv')
print(f'BMMC programmes {len(B)} | AML programmes {len(A)} | matched pairs {len(PAIRS)}')

print('\naggregate over programmes, cross-validation only')
agg = []
for nm, col, frame in (('BMMC C=0.01', 'diff_C0.01', B), ('BMMC C=0.1', 'diff_C0.1', B),
                       ('AML-mal C=0.01', 'cv_0.01', A), ('AML-mal C=0.1', 'cv_0.1', A)):
    v = frame[col].dropna().values
    sgn = binomtest(int((v > 0).sum()), len(v), 0.5).pvalue
    wil = wilcoxon(v).pvalue
    agg.append(dict(cell=nm, n=len(v), positive=int((v > 0).sum()), mean=v.mean(),
                    median=float(np.median(v)), sign_p=sgn, wilcoxon_p=wil))
    print(f'  {nm:16s} positive {int((v>0).sum()):2d}/{len(v):2d}  mean {v.mean():+.4f}  '
          f'median {np.median(v):+.4f}  sign test p={sgn:.3g}  Wilcoxon p={wil:.3g}')
pd.DataFrame(agg).to_csv(f'{ROOT}/results/cross_reference_aggregate.csv', index=False)

rows, unmatched = [], []
for bn, an in PAIRS:
    br = B[B.program == bn]; ar = A[A.program == an]
    if not len(br) or not len(ar):
        unmatched.append(f'{bn} / {an}'); continue
    rows.append(dict(bmmc=bn, aml=an,
                     b001=br['diff_C0.01'].iloc[0], a001=ar['cv_0.01'].iloc[0],
                     b01=br['diff_C0.1'].iloc[0], a01=ar['cv_0.1'].iloc[0]))
M = pd.DataFrame(rows)
if unmatched: print('  unmatched pairs:', '; '.join(unmatched))
print(f'\nmatched programmes: {len(M)} of {len(B)} healthy and {len(A)} AML')
print('  healthy programmes with no AML counterpart: '
      + ', '.join(sorted(set(B.program) - {b for b, _ in PAIRS})))
print('  AML programmes with no healthy counterpart: '
      + ', '.join(sorted(set(A.program) - {a for _, a in PAIRS})))
print(M.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))

print('\ndo the two references agree on which programmes favour the chromatin rule?')
for c in ('001', '01'):
    rs = spearmanr(M[f'b{c}'], M[f'a{c}']); rp = pearsonr(M[f'b{c}'], M[f'a{c}'])
    agree = int(((M[f'b{c}'] > 0) == (M[f'a{c}'] > 0)).sum())
    lbl = 'C=0.01' if c == '001' else 'C=0.1'
    print(f'  {lbl}  Spearman rho {rs.statistic:+.3f} p={rs.pvalue:.3g} | '
          f'Pearson r {rp.statistic:+.3f} p={rp.pvalue:.3g} | '
          f'sign agreement {agree}/{len(M)}')
print('\nprogrammes where the chromatin rule leads in BOTH references:')
for c, lbl in (('001', 'C=0.01'), ('01', 'C=0.1')):
    both = M[(M[f'b{c}'] > 0) & (M[f'a{c}'] > 0)]
    print(f'  {lbl}: {", ".join(both.bmmc)} ({len(both)}/{len(M)})')

# ---------------------------------------------------------------------------
# The stem/progenitor contrast. The compartment is named from the biology of the disease,
# not from the values, and its membership is stated here in full. The permutation moves the
# label over the 17 matched programmes; it quantifies how unusual the gap is under
# exchangeable programme labels and does not establish that lineages are exchangeable.
STEMPROG = ['HSC', 'G/M prog', 'ID2-hi myeloid prog']
BROADER = STEMPROG + ['Lymph prog']        # a lymphoid-inclusive progenitor definition
M['mean4'] = M[['b001', 'a001', 'b01', 'a01']].mean(1)
M['allpos'] = (M[['b001', 'a001', 'b01', 'a01']] > 0).all(1)
M['stem'] = M.bmmc.isin(STEMPROG)
M.to_csv(f'{ROOT}/results/cross_reference_programs.csv', index=False)
print('\nwrote results/cross_reference_programs.csv and results/cross_reference_aggregate.csv')

print('\n=== stem/progenitor compartment against the remaining programmes ===')
print(M[['bmmc', 'aml', 'mean4', 'allpos', 'stem']]
      .sort_values('mean4', ascending=False)
      .to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
rng = np.random.default_rng(5)
allv = M.mean4.values
for lbl, members in (('myeloid stem/progenitor', STEMPROG),
                     ('progenitor, lymphoid included', BROADER)):
    stem = M.bmmc.isin(members)
    s, o = M.loc[stem, 'mean4'].values, M.loc[~stem, 'mean4'].values
    obs = s.mean() - o.mean(); ns = len(s)
    perm = np.array([(lambda q: q[:ns].mean() - q[ns:].mean())(rng.permutation(allv))
                     for _ in range(N_PERM)])
    p1 = max((perm >= obs).mean(), 1 / N_PERM)
    p2 = max((np.abs(perm) >= abs(obs)).mean(), 1 / N_PERM)
    print(f'\n  {lbl} ({", ".join(members)})')
    print(f'    n={len(s)} mean A-R {s.mean():+.4f} | rest (n={len(o)}) {o.mean():+.4f} | '
          f'gap {obs:+.4f}')
    print(f'    permutation over programme labels, {N_PERM//1000}k draws: '
          f'one-sided p={p1:.4g}, two-sided p={p2:.4g}')
    print(f'    positive in all four CV cells: {int(M.loc[stem, "allpos"].sum())}/{len(s)}; '
          f'among the rest {int(M.loc[~stem, "allpos"].sum())}/{len(o)}')
print(f'\nprogrammes positive in all four CV cells: {int(M.allpos.sum())}/{len(M)} '
      f'({", ".join(M.loc[M.allpos, "bmmc"])})')
print('  reported as a count. The four cells share one cohort and two correlated penalties, '
      'so a binomial null that multiplies four independent coin flips does not apply.')
print('\nranked by the mean over the four CV cells, the top three are: '
      + ', '.join(M.sort_values("mean4", ascending=False).bmmc.head(3)))
