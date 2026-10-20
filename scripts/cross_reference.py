"""Is the chromatin-vs-RNA wiring advantage a property of the PROGRAM, reproducible across
two independent references, or does it land on different programs each time?

HSC was picked because it was best in the BMMC reference, so "HSC also works in the AML
reference" is a selected comparison. The unselected test is whether the per-program
A-minus-R values correlate across the two references over programs that can be matched by
cell identity, and whether the aggregate advantage is positive in both.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, wilcoxon, binomtest
ROOT = '/home/holiday01/2026_cm'

# cell identity correspondence between the two reference nomenclatures, written from the
# cell types alone, before looking at any A-minus-R value
PAIRS = [('HSC', 'HSC'), ('G/M prog', 'GMP'), ('ID2-hi myeloid prog', 'Progenitor'),
         ('Lymph prog', 'CLP'), ('CD14+ Mono', 'Monocytes'), ('CD16+ Mono', 'CD16.Monocytes'),
         ('Proerythroblast', 'Early.Erythrocyte'), ('Normoblast', 'Late.Erythrocyte'),
         ('Naive CD20+ B', 'B.Cell'), ('Transitional B', 'Pre.B.Cell'),
         ('CD4+ T naive', 'CD4.Naive'), ('CD4+ T activated', 'CD4.Memory'),
         ('CD8+ T naive', 'CD8.Naive'), ('CD8+ T', 'CD8.Effector'), ('NK', 'NK'),
         ('pDC', 'pDC'), ('cDC2', 'cDC')]

B = pd.read_csv(f'{ROOT}/results/per_program.csv')          # BMMC, from per_program.py
A = pd.read_csv(f'{ROOT}/results/eval_programs_aml_malignant_per_program.csv')
print(f'BMMC programs {len(B)} | AML programs {len(A)} | matched pairs {len(PAIRS)}')

print('\naggregate: is chromatin wiring better than RNA wiring on average over programs?')
for nm, col, frame in (('BMMC C=0.01', 'diff_C0.01', B), ('BMMC C=0.1', 'diff_C0.1', B),
                       ('AML  C=0.01', 'cv_0.01', A), ('AML  C=0.1', 'cv_0.1', A)):
    v = frame[col].dropna().values
    bt = binomtest(int((v > 0).sum()), len(v), 0.5)
    print(f'  {nm}  mean {v.mean():+.4f}  median {np.median(v):+.4f}  '
          f'positive {int((v>0).sum())}/{len(v)}  sign-test p={bt.pvalue:.3g}  '
          f'wilcoxon p={wilcoxon(v).pvalue:.3g}')

rows = []
for bn, an in PAIRS:
    br = B[B.program == bn]; ar = A[A.program == an]
    if not len(br) or not len(ar):
        print(f'  !! unmatched: {bn} / {an}'); continue
    rows.append(dict(bmmc=bn, aml=an,
                     b001=br['diff_C0.01'].iloc[0], a001=ar['cv_0.01'].iloc[0],
                     b01=br['diff_C0.1'].iloc[0], a01=ar['cv_0.1'].iloc[0]))
M = pd.DataFrame(rows)
print(f'\nmatched programs: {len(M)}')
print(M.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
print('\ndo the two references agree on WHICH programs favour chromatin?')
for c in ('001', '01'):
    rs = spearmanr(M[f'b{c}'], M[f'a{c}']); rp = pearsonr(M[f'b{c}'], M[f'a{c}'])
    agree = int(((M[f'b{c}'] > 0) == (M[f'a{c}'] > 0)).sum())
    bt = binomtest(agree, len(M), 0.5)
    lbl = 'C=0.01' if c == '001' else 'C=0.1'
    print(f'  {lbl}  Spearman rho {rs.statistic:+.3f} p={rs.pvalue:.3g} | '
          f'Pearson r {rp.statistic:+.3f} p={rp.pvalue:.3g} | '
          f'sign agreement {agree}/{len(M)} p={bt.pvalue:.3g}')
print('\nprograms where chromatin wins in BOTH references:')
for c, lbl in (('001', 'C=0.01'), ('01', 'C=0.1')):
    both = M[(M[f'b{c}'] > 0) & (M[f'a{c}'] > 0)]
    print(f'  {lbl}: {", ".join(both.bmmc)} ({len(both)}/{len(M)})')
M.to_csv(f'{ROOT}/results/cross_reference_programs.csv', index=False)
print('\nwrote results/cross_reference_programs.csv')

# ---------------------------------------------------------------------------
# One pre-specified test. The stem/progenitor compartment is where AML arises, so it is
# named from biology, not from the numbers. Everything else is the comparison group.
# Unit of analysis is the matched program, averaged over the four cells (2 references x 2
# penalties), and the stem/progenitor label is permuted over the 17 programs.
STEMPROG = {'HSC', 'G/M prog', 'ID2-hi myeloid prog'}
M = pd.read_csv(f'{ROOT}/results/cross_reference_programs.csv')
M['mean4'] = M[['b001', 'a001', 'b01', 'a01']].mean(1)
M['allpos'] = (M[['b001', 'a001', 'b01', 'a01']] > 0).all(1)
M['stem'] = M.bmmc.isin(STEMPROG)
print('\n\n=== one pre-specified test: stem/progenitor compartment vs the rest ===')
print(M[['bmmc', 'aml', 'mean4', 'allpos', 'stem']]
      .sort_values('mean4', ascending=False)
      .to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
s = M.loc[M.stem, 'mean4'].values; o = M.loc[~M.stem, 'mean4'].values
obs = s.mean() - o.mean()
rng = np.random.default_rng(5)
allv = M.mean4.values; ns = len(s)
perm = np.array([(lambda q: q[:ns].mean() - q[ns:].mean())(rng.permutation(allv))
                 for _ in range(200000)])
p1 = max((perm >= obs).mean(), 1 / 200000)
p2 = max((np.abs(perm) >= abs(obs)).mean(), 1 / 200000)
print(f'\nstem/progenitor (n={len(s)}) mean A-R {s.mean():+.4f} | '
      f'rest (n={len(o)}) {o.mean():+.4f} | gap {obs:+.4f}')
print(f'  permutation over program labels, 200k draws: one-sided p={p1:.4g}, two-sided p={p2:.4g}')
print(f'  all three stem/progenitor programs positive in all 4 cells: '
      f'{int(M.loc[M.stem, "allpos"].sum())}/3; among the other 14: '
      f'{int(M.loc[~M.stem, "allpos"].sum())}/14')
k = int(M.allpos.sum())
print(f'  programs positive in all 4 cells: {k}/17 (expected {17/16:.1f} under a 50% null, '
      f'binomial p={binomtest(k, 17, 1/16, alternative="greater").pvalue:.4g})')
print('\n  ranked by mean over the 4 cells, the top three are: '
      + ', '.join(M.sort_values("mean4", ascending=False).bmmc.head(3)))
