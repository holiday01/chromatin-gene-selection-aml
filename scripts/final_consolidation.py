"""Pool the three reference constructions and state exactly which claims survive.

  BMMC      - healthy adult bone marrow multiome, all cells          (22 programs)
  AML-mal   - pediatric AML at diagnosis, malignant cells only       (19 programs)
  AML-all   - pediatric AML at diagnosis, all cells                  (20 programs)

Every claim is scored over the cells it can be scored in: 3 references x 2 penalties
(x 2 splits where available). Nothing here refits a model; it reads the saved results.
"""
import json, numpy as np, pandas as pd
from scipy.stats import binomtest, spearmanr
ROOT = '/home/holiday01/2026_cm'
REFS = {'BMMC': 'results/eval_programs_bmmc.json',
        'AML-mal': 'results/eval_programs_aml_malignant.json',
        'AML-all': 'results/eval_programs_aml_all.json'}
PROG = {'BMMC': 'results/eval_programs_bmmc_per_program.csv',
        'AML-mal': 'results/eval_programs_aml_malignant_per_program.csv',
        'AML-all': 'results/eval_programs_aml_all_per_program.csv'}
# cell identity correspondence, written from cell type names only
MAP = {'HSC': 'HSC', 'G/M prog': 'GMP', 'ID2-hi myeloid prog': 'Progenitor',
       'Lymph prog': 'CLP', 'CD14+ Mono': 'Monocytes', 'CD16+ Mono': 'CD16.Monocytes',
       'Proerythroblast': 'Early.Erythrocyte', 'Normoblast': 'Late.Erythrocyte',
       'Naive CD20+ B': 'B.Cell', 'Transitional B': 'Pre.B.Cell',
       'CD4+ T naive': 'CD4.Naive', 'CD4+ T activated': 'CD4.Memory',
       'CD8+ T naive': 'CD8.Naive', 'CD8+ T': 'CD8.Effector', 'NK': 'NK',
       'pDC': 'pDC', 'cDC2': 'cDC', 'Plasma cell': 'Plasma'}
STEM = ['HSC', 'G/M prog', 'ID2-hi myeloid prog']
MYE = {'HSC','G/M prog','ID2-hi myeloid prog','Lymph prog','MK/E prog','Proerythroblast',
       'Erythroblast','Normoblast','CD14+ Mono','CD16+ Mono'}
LYM = {'CD4+ T naive','CD4+ T activated','CD8+ T','CD8+ T naive','NK','ILC','B1 B',
       'Transitional B','Naive CD20+ B','Plasma cell'}

P = {r: pd.read_csv(f'{ROOT}/{PROG[r]}').set_index('program') for r in REFS}

print('=== claim 1: within the HSC program, chromatin wiring beats RNA wiring ===')
cells = []
for r in REFS:
    nm = MAP['HSC'] if r != 'BMMC' else 'HSC'
    for C in ('0.01', '0.1'):
        for kind in ('cv', 'xw'):
            v = P[r].loc[nm, f'{kind}_{C}']
            if pd.notna(v): cells.append((r, C, kind, v))
vals = np.array([c[3] for c in cells])
for r, C, kind, v in cells:
    print(f'  {r:9s} C={C:5s} {kind:3s}  {v:+.4f}')
bt = binomtest(int((vals > 0).sum()), len(vals), 0.5)
print(f'  -> positive in {int((vals>0).sum())}/{len(vals)} cells, '
      f'range {vals.min():+.4f} to {vals.max():+.4f}, mean {vals.mean():+.4f}, '
      f'sign-test p={bt.pvalue:.4g}')

print('\n=== claim 2: the stem/progenitor compartment, pooled over three references ===')
rows = []
for b, a in MAP.items():
    rec = {'program': b}
    for r in REFS:
        nm = b if r == 'BMMC' else a
        if nm not in P[r].index: continue
        for C in ('0.01', '0.1'):
            rec[f'{r}_{C}'] = P[r].loc[nm, f'cv_{C}']
    rows.append(rec)
M = pd.DataFrame(rows).set_index('program')
cols = [c for c in M.columns]
M['mean_all'] = M[cols].mean(1)
M['n_pos'] = (M[cols] > 0).sum(1)
M['n_cells'] = M[cols].notna().sum(1)
M['stem'] = M.index.isin(STEM)
print(M[cols + ['mean_all', 'n_pos', 'n_cells', 'stem']]
      .sort_values('mean_all', ascending=False)
      .to_string(float_format=lambda v: f'{v:+.4f}'))
s, o = M.loc[M.stem, 'mean_all'].values, M.loc[~M.stem, 'mean_all'].values
obs = s.mean() - o.mean()
rng = np.random.default_rng(5); allv = M.mean_all.values; ns = len(s)
perm = np.array([(lambda q: q[:ns].mean() - q[ns:].mean())(rng.permutation(allv))
                 for _ in range(200000)])
print(f'\n  stem/progenitor (n={len(s)}) {s.mean():+.4f} vs rest (n={len(o)}) {o.mean():+.4f} | '
      f'gap {obs:+.4f} | one-sided permutation p={max((perm>=obs).mean(),1/200000):.4g}')
print(f'  all six cells positive: stem {int((M.loc[M.stem,"n_pos"]==M.loc[M.stem,"n_cells"]).sum())}/{len(s)}'
      f' | rest {int((M.loc[~M.stem,"n_pos"]==M.loc[~M.stem,"n_cells"]).sum())}/{len(o)}')

print('\n=== claim 3: the myeloid-vs-lymphoid lineage split ===')
for r in REFS:
    d = P[r]
    mye = [i for i in d.index if (i in MYE if r == 'BMMC' else
           any(MAP.get(k) == i for k in MYE))]
    lym = [i for i in d.index if (i in LYM if r == 'BMMC' else
           any(MAP.get(k) == i for k in LYM))]
    for C in ('0.01', '0.1'):
        m, l = d.loc[mye, f'cv_{C}'].dropna(), d.loc[lym, f'cv_{C}'].dropna()
        allv2 = np.concatenate([m.values, l.values]); nm2 = len(m)
        pm = np.array([(lambda q: q[:nm2].mean() - q[nm2:].mean())(rng.permutation(allv2))
                       for _ in range(20000)])
        ob = m.mean() - l.mean()
        print(f'  {r:9s} C={C:5s} myeloid {m.mean():+.4f} (n={len(m)}) vs '
              f'lymphoid {l.mean():+.4f} (n={len(l)}) gap {ob:+.4f} '
              f'perm p={max((np.abs(pm)>=abs(ob)).mean(),1/20000):.4g}')

print('\n=== claim 4 (DROPPED): the chromatin gene subset beats the alternatives ===')
E = {r: json.load(open(f'{ROOT}/{f}')) for r, f in REFS.items() if r != 'BMMC'}
E['BMMC'] = json.load(open(f'{ROOT}/results/eval_programs_bmmc.json'))
for key, lbl in (('A subset', 'A subset'),):
    for r in REFS:
        cv = E[r]['cv']
        for C in ('0.01', '0.1'):
            a = cv[key][C]
            rand = [v[C] for k, v in cv.items() if k.startswith('sub-rand')]
            tv = [v[C] for k, v in cv.items() if 'by variance' in k][0]
            print(f'  {r:9s} C={C:5s} A {a:.3f} | var-matched random {np.mean(rand):.3f} '
                  f'(A above {sum(1 for x in rand if a>x)}/{len(rand)}) | '
                  f'top-by-variance {tv:.3f} (diff {a-tv:+.4f})')
M.to_csv(f'{ROOT}/results/final_consolidation.csv')
print('\nwrote results/final_consolidation.csv')
