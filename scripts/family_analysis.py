"""Per-drug subgroup analysis on the ex vivo comparison, with the drug->family map
read from the CORRECT sheet. The previous version read sheet 0 ('synonyms'), which is a
family-name -> abbreviation glossary with no drug column, so every assignment was garbage.
Also asks the mechanistically pointed question: does the chromatin arm help more for drugs
whose own protein targets sit inside the chromatin programs?
"""
import numpy as np, pandas as pd, json
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr
ROOT = '/home/holiday01/2026_cm'
XL = f'{ROOT}/data/raw/beataml2/beataml_drug_families.xlsx'

P = pd.read_csv(f'{ROOT}/results/probe_per_drug.csv', index_col=0)
P = P.drop(columns=[c for c in ('family', 'A_minus_R') if c in P.columns])
fam = pd.read_excel(XL, sheet_name='drug_family')
gen = pd.read_excel(XL, sheet_name='drug_gene')
print(f'drug_family sheet: {len(fam)} rows, {fam.inhibitor.nunique()} inhibitors, '
      f'{fam.family.nunique()} families')

def norm(s): return ''.join(ch for ch in str(s).lower() if ch.isalnum())
# Beat AML result rows are the inhibitor strings verbatim, so an exact normalised key is enough
fmap = {}
for inh, f_ in zip(fam.inhibitor, fam.family):
    fmap.setdefault(norm(inh), []).append(f_)
hit = [norm(d) in fmap for d in P.index]
print(f'exact-name match: {sum(hit)}/{len(P)} drugs')

P['A_minus_R'] = P['A gene subset'] - P['R gene subset']
P['families'] = [';'.join(fmap.get(norm(d), ['unassigned'])) for d in P.index]
PRED = P['all genes'] > 0.2                       # drugs with real patient signal
print(f'drugs with all-gene Spearman > 0.2: {int(PRED.sum())}/{len(P)}')

long = []
for d in P.index:
    for f_ in fmap.get(norm(d), ['unassigned']):
        long.append((d, f_, P.loc[d, 'A_minus_R'], P.loc[d, 'all genes']))
L = pd.DataFrame(long, columns=['drug', 'family', 'A_minus_R', 'allgene'])
Lp = L[L.allgene > 0.2]
print('\nA-minus-R by family, predictable drugs only (n>=4), chromatin-best first:')
fb = Lp.groupby('family').agg(n=('A_minus_R', 'size'), med=('A_minus_R', 'median'),
                              wins=('A_minus_R', lambda s: int((s > 0).sum())))
fb = fb[fb.n >= 4].sort_values('med', ascending=False)
for f_, r in fb.iterrows():
    sub = Lp.loc[Lp.family == f_, 'A_minus_R']
    oth = Lp.loc[Lp.family != f_, 'A_minus_R']
    p = mannwhitneyu(sub, oth).pvalue
    print(f'  {f_[:58]:58s} n={int(r.n):3d} med {r.med:+.4f} wins {int(r.wins)}/{int(r.n)} p={p:.3g}')

# ---- do the drug's own targets live in the chromatin programs? ----
pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
pgenes = pr['genes'].astype(str)
W_A, W_R = pr['W_A'], pr['W_R']
A_genes = set(pgenes[(W_A > 0).any(1)])
R_genes = set(pgenes[(W_R > 0).any(1)])
tier1 = gen[gen.targetome_adj_tier.astype(str).str.startswith('TIER_1')]
tmap = tier1.groupby(tier1.inhibitor.map(norm)).Symbol.apply(lambda s: set(s.dropna()))
nt, inA, inR = [], [], []
for d in P.index:
    t = tmap.get(norm(d), set())
    nt.append(len(t)); inA.append(len(t & A_genes)); inR.append(len(t & R_genes))
P['n_tier1_targets'], P['targets_in_A'], P['targets_in_R'] = nt, inA, inR
has = (P.n_tier1_targets > 0) & PRED
print(f'\npredictable drugs with >=1 Tier-1 target: {int(has.sum())}')
gA = P.loc[has & (P.targets_in_A > 0), 'A_minus_R']
gN = P.loc[has & (P.targets_in_A == 0), 'A_minus_R']
print(f'  target inside chromatin programs  n={len(gA):3d} median A-R {gA.median():+.4f} '
      f'wins {int((gA>0).sum())}/{len(gA)}')
print(f'  target outside                    n={len(gN):3d} median A-R {gN.median():+.4f} '
      f'wins {int((gN>0).sum())}/{len(gN)}')
if len(gA) >= 3 and len(gN) >= 3:
    print(f'  Mann-Whitney p={mannwhitneyu(gA, gN).pvalue:.3g}')
    print(f'  one-sample on the inside group: wilcoxon p='
          f'{wilcoxon(gA).pvalue if len(gA)>5 else float("nan"):.3g}')
r = spearmanr(P.loc[has, 'targets_in_A'], P.loc[has, 'A_minus_R'])
print(f'  continuous: rho(#targets in A, A-R) = {r.statistic:+.3f}  p={r.pvalue:.3g}')
P.to_csv(f'{ROOT}/results/probe_per_drug.csv')
print('\nwrote results/probe_per_drug.csv')
