"""Null models for peak-to-gene link identity: does it matter WHICH peaks a gene is linked
to, or only how many and where it sits?

S8 already showed that the numeric link strength is uninformative, but that test kept the
gene set FIXED and only shuffled the weights. Here the gene set itself is rebuilt from
permuted links, which is the test that actually bears on regulatory peak identity.

  null_global  degree-preserving: keep each gene's link count and r values, draw its peaks
               from the global pool of linked peaks. Destroys locality AND identity.
  null_window  position-preserving: replace each gene's linked peaks with peaks drawn from
               that gene's own +/-250 kb window. Keeps locus, count and r; destroys only the
               correlation-based choice WITHIN the locus. This is the strict test -- it asks
               whether the specific correlated peaks beat any neighbouring peak.

For each permutation the HSC arm A gene set is rebuilt and scored exactly as the real one.

Usage: link_permutation.py [n_perm]
"""
import os, sys, json
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
from multiprocessing import Pool
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 20
C_GRID = [1e-2, 1e-1]
WINDOW, TOP_M = 250_000, 200

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
sp_ = np.load(f'{ROOT}/data/programs/specificity_bmmc.npz', allow_pickle=True)
pgenes = pr['genes'].astype(str); cts = list(pr['cell_types'].astype(str))
elig = pr['eligible']; link_g, link_p, link_r = pr['link_g'], pr['link_p'], pr['link_r']
peaks = pr['peaks'].astype(str)
S_peak = sp_['S_peak']
assert list(sp_['peaks'].astype(str)) == list(peaks), 'peak order differs between files'
assert list(sp_['genes'].astype(str)) == list(pgenes), 'gene order differs between files'
assert list(sp_['cell_types'].astype(str)) == cts, 'cell type order differs'
jH = cts.index('HSC')
W_A_hsc = np.array(pr['W_A'][:, jH])   # materialise before fork: NpzFile is lazy and not fork-safe
print(f'genes {len(pgenes)} | peaks {len(peaks)} | links {len(link_r)} | eligible {elig.sum()}')

# gene TSS and per-chromosome peak midpoints, for the window-preserving null
ann = pd.read_csv(f'{ROOT}/data/ref/gencode_v38_genes_tss.tsv', sep='\t', header=None,
                  names=['chrom','start','end','strand','tss','ensg','symbol','gene_type'])
ann = ann.drop_duplicates('symbol').set_index('symbol')
tss = np.array([float(ann.tss[g]) if g in ann.index else np.nan for g in pgenes])
gchrom = np.array([ann.chrom[g] if g in ann.index else '' for g in pgenes])
parts = pd.Series(peaks).str.split('-', n=2, expand=True)
pchrom = parts[0].values
pmid = ((parts[1].astype(int) + parts[2].astype(int)) // 2).values
order = {}
for c in np.unique(pchrom):
    idx = np.where(pchrom == c)[0]
    o = idx[np.argsort(pmid[idx])]
    order[c] = (o, pmid[o])

def build_W(lg, lp, lr):
    S = np.zeros((len(pgenes), len(cts)), dtype=np.float32)
    np.add.at(S, lg, lr[:, None] * S_peak[lp])
    s = np.where(elig, S[:, jH], -np.inf)
    idx = np.argpartition(-s, TOP_M)[:TOP_M]
    return idx[np.isfinite(s[idx]) & (s[idx] > 0)]

def perm_global(rng):
    pool = np.unique(link_p)
    return link_g, rng.choice(pool, size=len(link_p), replace=True), link_r

def perm_window(rng):
    newp = link_p.copy()
    for g in np.unique(link_g):
        if not np.isfinite(tss[g]) or gchrom[g] not in order: continue
        o, mids = order[gchrom[g]]
        lo, hi = np.searchsorted(mids, [tss[g] - WINDOW, tss[g] + WINDOW])
        if hi - lo < 2: continue
        m = link_g == g
        newp[m] = rng.choice(o[lo:hi], size=m.sum(), replace=True)
    return link_g, newp, link_r

expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr.drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
S = spec.set_index('dbgap_rnaseq_sample')
lab, coh = S.induction_response.to_dict(), S.cohort.to_dict()
yall = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
cw = np.array([str(coh.get(s)) for s in cols])
okm = yall >= 0; y = yall[okm]
tr, te = okm & (cw == 'Waves1+2'), okm & (cw == 'Waves3+4')
XF = expr.loc[:, cols].to_numpy(dtype=np.float64).T
EX = {g: i for i, g in enumerate(expr.index)}
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
             .split(np.zeros((okm.sum(), 1)), y))

def score(gidx):
    sel = np.array([EX[pgenes[i]] for i in gidx if pgenes[i] in EX])
    if len(sel) < 5: return None
    Mw = XF[:, sel]; M = Mw[okm]; out = {}
    for C in C_GRID:
        cv = []
        for a, b in folds:
            mu, sd = M[a].mean(0), M[a].std(0) + 1e-8
            cv.append(roc_auc_score(y[b], LogisticRegression(C=C, max_iter=5000)
                      .fit((M[a]-mu)/sd, y[a]).decision_function((M[b]-mu)/sd)))
        mu, sd = Mw[tr].mean(0), Mw[tr].std(0) + 1e-8
        out[C] = (np.array(cv), roc_auc_score(yall[te], LogisticRegression(C=C, max_iter=5000)
                  .fit((Mw[tr]-mu)/sd, yall[tr]).decision_function((Mw[te]-mu)/sd)), len(sel))
    return out

def job(t):
    kind, seed = t
    rng = np.random.default_rng(seed)
    lg, lp, lr = (perm_global if kind == 'global' else perm_window)(rng)
    gidx = build_W(lg, lp, lr)
    ov = len(REAL_SET & set(gidx))
    return kind, seed, score(gidx), ov

REAL_SET = set(np.flatnonzero(W_A_hsc))

if __name__ == '__main__':
    real_idx = np.flatnonzero(W_A_hsc)
    rebuilt = build_W(link_g, link_p, link_r)
    print(f'sanity: rebuilt arm A reproduces {len(set(real_idx) & set(rebuilt))}/{len(real_idx)} '
          f'of the stored gene set')
    R = score(real_idx)
    tasks = [(k, 1000 + i) for k in ('window',) for i in range(NPERM)]
    with Pool(12) as p:
        out = p.map(job, tasks)
    print(f'\nreal arm A: ' + '  '.join(f'C={C:g} CV {R[C][0].mean():.3f} xwave {R[C][1]:.3f}'
                                        for C in C_GRID))
    res = {}
    for kind in ('window',):
        sub = [o for o in out if o[0] == kind and o[2]]
        print(f'\n--- null: {kind} ({len(sub)} permutations) ---')
        print(f'  gene overlap with the real arm A set: median {np.median([o[3] for o in sub]):.0f}/200')
        for C in C_GRID:
            ms = [o[2][C][0].mean() for o in sub]
            xs = [o[2][C][1] for o in sub]
            beats = sum(1 for o in sub if (R[C][0] - o[2][C][0]).mean() > 0
                        and wilcoxon(R[C][0] - o[2][C][0]).pvalue < 0.05)
            above = sum(1 for x in xs if R[C][1] > x)
            pemp = (sum(1 for m in ms if m >= R[C][0].mean()) + 1) / (len(ms) + 1)
            print(f'  C={C:g}  real {R[C][0].mean():.3f} vs null {np.mean(ms):.3f}+/-{np.std(ms):.3f}'
                  f'  (real beats {beats}/{len(sub)}, empirical p={pemp:.3f})'
                  f' | xwave real {R[C][1]:.3f} vs {np.mean(xs):.3f}+/-{np.std(xs):.3f} above {above}/{len(sub)}')
            res[f'{kind}_C{C:g}'] = {'real_cv': float(R[C][0].mean()), 'null_cv_mean': float(np.mean(ms)),
                                     'null_cv_sd': float(np.std(ms)), 'beats': beats, 'n': len(sub),
                                     'emp_p': float(pemp), 'real_xw': float(R[C][1]),
                                     'null_xw_mean': float(np.mean(xs)), 'above': above}
    json.dump(res, open(f'{ROOT}/results/link_permutation.json','w'), indent=2)
    print('\nwrote results/link_permutation.json')
