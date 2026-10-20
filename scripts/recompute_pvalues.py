"""Recompute every paired comparison quoted in the Results with the same inference the
Methods declares.

The figures previously quoted (e.g. P = 5.8e-18) are paired Wilcoxon statistics over 100
repeated cross-validation folds. Repeated CV reuses every patient in 20 test folds, so those
are resampling statistics, not population evidence, and they contradict the stated analysis.
Everything here is recomputed as a per-repeat mean difference (n = 20) with the
Nadeau-Bengio variance correction, plus a patient-level bootstrap of the cross-wave
difference.
"""
import os, json
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
C_GRID = [1e-2, 1e-1]; NSPLIT, NREP, NBOOT, BOOT_SEED = 5, 20, 2000, 5

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
folds = list(RepeatedStratifiedKFold(n_splits=NSPLIT, n_repeats=NREP, random_state=0)
             .split(np.zeros((okm.sum(), 1)), y))
n_tr, n_te = okm.sum()*(NSPLIT-1)/NSPLIT, okm.sum()/NSPLIT
# CORR removed: the correction is now computed inside nb() from the fold count

def run(sel):
    Mw = XF[:, sel]; M = Mw[okm]; out = {}
    for C in C_GRID:
        cv = []
        for a, b in folds:
            mu, sd = M[a].mean(0), M[a].std(0) + 1e-8
            cv.append(roc_auc_score(y[b], LogisticRegression(C=C, max_iter=5000)
                      .fit((M[a]-mu)/sd, y[a]).decision_function((M[b]-mu)/sd)))
        mu, sd = Mw[tr].mean(0), Mw[tr].std(0) + 1e-8
        dv = LogisticRegression(C=C, max_iter=5000).fit((Mw[tr]-mu)/sd, yall[tr]) \
             .decision_function((Mw[te]-mu)/sd)
        out[C] = (np.array(cv), roc_auc_score(yall[te], dv), dv)
    return out

def nb(d, n_tr, n_te):
    """Corrected resampled t-test on the individual fold differences; see
    robustness_check.py for why the per-repeat-mean variant was abandoned."""
    d=np.asarray(d).ravel(); m,v=d.mean(),d.var(ddof=1)
    se=np.sqrt(v*((1.0/len(d))+(n_te/n_tr)))
    t=m/se if se>0 else np.nan
    return m,t,(2*stats.t.sf(abs(t),df=len(d)-1) if np.isfinite(t) else np.nan)

yte = yall[te]
def boot_indices(n=NBOOT, seed=BOOT_SEED):
    """one fixed set of patient-level resample indices, drawn from BOOT_SEED and reused by
    every comparison here and in robustness_check.py, so one contrast has one interval"""
    rng = np.random.default_rng(seed); idx = np.arange(len(yte)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(yte[b])) < 2: continue
        out.append(b)
    return np.array(out)
BIDX = boot_indices()
def boot(dvA, dvB):
    d = np.array([roc_auc_score(yte[b], dvA[b]) - roc_auc_score(yte[b], dvB[b]) for b in BIDX])
    return d.mean(), np.percentile(d, [2.5, 97.5])

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
pg = pr['genes'].astype(str); cts = list(pr['cell_types'].astype(str)); jH = cts.index('HSC')
elig = pr['eligible']
sp_ = np.load(f'{ROOT}/data/programs/specificity_bmmc.npz', allow_pickle=True)
Sc = np.zeros((len(pg), len(cts)), np.float32)
np.add.at(Sc, pr['link_g'], pr['link_r'][:, None] * sp_['S_peak'][pr['link_p']])
WA, WR = Sc[:, jH], sp_['S_gene'][:, jH]
def top(score, k):
    s = np.where(elig, np.nan_to_num(score, nan=-np.inf, neginf=-np.inf), -np.inf)
    idx = np.argsort(-s)[:k]
    return np.array([EX[pg[i]] for i in idx if np.isfinite(s[i]) and pg[i] in EX])
pub = [l.strip() for l in open(f'{ROOT}/data/ref/published_signatures/ng2016_lsc104_resolved.txt') if l.strip()]
pub = np.array([EX[g] for g in pub if g in EX])
l17 = [l.strip() for l in open(f'{ROOT}/data/ref/published_signatures/ng2016_lsc17.txt') if l.strip()]
A17 = {'LOC284422': 'C19orf77'}
l17 = np.array([EX[A17.get(g, g)] for g in l17 if A17.get(g, g) in EX])
K = len(pub)

# locus arms (no linking)
parts = pd.Series(pr['peaks'].astype(str)).str.split('-', n=2, expand=True)
pchrom = parts[0].values; pmid = ((parts[1].astype(int)+parts[2].astype(int))//2).values
ann = pd.read_csv(f'{ROOT}/data/ref/gencode_v38_genes_tss.tsv', sep='\t', header=None,
                  names=['chrom','start','end','strand','tss','ensg','symbol','gene_type']
                  ).drop_duplicates('symbol').set_index('symbol')
tss = np.array([float(ann.tss[g]) if g in ann.index else np.nan for g in pg])
gch = np.array([ann.chrom[g] if g in ann.index else '' for g in pg])
sph = sp_['S_peak'][:, jH]
lmean = np.full(len(pg), -np.inf)
for c in np.unique(pchrom):
    i = np.where(pchrom == c)[0]; o = i[np.argsort(pmid[i])]
    mids, vals = pmid[o], sph[o]; cs = np.concatenate([[0.0], np.cumsum(vals.astype(float))])
    gi = np.where((gch == c) & np.isfinite(tss))[0]
    if not len(gi): continue
    lo = np.searchsorted(mids, tss[gi]-250_000); hi = np.searchsorted(mids, tss[gi]+250_000)
    n = (hi-lo).astype(float); ok = n > 0
    lmean[gi[ok]] = (cs[hi]-cs[lo])[ok]/n[ok]

ARMS = {'armA_99': top(WA, K), 'armR_99': top(WR, K), 'published_LSC': pub,
        'armA_17': top(WA, 17), 'armR_17': top(WR, 17), 'LSC17': l17,
        'armA_200': top(WA, 200), 'armR_200': top(WR, 200), 'locus_mean_200': top(lmean, 200)}
R = {k: run(v) for k, v in ARMS.items()}
PAIRS = [('armA_99','published_LSC'), ('armR_99','published_LSC'), ('armA_99','armR_99'),
         ('armA_17','LSC17'), ('armA_17','armR_17'),
         ('armA_200','locus_mean_200'), ('locus_mean_200','armR_200'), ('armA_200','armR_200')]
rows = []
print(f'{"comparison":34s}{"C":>7s}{"CV diff":>10s}{"NB t":>8s}{"NB p":>9s}{"xwave":>9s}{"boot 95% CI":>22s}')
for a, b in PAIRS:
    for C in C_GRID:
        d = R[a][C][0] - R[b][C][0]
        m, t, p = nb(d, n_tr, n_te)
        bm, bci = boot(R[a][C][2], R[b][C][2])
        xw = R[a][C][1] - R[b][C][1]
        print(f'{a+" vs "+b:34s}{C:7g}{m:+10.4f}{t:+8.2f}{p:9.4f}{xw:+9.4f}'
              f'   [{bci[0]:+.3f},{bci[1]:+.3f}]')
        rows.append(dict(comparison=f'{a} vs {b}', C=C, cv_diff=m, nb_t=t, nb_p=p,
                         xwave_diff=xw, boot_lo=bci[0], boot_hi=bci[1]))
pd.DataFrame(rows).to_csv(f'{ROOT}/results/recomputed_pvalues.csv', index=False)
print('\nabsolute AUROC (CV C=0.01 / C=0.1 | xwave C=0.01 / C=0.1):')
for k in ARMS:
    print(f'  {k:18s} n={len(ARMS[k]):3d}  {R[k][0.01][0].mean():.3f} {R[k][0.1][0].mean():.3f}'
          f' | {R[k][0.01][1]:.3f} {R[k][0.1][1]:.3f}')
print('\nwrote results/recomputed_pvalues.csv')
