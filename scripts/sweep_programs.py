"""Does the chromatin gene set keep its clinical-endpoint advantage across the
peak-to-gene hyperparameters? If the +0.02 AUROC exists only at WINDOW=250kb,
MIN_R=0.30, TOP_M=200, it is a fluke.

Correlations are computed once at the widest window with the loosest r threshold and
the link distance is stored, so every (WINDOW, MIN_R) pair is just a filter on that set.
Each config is then evaluated on the same 100 CV folds as the main analysis, against the
size-matched RNA arm and a variance-matched random draw.
"""
import os, json, itertools
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
from multiprocessing import Pool
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
H5 = '/mnt/10t/scrna_atac/data/raw/multiome_bmmc_neurips/bmmc_multiome.h5ad'
MIN_CELLS = 20
W_MAX, R_MIN = 500_000, 0.20
WINDOWS = [100_000, 250_000, 500_000]
MIN_RS  = [0.20, 0.30, 0.40]
TOP_MS  = [100, 200, 400]
C_GRID  = [1e-3, 1e-2, 1e-1, 1.0]

print('loading multiome ...', flush=True)
A = ad.read_h5ad(H5)
is_gex = (A.var.feature_types == 'GEX').values
gex_names = A.var.index[is_gex].astype(str); atac_names = A.var.index[~is_gex].astype(str)
X = A.X.tocsc() if not sp.isspmatrix_csc(A.X) else A.X
G = X[:, np.where(is_gex)[0]].tocsr(); P = X[:, np.where(~is_gex)[0]].tocsr()
ct = A.obs.cell_type.astype(str).values; bt = A.obs.batch.astype(str).values
key = pd.Series([f'{c}||{b}' for c, b in zip(ct, bt)])
cnt = key.value_counts(); keep = cnt[cnt >= MIN_CELLS].index; mask = key.isin(keep).values
groups = pd.Categorical(key[mask], categories=sorted(keep))
M = sp.csr_matrix((np.ones(mask.sum()), (groups.codes, np.where(mask)[0])), shape=(len(keep), G.shape[0]))
meta_ct = np.array([k.split('||')[0] for k in groups.categories])

def pseudobulk(mat):
    s = M @ mat; s = s.toarray() if sp.issparse(s) else np.asarray(s)
    tot = s.sum(1, keepdims=True); tot[tot == 0] = 1
    return np.log1p(s / tot * 1e4).astype(np.float32)
Gm, Pm = pseudobulk(G), pseudobulk(P)
ann = pd.read_csv(f'{ROOT}/data/ref/gencode_v38_genes_tss.tsv', sep='\t', header=None,
                  names=['chrom','start','end','strand','tss','ensg','symbol','gene_type'])
gv = A.var[is_gex].copy()
gv['ensg'] = gv.gene_id.astype(str).str.replace(r'\..*$', '', regex=True); gv['gi'] = np.arange(len(gv))
gmap = gv.reset_index().merge(ann[['ensg','chrom','tss']].drop_duplicates('ensg'), on='ensg', how='inner')
pk = pd.DataFrame({'peak': atac_names, 'pi': np.arange(len(atac_names))})
pp = pk.peak.str.split('-', n=2, expand=True)
pk['chrom'] = pp[0]; pk['mid'] = (pp[1].astype(int) + pp[2].astype(int)) // 2

def zs(x):
    x = x - x.mean(0, keepdims=True); sd = x.std(0, keepdims=True); sd[sd == 0] = 1
    return x / sd
Gz, Pz = zs(Gm), zs(Pm); n_meta = Gz.shape[0]
rows, cols, vals, dist = [], [], [], []
for chrom, gsub in gmap.groupby('chrom'):
    psub = pk[pk.chrom == chrom]
    if len(psub) == 0: continue
    pmid, pidx = psub.mid.values, psub.pi.values
    o = np.argsort(pmid); pmid, pidx = pmid[o], pidx[o]
    for tss, gi in zip(gsub.tss.values, gsub.gi.values):
        lo, hi = np.searchsorted(pmid, [tss - W_MAX, tss + W_MAX])
        if hi <= lo: continue
        sel, dd = pidx[lo:hi], np.abs(pmid[lo:hi] - tss)
        r = (Pz[:, sel] * Gz[:, gi:gi+1]).sum(0) / (n_meta - 1)
        ok = r >= R_MIN
        if ok.any():
            rows.append(np.full(int(ok.sum()), gi)); cols.append(sel[ok])
            vals.append(r[ok]); dist.append(dd[ok])
LG, LP, LR, LD = (np.concatenate(x) for x in (rows, cols, vals, dist))
print(f'  master link set: {len(LR):,} links, {len(np.unique(LG)):,} genes '
      f'(<= {W_MAX//1000} kb, r >= {R_MIN})', flush=True)

cts = sorted(set(meta_ct))
def specificity(Zm):
    S = np.zeros((Zm.shape[1], len(cts)), np.float32)
    for j, c in enumerate(cts):
        m = meta_ct == c; S[:, j] = Zm[m].mean(0) - Zm[~m].mean(0)
    return S
S_gene, S_peak = specificity(Gm), specificity(Pm)
beat = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet', columns=['display_label'])
usable = np.array([g in set(beat.display_label.astype(str)) for g in gex_names])

def topm(score, elig, m):
    W = np.zeros_like(score)
    for j in range(score.shape[1]):
        s = np.where(elig, score[:, j], -np.inf)
        idx = np.argpartition(-s, m)[:m]
        idx = idx[np.isfinite(s[idx]) & (s[idx] > 0)]
        W[idx, j] = s[idx]
    return W

# ---- Beat AML matrix and labels, shared by every config ----
expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr[expr.display_label.isin(set(gex_names))].drop_duplicates('display_label').set_index('display_label')
cols_b = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
Xb = expr[cols_b].to_numpy(dtype=np.float64).T
name2col = {g: i for i, g in enumerate(expr.index)}
lab = dict(zip(spec.dbgap_rnaseq_sample, spec.induction_response))
yb = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols_b])
okb = yb >= 0; yb = yb[okb]; Xb = Xb[okb]
var_b = Xb.var(0)
dec = pd.qcut(pd.Series(var_b), 10, labels=False, duplicates='drop').to_numpy()
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0).split(Xb, yb))
rng = np.random.default_rng(7)

def genes_to_cols(mask_on_gex):
    gs = gex_names[mask_on_gex]
    return np.array(sorted({name2col[g] for g in gs if g in name2col}))

tasks = {}
cfg_info = {}
for w, mr, tm in itertools.product(WINDOWS, MIN_RS, TOP_MS):
    keep_l = (LD <= w) & (LR >= mr)
    lg, lp, lr = LG[keep_l], LP[keep_l], LR[keep_l]
    S_chrom = np.zeros_like(S_gene); np.add.at(S_chrom, lg, lr[:, None] * S_peak[lp])
    linked = np.zeros(len(gex_names), bool); linked[np.unique(lg)] = True
    elig = usable & linked
    WA, WR = topm(S_chrom, elig, tm), topm(S_gene, elig, tm)
    cA, cR = genes_to_cols((WA > 0).any(1)), genes_to_cols((WR > 0).any(1))
    tag = f'w{w//1000}_r{mr}_m{tm}'
    # variance-matched random draw of A's size, drawn outside A
    need = pd.Series(dec[cA]).value_counts().to_dict()
    pick = []
    for d, k in need.items():
        pool = np.setdiff1d(np.where(dec == d)[0], cA)
        pick.append(rng.choice(pool, min(k, len(pool)), replace=False))
    cM = np.concatenate(pick)
    cfg_info[tag] = dict(window=w, min_r=mr, top_m=tm, n_links=int(keep_l.sum()),
                         eligible=int(elig.sum()), nA=len(cA), nR=len(cR))
    for arm, sel in (('A', cA), ('R', cR), ('M', cM)):
        tasks[f'{tag}|{arm}'] = sel

def run(item):
    name, sel = item
    Mx = Xb[:, sel]; out = {C: [] for C in C_GRID}
    for tr, te in folds:
        mu, sd = Mx[tr].mean(0), Mx[tr].std(0) + 1e-8
        Xt, Xv = (Mx[tr] - mu) / sd, (Mx[te] - mu) / sd
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=5000).fit(Xt, yb[tr])
            out[C].append(roc_auc_score(yb[te], clf.decision_function(Xv)))
    return name, {C: np.array(v) for C, v in out.items()}

if __name__ == '__main__':
    print(f'evaluating {len(tasks)} arms over {len(folds)} folds ...', flush=True)
    with Pool(22) as p:
        res = dict(p.map(run, list(tasks.items())))
    print('\n  config            links  elig   nA   nR | '
          + ' | '.join(f'C={C:g}: A    R    rand' for C in (1e-2, 1e-1)))
    rowsout = []
    for tag, info in cfg_info.items():
        line = (f'  {tag:17s}{info["n_links"]:7d}{info["eligible"]:6d}'
                f'{info["nA"]:5d}{info["nR"]:5d} |')
        for C in (1e-2, 1e-1):
            a, r_, m_ = (res[f'{tag}|{x}'][C].mean() for x in 'ARM')
            line += f' {a:.3f} {r_:.3f} {m_:.3f} |'
        pA_R = {C: wilcoxon(res[f'{tag}|A'][C] - res[f'{tag}|R'][C]).pvalue for C in C_GRID}
        pA_M = {C: wilcoxon(res[f'{tag}|A'][C] - res[f'{tag}|M'][C]).pvalue for C in C_GRID}
        print(line, flush=True)
        rowsout.append(dict(tag=tag, **info,
            **{f'A_C{C:g}': float(res[f'{tag}|A'][C].mean()) for C in C_GRID},
            **{f'R_C{C:g}': float(res[f'{tag}|R'][C].mean()) for C in C_GRID},
            **{f'M_C{C:g}': float(res[f'{tag}|M'][C].mean()) for C in C_GRID},
            **{f'p_A_vs_R_C{C:g}': float(pA_R[C]) for C in C_GRID},
            **{f'p_A_vs_M_C{C:g}': float(pA_M[C]) for C in C_GRID}))
    D = pd.DataFrame(rowsout)
    D.to_csv(f'{ROOT}/results/sweep_programs.csv', index=False)
    print('\nhow often does chromatin beat the RNA arm, across the 27 configs:')
    for C in C_GRID:
        d = D[f'A_C{C:g}'] - D[f'R_C{C:g}']
        sig = ((d > 0) & (D[f'p_A_vs_R_C{C:g}'] < 0.05)).sum()
        print(f'  C={C:<6g} A>R in {int((d>0).sum())}/27 configs, '
              f'significant in {int(sig)}/27 | median diff {d.median():+.4f}')
    print('\nhow often does chromatin beat the variance-matched random draw:')
    for C in C_GRID:
        d = D[f'A_C{C:g}'] - D[f'M_C{C:g}']
        sig = ((d > 0) & (D[f'p_A_vs_M_C{C:g}'] < 0.05)).sum()
        print(f'  C={C:<6g} A>rand in {int((d>0).sum())}/27 configs, '
              f'significant in {int(sig)}/27 | median diff {d.median():+.4f}')
    print('\nwrote results/sweep_programs.csv')
