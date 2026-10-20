"""Save the per-peak and per-gene cell-type specificity matrices that build_programs.py
computes but does not keep, so link permutations can rebuild arm A without reloading the
3.1 GB multiome each time.

Mirrors build_programs.py exactly (same metacells, same pseudobulk, same specificity
definition). Writes S_peak (peaks x cell types) and S_gene (genes x cell types).
"""
import os, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, warnings
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
H5   = '/mnt/10t/scrna_atac/data/raw/multiome_bmmc_neurips/bmmc_multiome.h5ad'
OUT  = f'{ROOT}/data/programs'
MIN_CELLS = 20

print('loading multiome ...', flush=True)
A = ad.read_h5ad(H5)
is_gex = (A.var.feature_types == 'GEX').values
X = A.X.tocsc() if not sp.isspmatrix_csc(A.X) else A.X
G = X[:, np.where(is_gex)[0]].tocsr()
P = X[:, np.where(~is_gex)[0]].tocsr()
ct = A.obs.cell_type.astype(str).values
bt = A.obs.batch.astype(str).values
key = pd.Series([f'{c}||{b}' for c, b in zip(ct, bt)])
counts = key.value_counts(); keep = counts[counts >= MIN_CELLS].index
mask = key.isin(keep).values
groups = pd.Categorical(key[mask], categories=sorted(keep))
M = sp.csr_matrix((np.ones(mask.sum()), (groups.codes, np.where(mask)[0])),
                  shape=(len(keep), G.shape[0]))
meta_ct = np.array([k.split('||')[0] for k in groups.categories])
def pseudobulk(mat):
    s = M @ mat
    s = s.toarray() if sp.issparse(s) else np.asarray(s)
    tot = s.sum(axis=1, keepdims=True); tot[tot == 0] = 1
    return np.log1p(s / tot * 1e4).astype(np.float32)
Gm, Pm = pseudobulk(G), pseudobulk(P)
cts = sorted(set(meta_ct))
def specificity(Zm):
    S = np.zeros((Zm.shape[1], len(cts)), dtype=np.float32)
    for j, c in enumerate(cts):
        m = meta_ct == c
        S[:, j] = Zm[m].mean(0) - Zm[~m].mean(0)
    return S
S_gene, S_peak = specificity(Gm), specificity(Pm)
print(f'  metacells {M.shape[0]} | S_gene {S_gene.shape} | S_peak {S_peak.shape}', flush=True)
np.savez_compressed(f'{OUT}/specificity_bmmc.npz', S_gene=S_gene, S_peak=S_peak,
                    cell_types=np.array(cts),
                    genes=A.var.index[is_gex].astype(str).values,
                    peaks=A.var.index[~is_gex].astype(str).values)
print(f'wrote {OUT}/specificity_bmmc.npz')
