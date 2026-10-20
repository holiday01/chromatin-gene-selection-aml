"""S1: build the two fixed first-layer program matrices from the BMMC multiome.

Arm A (chromatin): gene membership in each cell-type program is scored through the
                   accessibility of the gene's linked peaks (peak-to-gene links within
                   +/-250 kb of the TSS, positive Pearson correlation across metacells).
Arm R (RNA):       the same programs, but membership is scored from the gene's own
                   expression specificity in the same cells.

Both arms produce a genes x 22 matrix with the SAME number of genes per program, so the
network downstream is identical and only the wiring differs.

Outputs to data/programs/.
"""
import os, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, json, warnings
warnings.filterwarnings('ignore')

ROOT = '/home/holiday01/2026_cm'
H5   = '/mnt/10t/scrna_atac/data/raw/multiome_bmmc_neurips/bmmc_multiome.h5ad'
OUT  = f'{ROOT}/data/programs'; os.makedirs(OUT, exist_ok=True)
WINDOW   = 250_000     # +/- around TSS
MIN_R    = 0.30        # keep positive peak-gene links at or above this Pearson r
TOP_M    = 200         # genes per program (identical in both arms)
MIN_CELLS= 20          # minimum cells per metacell

print('loading multiome ...', flush=True)
A = ad.read_h5ad(H5)
is_gex = (A.var.feature_types == 'GEX').values
gex_names  = A.var.index[is_gex].astype(str)
atac_names = A.var.index[~is_gex].astype(str)
X = A.X.tocsc() if not sp.isspmatrix_csc(A.X) else A.X
G = X[:, np.where(is_gex)[0]].tocsr()
P = X[:, np.where(~is_gex)[0]].tocsr()
print(f'  GEX {G.shape}  ATAC {P.shape}', flush=True)

# ---------- metacells: cell type x batch ----------
ct = A.obs.cell_type.astype(str).values
bt = A.obs.batch.astype(str).values
key = pd.Series([f'{c}||{b}' for c, b in zip(ct, bt)])
counts = key.value_counts()
keep_keys = counts[counts >= MIN_CELLS].index
mask = key.isin(keep_keys).values
groups = pd.Categorical(key[mask], categories=sorted(keep_keys))
M = sp.csr_matrix((np.ones(mask.sum()), (groups.codes, np.where(mask)[0])),
                  shape=(len(keep_keys), G.shape[0]))
meta_ct = np.array([k.split('||')[0] for k in groups.categories])
print(f'  metacells: {M.shape[0]} (cell_type x batch, >= {MIN_CELLS} cells), '
      f'cell types {len(set(meta_ct))}', flush=True)

def pseudobulk(mat):
    s = M @ mat                                   # sum within metacell
    s = s.toarray() if sp.issparse(s) else np.asarray(s)
    tot = s.sum(axis=1, keepdims=True); tot[tot == 0] = 1
    return np.log1p(s / tot * 1e4).astype(np.float32)

Gm = pseudobulk(G)   # metacells x genes
Pm = pseudobulk(P)   # metacells x peaks
print(f'  pseudobulk GEX {Gm.shape}  ATAC {Pm.shape}', flush=True)

# ---------- peak and gene coordinates ----------
ann = pd.read_csv(f'{ROOT}/data/ref/gencode_v38_genes_tss.tsv', sep='\t', header=None,
                  names=['chrom','start','end','strand','tss','ensg','symbol','gene_type'])
gv = A.var[is_gex].copy()
gv['ensg'] = gv.gene_id.astype(str).str.replace(r'\..*$', '', regex=True)
gv['gi'] = np.arange(len(gv))
gmap = gv.reset_index().merge(ann[['ensg','chrom','tss']].drop_duplicates('ensg'), on='ensg', how='inner')
pk = pd.DataFrame({'peak': atac_names, 'pi': np.arange(len(atac_names))})
parts = pk.peak.str.split('-', n=2, expand=True)
pk['chrom'] = parts[0]; pk['mid'] = (parts[1].astype(int) + parts[2].astype(int)) // 2
print(f'  genes with coordinates {len(gmap)} | peaks parsed {len(pk)}', flush=True)

# ---------- candidate links within the window, correlated across metacells ----------
def zs(x):
    x = x - x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True); sd[sd == 0] = 1
    return x / sd
Gz, Pz = zs(Gm), zs(Pm)
n_meta = Gz.shape[0]

rows, cols, vals = [], [], []
for chrom, gsub in gmap.groupby('chrom'):
    psub = pk[pk.chrom == chrom]
    if len(psub) == 0: continue
    pmid = psub.mid.values; pidx = psub.pi.values
    order = np.argsort(pmid); pmid = pmid[order]; pidx = pidx[order]
    for tss, gi in zip(gsub.tss.values, gsub.gi.values):
        lo, hi = np.searchsorted(pmid, [tss - WINDOW, tss + WINDOW])
        if hi <= lo: continue
        sel = pidx[lo:hi]
        r = (Pz[:, sel] * Gz[:, gi:gi+1]).sum(0) / (n_meta - 1)
        ok = r >= MIN_R
        if ok.any():
            rows.append(np.full(ok.sum(), gi)); cols.append(sel[ok]); vals.append(r[ok])
link_g = np.concatenate(rows); link_p = np.concatenate(cols); link_r = np.concatenate(vals).astype(np.float32)
print(f'  peak-to-gene links kept: {len(link_r):,} over {len(np.unique(link_g)):,} genes '
      f'(median links/gene {np.median(np.bincount(link_g)[np.unique(link_g)]):.0f})', flush=True)

# ---------- cell-type specificity ----------
cts = sorted(set(meta_ct))
def specificity(Zm):
    """log fold change of a feature in one cell type vs all others, per cell type."""
    S = np.zeros((Zm.shape[1], len(cts)), dtype=np.float32)
    for j, c in enumerate(cts):
        m = meta_ct == c
        S[:, j] = Zm[m].mean(0) - Zm[~m].mean(0)
    return S
S_gene = specificity(Gm)      # genes x cell types   (RNA arm signal)
S_peak = specificity(Pm)      # peaks x cell types   (chromatin arm signal)

# chromatin score: sum over a gene's linked peaks of r * peak specificity
S_chrom = np.zeros_like(S_gene)
np.add.at(S_chrom, link_g, link_r[:, None] * S_peak[link_p])

# ---------- pick the same number of genes per program in both arms ----------
beat = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet', columns=['display_label'])
shared = set(beat.display_label.astype(str))
usable = np.array([g in shared for g in gex_names])
linked = np.zeros(len(gex_names), bool); linked[np.unique(link_g)] = True
print(f'  genes in Beat AML: {usable.sum():,} | with >=1 chromatin link: {linked.sum():,} '
      f'| both: {(usable & linked).sum():,}', flush=True)

def topm(score, eligible, m):
    W = np.zeros_like(score)
    for j in range(score.shape[1]):
        s = np.where(eligible, score[:, j], -np.inf)
        idx = np.argpartition(-s, m)[:m]
        idx = idx[np.isfinite(s[idx]) & (s[idx] > 0)]
        W[idx, j] = s[idx]
    return W

elig = usable & linked          # identical gene pool for both arms, so only wiring differs
W_A = topm(S_chrom, elig, TOP_M)
W_R = topm(S_gene,  elig, TOP_M)
ov = [(len(set(np.flatnonzero(W_A[:, j])) & set(np.flatnonzero(W_R[:, j]))) /
       max(1, len(set(np.flatnonzero(W_A[:, j]))))) for j in range(len(cts))]
print(f'\narm A nonzero {int((W_A>0).sum()):,} | arm R nonzero {int((W_R>0).sum()):,}')
print(f'gene overlap between the two arms, per program: median {np.median(ov)*100:.1f}% '
      f'(min {min(ov)*100:.0f}%, max {max(ov)*100:.0f}%)')

np.savez_compressed(f'{OUT}/programs.npz', W_A=W_A, W_R=W_R,
                    genes=np.array(gex_names), cell_types=np.array(cts),
                    eligible=elig, link_g=link_g, link_p=link_p, link_r=link_r,
                    peaks=np.array(atac_names))
json.dump({'window': WINDOW, 'min_r': MIN_R, 'top_m': TOP_M, 'min_cells': MIN_CELLS,
           'n_metacells': int(M.shape[0]), 'n_links': int(len(link_r)),
           'n_programs': len(cts), 'cell_types': cts,
           'genes_eligible': int(elig.sum()),
           'arm_overlap_median_pct': float(np.median(ov) * 100)},
          open(f'{OUT}/programs_meta.json', 'w'), indent=2)
print(f'\nwrote {OUT}/programs.npz and programs_meta.json')
