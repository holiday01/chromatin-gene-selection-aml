"""Rebuild the two program matrices from the Lambo pediatric AML reference instead of the
healthy BMMC multiome, so the central claim can be tested against the obvious objection
that a healthy marrow reference cannot carry AML-specific regulatory structure.

The two series are separate libraries. Every ATAC cell carries a Linked_scRNA_barcode, but
that annotation is an imputed assignment, not a co-assay: it is many-to-one, a single RNA
cell can stand in for hundreds of ATAC cells, and some barcodes name cells that are absent
from the processed RNA matrices. Cells are therefore kept only when the linked barcode
resolves to a real RNA cell, so that one inclusion policy covers both modalities and no
metacell carries an all-zero RNA vector that would otherwise be read as zero expression.

Diagnosis samples only, on both sides: the ATAC libraries and the linked RNA libraries must
both be _DX. By default malignant cells only. Peak sets differ between samples, so a union
set is built first and every sample peak is assigned to the union interval that CONTAINS it;
assigning by nearest midpoint instead misplaces about 3% of peaks, because union intervals
differ in width.

Usage: build_programs_aml.py <ATAC dir> <RNA dir> <out npz> [all|malignant]
"""
import sys, os, gzip, glob, json, re
import numpy as np, pandas as pd, scipy.sparse as sp
from multiprocessing import Pool
ATAC, RNA, OUTNPZ = sys.argv[1], sys.argv[2], sys.argv[3]
CELLS = sys.argv[4] if len(sys.argv) > 4 else 'malignant'
ROOT = '/home/holiday01/2026_cm'
WINDOW, MIN_R, TOP_M, MIN_CELLS = 250_000, 0.30, 200, 20

def rd(path, **kw): return pd.read_csv(path, sep='\t', **kw)
def mtx(path):
    """Fast MatrixMarket read: rows, cols, vals, shape. pandas beats scipy.io.mmread here."""
    with gzip.open(path, 'rt') as fh:
        fh.readline()
        hdr = fh.readline().split()
    nr, nc = int(hdr[0]), int(hdr[1])
    d = pd.read_csv(path, sep=r'\s+', skiprows=2, header=None,
                    names=['r', 'c', 'v'], dtype={'r': np.int32, 'c': np.int32, 'v': np.float32})
    return d.r.values - 1, d.c.values - 1, d.v.values, (nr, nc)

# ---------- which samples ----------
meta_files = sorted(glob.glob(f'{ATAC}/*_filtered_metadata.tsv.gz'))
samples = []
for f in meta_files:
    tok = re.sub(r'^GSM\d+_|_filtered_metadata\.tsv\.gz$', '', os.path.basename(f))
    if not tok.endswith('_DX'):
        continue
    gsm = os.path.basename(f).split('_')[0]
    samples.append(dict(tok=tok, gsm=gsm, meta=f,
                        peaks=f.replace('_filtered_metadata', '_peaks'),
                        bc=f.replace('_filtered_metadata', '_filtered_barcodes'),
                        mat=f.replace('_filtered_metadata', '_filtered_peak_matrix').replace('.tsv.gz', '.mtx.gz')))
print(f'Diagnosis ATAC samples: {len(samples)}', flush=True)
for s in samples:
    for k in ('peaks', 'bc', 'mat'):
        assert os.path.exists(s[k]), s[k]

# ---------- union peak set ----------
print('building union peak set ...', flush=True)
allpk = []
for s in samples:
    p = rd(s['peaks'], usecols=['seqnames', 'start', 'end'])
    allpk.append(p)
U = pd.concat(allpk, ignore_index=True).sort_values(['seqnames', 'start'], kind='mergesort')
merged = []
for chrom, g in U.groupby('seqnames', sort=False):
    st, en = g.start.values, g.end.values
    cs, ce = st[0], en[0]
    for a, b in zip(st[1:], en[1:]):
        if a <= ce:
            ce = max(ce, b)
        else:
            merged.append((chrom, cs, ce)); cs, ce = a, b
    merged.append((chrom, cs, ce))
Upk = pd.DataFrame(merged, columns=['chrom', 'start', 'end'])
Upk['mid'] = (Upk.start + Upk.end) // 2
Upk = Upk.sort_values(['chrom', 'mid'], kind='mergesort').reset_index(drop=True)
Upk['ui'] = np.arange(len(Upk))
print(f'  union peaks: {len(Upk):,} (from {len(U):,} sample peaks)', flush=True)
Upk = Upk.sort_values(['chrom', 'start'], kind='mergesort').reset_index(drop=True)
Upk['ui'] = np.arange(len(Upk))
chrom_slice = {c: (g.index[0], g.index[-1] + 1) for c, g in Upk.groupby('chrom', sort=False)}
umid = Upk.mid.values
ustart, uend = Upk.start.values, Upk.end.values

# ---------- metacell assignment, shared by both modalities ----------
rows = []
for s in samples:
    m = rd(s['meta'], usecols=['Cell_Barcode', 'Library_ID', 'Classified_Celltype', 'Malignant',
                               'Linked_scRNA_barcode', 'Linked_scRNA_Sample', 'Patient_ID'])
    m['tok'] = s['tok']
    rows.append(m)
MD = pd.concat(rows, ignore_index=True)
if CELLS == 'malignant':
    MD = MD[MD.Malignant == 'Malignant']
MD = MD[MD.Linked_scRNA_barcode.notna()]
# The ATAC samples are restricted to diagnosis, but Linked_scRNA_Sample can point at a
# relapse or remission library; 16% of malignant diagnosis ATAC cells do. Apply the same
# timepoint restriction to the RNA side so that both modalities describe diagnosis.
_n0 = len(MD)
MD = MD[MD.Linked_scRNA_Sample.astype(str).str.endswith('_DX')]
print(f'  linked RNA restricted to diagnosis: dropped {_n0 - len(MD):,} of {_n0:,} cells '
      f'({100*(_n0-len(MD))/max(_n0,1):.1f}%)', flush=True)
# A linked barcode can name a cell that is absent from the processed RNA matrix. Those
# records were previously kept, so a metacell whose linked barcodes were all missing
# contributed an all-zero RNA vector that entered the correlation and specificity
# calculations as real zero expression. Resolve the barcodes first and keep only cells that
# exist in both modalities, so one inclusion policy covers both.
_n1 = len(MD)
_ok = np.zeros(len(MD), bool)
for _tok, _g in MD.groupby(MD.Linked_scRNA_Sample.astype(str)):
    _c = glob.glob(f'{RNA}/GSM*_{_tok}_processed_matrix.mtx.gz')
    if not _c: continue
    _bc = set(pd.read_csv(_c[0].replace('_processed_matrix.mtx.gz', '_processed_barcodes.tsv.gz'),
                          header=None)[0].values)
    _ok[MD.index.get_indexer(_g.index)] = _g.Linked_scRNA_barcode.isin(_bc).values
MD = MD[_ok]
print(f'  linked barcode absent from the RNA matrix: dropped {_n1 - len(MD):,} of {_n1:,} '
      f'({100*(_n1-len(MD))/max(_n1,1):.1f}%)', flush=True)
_key = MD.Linked_scRNA_Sample.astype(str) + '|' + MD.Linked_scRNA_barcode.astype(str)
_vc = _key.value_counts()
print(f'  cross-modal assignment is many-to-one: {len(MD):,} ATAC cells -> {_vc.size:,} '
      f'distinct RNA cells (max reuse {_vc.max()}, median {int(_vc.median())})', flush=True)
MD['mckey'] = MD.Classified_Celltype.astype(str) + '||' + MD.Library_ID.astype(str)
cnt = MD.mckey.value_counts()
MD = MD[MD.mckey.isin(cnt[cnt >= MIN_CELLS].index)]
mckeys = sorted(MD.mckey.unique())
mc_index = {k: i for i, k in enumerate(mckeys)}
meta_ct = np.array([k.split('||')[0] for k in mckeys])
print(f'cells used: {len(MD):,} ({CELLS}) | metacells: {len(mckeys)} '
      f'(celltype x library, >= {MIN_CELLS} cells) | cell types {len(set(meta_ct))}', flush=True)
print('  cells per cell type:', MD.Classified_Celltype.value_counts().head(10).to_dict(), flush=True)

def atac_one(s):
    """peaks x metacells counts for one sample, already mapped into the union peak set."""
    md = MD[MD.tok == s['tok']]
    if not len(md): return None
    bc = pd.read_csv(s['bc'], header=None)[0].values
    bpos = {b: i for i, b in enumerate(bc)}
    keep = md[md.Cell_Barcode.isin(bpos)]
    if not len(keep): return None
    cidx = np.array([bpos[b] for b in keep.Cell_Barcode])
    gidx = np.array([mc_index[k] for k in keep.mckey])
    r, c, v, (nr, nc) = mtx(s['mat'])
    A = sp.csr_matrix((v, (r, c)), shape=(nr, nc))
    sel = sp.csr_matrix((np.ones(len(cidx)), (cidx, gidx)), shape=(nc, len(mckeys)))
    PB = (A @ sel).tocoo()                              # sample peaks x metacells
    pk = rd(s['peaks'], usecols=['seqnames', 'start', 'end'])
    ui = np.full(len(pk), -1, np.int64)
    unmapped = 0
    for chrom, g in pk.groupby('seqnames', sort=False):
        if chrom not in chrom_slice: continue
        lo, hi = chrom_slice[chrom]
        gs, ge = g.start.values, g.end.values
        # the union set is built by merging overlapping sample peaks, so each sample peak
        # lies inside exactly one union interval. Take the last interval whose start is at
        # or before the peak start, then require that it also covers the peak end. Choosing
        # the nearest union MIDPOINT instead maps 3.2% of peaks into an interval that does
        # not contain them, because union intervals differ in width.
        cand = np.searchsorted(ustart[lo:hi], gs, side='right') - 1
        okc = cand >= 0
        idx = lo + np.clip(cand, 0, hi - lo - 1)
        covered = okc & (ustart[idx] <= gs) & (uend[idx] >= ge)
        ui[g.index.values[covered]] = idx[covered]
        unmapped += int((~covered).sum())
    ok = (PB.row < len(ui)) & (ui[PB.row] >= 0)
    return (sp.coo_matrix((PB.data[ok], (ui[PB.row[ok]], PB.col[ok])),
                          shape=(len(Upk), len(mckeys))).tocsr(), unmapped)

def rna_one(tok):
    """genes x metacells counts, from the scRNA cells linked to this sample's ATAC cells."""
    md = MD[MD.Linked_scRNA_Sample.astype(str) == tok]
    if not len(md): return None
    cand = glob.glob(f'{RNA}/GSM*_{tok}_processed_matrix.mtx.gz')
    if not cand: return None
    base = cand[0].replace('_processed_matrix.mtx.gz', '')
    bc = pd.read_csv(f'{base}_processed_barcodes.tsv.gz', header=None)[0].values
    gn = pd.read_csv(f'{base}_processed_genes.tsv.gz', header=None)[0].astype(str).values
    bpos = {b: i for i, b in enumerate(bc)}
    keep = md[md.Linked_scRNA_barcode.isin(bpos)]
    if not len(keep): return None
    cidx = np.array([bpos[b] for b in keep.Linked_scRNA_barcode])
    gidx = np.array([mc_index[k] for k in keep.mckey])
    r, c, v, (nr, nc) = mtx(f'{base}_processed_matrix.mtx.gz')
    A = sp.csr_matrix((v, (r, c)), shape=(nr, nc))
    sel = sp.csr_matrix((np.ones(len(cidx)), (cidx, gidx)), shape=(nc, len(mckeys)))
    return gn, (A @ sel).tocsr()

if __name__ == '__main__':
    print('reading ATAC matrices ...', flush=True)
    with Pool(8) as p:
        parts = p.map(atac_one, samples)
    Pm_raw, unmapped_total = None, 0
    for q in parts:
        if q is None: continue
        mat, nun = q
        unmapped_total += nun
        Pm_raw = mat if Pm_raw is None else Pm_raw + mat
    # counted per worker and summed here; a module-level counter would not cross the fork
    print(f'  sample peaks with no containing union interval: {unmapped_total:,}', flush=True)
    print(f'  union peak x metacell matrix {Pm_raw.shape}, nnz {Pm_raw.nnz:,}', flush=True)

    print('reading RNA matrices ...', flush=True)
    toks = sorted(set(MD.Linked_scRNA_Sample.astype(str)))
    with Pool(8) as p:
        rparts = p.map(rna_one, toks)
    genes_all = sorted({g for q in rparts if q is not None for g in q[0]})
    gpos = {g: i for i, g in enumerate(genes_all)}
    Gm_raw = sp.csr_matrix((len(genes_all), len(mckeys)), dtype=np.float32)
    for q in rparts:
        if q is None: continue
        gn, Mx = q
        ridx = np.array([gpos[g] for g in gn])
        lift = sp.csr_matrix((np.ones(len(ridx)), (ridx, np.arange(len(gn)))),
                             shape=(len(genes_all), len(gn)))
        Gm_raw = Gm_raw + lift @ Mx
    print(f'  gene x metacell matrix {Gm_raw.shape}, nnz {Gm_raw.nnz:,}', flush=True)

    def norm(Mx):
        a = np.asarray(Mx.todense(), dtype=np.float64).T          # metacells x features
        tot = a.sum(1, keepdims=True); tot[tot == 0] = 1
        return np.log1p(a / tot * 1e4).astype(np.float32)
    Gm, Pm = norm(Gm_raw), norm(Pm_raw)
    print(f'  pseudobulk GEX {Gm.shape}  ATAC {Pm.shape}', flush=True)

    ann = pd.read_csv(f'{ROOT}/data/ref/gencode_v38_genes_tss.tsv', sep='\t', header=None,
                      names=['chrom', 'start', 'end', 'strand', 'tss', 'ensg', 'symbol', 'gene_type'])
    ann = ann.drop_duplicates('symbol')
    gmap = pd.DataFrame({'symbol': genes_all, 'gi': np.arange(len(genes_all))}).merge(
        ann[['symbol', 'chrom', 'tss']], on='symbol', how='inner')
    print(f'  genes with coordinates {len(gmap):,} of {len(genes_all):,}', flush=True)

    def zs(x):
        x = x - x.mean(0, keepdims=True); sd = x.std(0, keepdims=True); sd[sd == 0] = 1
        return x / sd
    Gz, Pz = zs(Gm), zs(Pm); n_meta = Gz.shape[0]
    lr_, lg_, lp_ = [], [], []
    for chrom, gsub in gmap.groupby('chrom'):
        if chrom not in chrom_slice: continue
        lo, hi = chrom_slice[chrom]
        cm = umid[lo:hi]
        for tss, gi in zip(gsub.tss.values, gsub.gi.values):
            a, b = np.searchsorted(cm, [tss - WINDOW, tss + WINDOW])
            if b <= a: continue
            sel = np.arange(lo + a, lo + b)
            r = (Pz[:, sel] * Gz[:, gi:gi + 1]).sum(0) / (n_meta - 1)
            ok = r >= MIN_R
            if ok.any():
                lg_.append(np.full(int(ok.sum()), gi)); lp_.append(sel[ok]); lr_.append(r[ok])
    link_g, link_p, link_r = (np.concatenate(x) for x in (lg_, lp_, lr_))
    print(f'  peak-to-gene links kept: {len(link_r):,} over {len(np.unique(link_g)):,} genes',
          flush=True)

    cts = sorted(set(meta_ct))
    def spec(Zm):
        S = np.zeros((Zm.shape[1], len(cts)), np.float32)
        for j, c in enumerate(cts):
            m = meta_ct == c; S[:, j] = Zm[m].mean(0) - Zm[~m].mean(0)
        return S
    S_gene, S_peak = spec(Gm), spec(Pm)
    S_chrom = np.zeros_like(S_gene); np.add.at(S_chrom, link_g, link_r[:, None] * S_peak[link_p])
    beat = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet', columns=['display_label'])
    shared = set(beat.display_label.astype(str))
    usable = np.array([g in shared for g in genes_all])
    linked = np.zeros(len(genes_all), bool); linked[np.unique(link_g)] = True
    elig = usable & linked
    print(f'  genes in Beat AML {usable.sum():,} | with >=1 link {linked.sum():,} | '
          f'eligible {elig.sum():,}', flush=True)

    def topm(score, el, m):
        W = np.zeros_like(score)
        for j in range(score.shape[1]):
            s = np.where(el, score[:, j], -np.inf)
            idx = np.argpartition(-s, min(m, len(s) - 1))[:m]
            idx = idx[np.isfinite(s[idx]) & (s[idx] > 0)]
            W[idx, j] = s[idx]
        return W
    W_A, W_R = topm(S_chrom, elig, TOP_M), topm(S_gene, elig, TOP_M)
    ov = [len(set(np.flatnonzero(W_A[:, j])) & set(np.flatnonzero(W_R[:, j]))) /
          max(1, len(set(np.flatnonzero(W_A[:, j])))) for j in range(len(cts))]
    print(f'\nprograms: {len(cts)} | arm A nonzero {int((W_A>0).sum()):,} | '
          f'arm R nonzero {int((W_R>0).sum()):,}')
    print(f'gene overlap between arms, per program: median {np.median(ov)*100:.1f}% '
          f'(min {min(ov)*100:.0f}%, max {max(ov)*100:.0f}%)')
    print('cell types:', cts)
    # keep the UNTHRESHOLDED scores as well: W_A/W_R are already top-m masked, so any
    # re-ranking of them (per-link normalisation, partialling out link count) silently
    # returns the same gene set and looks like a null result.
    np.savez_compressed(OUTNPZ, W_A=W_A, W_R=W_R, S_chrom=S_chrom, S_gene=S_gene,
                        genes=np.array(genes_all),
                        cell_types=np.array(cts), eligible=elig,
                        link_g=link_g, link_p=link_p, link_r=link_r,
                        peaks=(Upk.chrom.astype(str) + '-' + Upk.start.astype(str) + '-' +
                               Upk.end.astype(str)).values)
    json.dump(dict(source='Lambo et al. 2023 Cancer Cell, pediatric AML (GSE235308 + GSE235063)', cells=CELLS,
                   stage='Diagnosis', window=WINDOW, min_r=MIN_R, top_m=TOP_M,
                   min_cells=MIN_CELLS, n_samples=len(samples), n_cells=int(len(MD)),
                   n_metacells=len(mckeys), n_union_peaks=int(len(Upk)),
                   n_links=int(len(link_r)), n_programs=len(cts), cell_types=cts,
                   genes_eligible=int(elig.sum()),
                   arm_overlap_median_pct=float(np.median(ov) * 100)),
              open(OUTNPZ.replace('.npz', '_meta.json'), 'w'), indent=2)
    print(f'\nwrote {OUTNPZ}')
