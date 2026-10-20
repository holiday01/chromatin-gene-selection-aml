"""Gene overlap between each permuted-link arm A set and the real one.

link_permutation.py and link_permutation_window100.py compute this overlap but print it
and do not store it, so the figure quoted in the text could not be checked against a
file. The overlap depends only on the permuted links and the selection rule, not on the
classifier, so it is recomputed here from the same seeds without refitting any model.
"""
import numpy as np, pandas as pd
ROOT = '/home/holiday01/2026_cm'
WINDOW, TOP_M = 250_000, 200

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
sp_ = np.load(f'{ROOT}/data/programs/specificity_bmmc.npz', allow_pickle=True)
pgenes = pr['genes'].astype(str); cts = list(pr['cell_types'].astype(str))
elig = pr['eligible']; link_g, link_p, link_r = pr['link_g'], pr['link_p'], pr['link_r']
peaks = pr['peaks'].astype(str); S_peak = sp_['S_peak']
assert list(sp_['peaks'].astype(str)) == list(peaks)
assert list(sp_['genes'].astype(str)) == list(pgenes)
jH = cts.index('HSC')
W_A_hsc = np.array(pr['W_A'][:, jH])

ann = pd.read_csv(f'{ROOT}/data/ref/gencode_v38_genes_tss.tsv', sep='\t', header=None,
                  names=['chrom', 'start', 'end', 'strand', 'tss', 'ensg', 'symbol', 'gene_type'])
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

REAL = set(np.flatnonzero(W_A_hsc))
rows = []
for kind, nperm, fn in (('global', 20, perm_global), ('window', 100, perm_window)):
    ov, dis = [], []
    for i in range(nperm):
        lg, lp, lr = fn(np.random.default_rng(1000 + i))
        g = build_W(lg, lp, lr)
        ov.append(len(REAL & set(g)))
        dis.append(len(np.unique(lp)))
    rows.append(dict(null=kind, n_permutations=nperm, real_set_size=len(REAL),
                     overlap_median=float(np.median(ov)), overlap_min=int(min(ov)),
                     overlap_max=int(max(ov)),
                     distinct_peaks_real=int(len(np.unique(link_p))),
                     distinct_peaks_median=float(np.median(dis))))
    print(f'{kind:7s} {nperm:4d} perms | overlap with the real 200-gene set '
          f'median {np.median(ov):.0f}, range {min(ov)} to {max(ov)} | '
          f'distinct peaks used, real {len(np.unique(link_p))} vs permuted median '
          f'{np.median(dis):.0f}')
T = pd.DataFrame(rows)
T.to_csv(f'{ROOT}/results/permutation_overlap.csv', index=False)
print('\nwrote results/permutation_overlap.csv')
