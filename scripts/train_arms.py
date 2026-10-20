"""S2+S3: project the fixed program layer onto Beat AML bulk and train the one model
described in literature/04_dl_model_design.md, once per arm.

Arms
  0  no patient information at all (drug embedding + log dose only)  -> internal floor
  R  22 cell-type programs wired from RNA co-expression
  A  22 cell-type programs wired from chromatin peak-to-gene links
Split: Waves1+2 train (201 specimens) -> Waves3+4 test (133 specimens).
"""
import os, json, numpy as np, pandas as pd, torch, torch.nn as nn, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
torch.manual_seed(0); np.random.seed(0)

ROOT='/home/holiday01/2026_cm'; DEV='cuda' if torch.cuda.is_available() else 'cpu'
pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
W_A, W_R, pgenes, cts = pr['W_A'], pr['W_R'], pr['genes'].astype(str), pr['cell_types'].astype(str)

expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
via  = pd.read_parquet(f'{ROOT}/data/interim/viability.parquet')

# align genes
expr = expr[expr.display_label.isin(set(pgenes))].drop_duplicates('display_label').set_index('display_label')
samp_cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
gi = {g: i for i, g in enumerate(pgenes)}
rows = np.array([gi[g] for g in expr.index])
Xg = expr[samp_cols].to_numpy(dtype=np.float32).T                      # samples x genes(subset)
WA = W_A[rows]; WR = W_R[rows]
print(f'bulk matrix {Xg.shape} | programs {WA.shape[1]} ({len(cts)} cell types)')

def activities(W):
    Z = Xg @ W                                                          # samples x 22
    Z = Z / np.clip(np.abs(Z).sum(1, keepdims=True), 1e-8, None)        # per-sample composition
    return Z.astype(np.float32)
ACT = {'R': activities(WR), 'A': activities(WA)}

d = via.merge(spec[['dbgap_rnaseq_sample','cohort']], on='dbgap_rnaseq_sample')
d = d[d.cohort.isin(['Waves1+2','Waves3+4'])].copy()
d = d[d.dbgap_rnaseq_sample.isin(samp_cols)]
sidx = {s: i for i, s in enumerate(samp_cols)}
didx = {g: i for i, g in enumerate(sorted(d.inhibitor.unique()))}
d['si'] = d.dbgap_rnaseq_sample.map(sidx); d['di'] = d.inhibitor.map(didx)
d['logdose'] = np.log10(d.well_concentration.astype(float))
tr = d[d.cohort=='Waves1+2']; te = d[d.cohort=='Waves3+4']
print(f'train {len(tr):,} points / {tr.si.nunique()} specimens | test {len(te):,} / {te.si.nunique()} | drugs {len(didx)}')

# lookup table from the training split only; the model predicts the residual from it
lut = tr.groupby(['inhibitor','dose_rank']).viability01.mean().rename('lut').reset_index()
gmean = tr.viability01.mean()
for df in (tr, te):
    df['lut'] = df.merge(lut, on=['inhibitor','dose_rank'], how='left').lut.fillna(gmean).values
    df['resid'] = df.viability01 - df.lut
ld_mu, ld_sd = tr.logdose.mean(), tr.logdose.std()
def tens(df, Z):
    out = {'y': torch.tensor(df.resid.values, dtype=torch.float32),
           'lut': torch.tensor(df.lut.values, dtype=torch.float32),
           'di': torch.tensor(df.di.values, dtype=torch.long),
           'ld': torch.tensor(((df.logdose.values - ld_mu) / ld_sd), dtype=torch.float32)}
    if Z is not None: out['z'] = torch.tensor(Z[df.si.values], dtype=torch.float32)
    return {k: v.to(DEV) for k, v in out.items()}

class Net(nn.Module):
    def __init__(self, n_prog, n_drug, emb=16):
        super().__init__()
        self.use_z = n_prog > 0
        self.emb = nn.Embedding(n_drug, emb)
        if self.use_z:
            self.enc = nn.Sequential(nn.Linear(n_prog, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        h = (32 if self.use_z else 0) + emb + 1
        self.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, b):
        parts = [self.emb(b['di']), b['ld'][:, None]]
        if self.use_z: parts.insert(0, self.enc(b['z']))
        return self.head(torch.cat(parts, 1)).squeeze(1)   # residual, unbounded

def run(arm, Z, epochs=60, bs=4096, lr=1e-3):
    if Z is not None:
        mu, sd = Z[tr.si.unique()].mean(0), Z[tr.si.unique()].std(0) + 1e-8
        Z = (Z - mu) / sd
    T, E = tens(tr, Z), tens(te, Z)
    net = Net(0 if Z is None else Z.shape[1], len(didx)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(T['y'])
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            b = {k: v[perm[i:i+bs]] for k, v in T.items()}
            opt.zero_grad(); loss = nn.functional.mse_loss(net(b), b['y']); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): p = (net(E) + E['lut']).clamp(0, 1).cpu().numpy()
    y = te.viability01.values
    rmse = float(np.sqrt(((p - y) ** 2).mean()))
    # per-drug between-patient Spearman, within each (drug, dose) cell, then averaged
    t = te.copy(); t['pred'] = p
    per = []
    for (dr, dz), g in t.groupby(['inhibitor','dose_rank']):
        if g.dbgap_rnaseq_sample.nunique() >= 30 and g.pred.nunique() > 1:
            per.append(spearmanr(g.pred, g.viability01).statistic)
    per = np.array([x for x in per if np.isfinite(x)])
    print(f'  arm {arm}: RMSE {rmse:.4f} | between-patient Spearman mean {per.mean():.3f} '
          f'median {np.median(per):.3f} over {len(per)} (drug,dose) cells | frac>0 {(per>0).mean()*100:.0f}%')
    return {'arm': arm, 'rmse': rmse, 'bp_spearman_mean': float(per.mean()),
            'bp_spearman_median': float(np.median(per)), 'n_cells': int(len(per)),
            'frac_positive': float((per > 0).mean())}

# the lookup-table floor, measured on the same test points
m = tr.groupby(['inhibitor','dose_rank']).viability01.mean().rename('pred').reset_index()
tf = te.merge(m, on=['inhibitor','dose_rank'], how='left'); tf['pred'] = tf.pred.fillna(tr.viability01.mean())
floor_rmse = float(np.sqrt(((tf.pred - tf.viability01) ** 2).mean()))
print(f'\nlookup floor (per drug x dose mean, no patient info): RMSE {floor_rmse:.4f}')

res = [run('0 (no patient info)', None), run('R (RNA programs)', ACT['R']), run('A (chromatin programs)', ACT['A'])]
print('\nreading: the floor is now the model\'s starting point, so RMSE below 0.2401 means patient')
print('information helped, and A vs R is the test of whether chromatin wiring adds anything.')
json.dump({'floor_rmse': floor_rmse, 'arms': res,
           'n_train_points': int(len(tr)), 'n_test_points': int(len(te)),
           'n_train_specimens': int(tr.si.nunique()), 'n_test_specimens': int(te.si.nunique())},
          open(f'{ROOT}/results_arms_first_pass.json', 'w'), indent=2)
print('\nwrote results_arms_first_pass.json')
