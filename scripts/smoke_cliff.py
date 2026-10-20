"""Smoke test: run CLIFF on its own bundled in-vitro example and report Pearson r.
Usage: python -I smoke_cliff.py <path-to-CLIFF-repo>
"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import pearsonr

repo = os.path.abspath(sys.argv[1])
sys.path.insert(0, repo)
from cliff.Cliff import Cliff
from cliff.DataContainer import DataContainer
from sklearn.preprocessing import StandardScaler

D = os.path.join(repo, 'data')
df_full = pd.read_csv(os.path.join(D, 'invitro_bulk_expr_cpm.csv'), index_col=0)
sign_mat = df_full.T.iloc[7:11].T
sign_mat.sort_index(axis=0, inplace=True)
S = StandardScaler().fit_transform(sign_mat)

W_single = pd.read_csv(os.path.join(D, 'invitro_climb_prop.csv'), index_col=0)
W_1 = W_single.copy(); W_1.index = W_single.index + '_n1'
W_2 = W_single.copy(); W_2.index = W_single.index + '_n2'
W = pd.concat([W_1, W_2]).T; W = W.iloc[0:4, :]

drug = sys.argv[2] if len(sys.argv) > 2 else 'Nilotinib'
dr = pd.read_csv(os.path.join(D, 'invitro_drug_resp', f'Y_48h_{drug}.csv'), index_col=0)
resp_bulk, resp_true = dr.iloc[8:], dr.iloc[:8]
true_norm = pd.DataFrame(np.clip(resp_true.values, 0.0, 1.0), index=resp_true.index, columns=resp_true.columns)
names = [x.replace('_n1', '').replace('_n2', '') for x in true_norm.index][::2]
true_av = pd.DataFrame(np.array([np.average(true_norm.values[:, j].reshape(-1, 2), axis=1)
                                 for j in range(resp_true.shape[1])])).T
true_av.index = names; true_av.columns = true_norm.columns
true_av.sort_index(axis=0, inplace=True)

Y = np.clip(np.array(resp_bulk), 0.0, 1.0)
print(f'drug={drug}  sign_mat={sign_mat.shape}  W={W.shape}  Y={Y.shape}  aligned={all(W.T.columns == sign_mat.columns)}')

dc = DataContainer(sign_mat, S, np.array(W), Y, n_dose=10)
PI_true = np.array(true_av.T)
obj = Cliff(dc, n_iter=100, n_fold_cv=4, param_C=30, PI_true=PI_true, print_steps=False)
obj.run()
pi = np.array(obj.PI).flatten()
gt = np.array(true_av.T).flatten()
r, p = pearsonr(pi, gt)
print(f'CLIFF PI shape={np.array(obj.PI).shape}  Pearson r={r:.3f}  p={p:.2e}')
print('SMOKE TEST PASSED' if np.isfinite(r) else 'SMOKE TEST FAILED')
