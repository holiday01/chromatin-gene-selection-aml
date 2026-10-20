"""Three challenges raised in review, tested on existing data.

(A) PENALTY SUBSET. Every HSC arm A vs arm R result so far used only C=0.01 and C=0.1.
    S8/S12 report four penalties and C=0.001 is where other chromatin effects break. The
    "12/12" grid is therefore defined over half the penalties the protocol claims to report.
    Here: all four of C in {0.001, 0.01, 0.1, 1}.

(B) IS IT CONNECTIVITY? S_chrom sums over a gene's linked peaks, so link count sits inside
    the estimator. S21.1 found that in both AML references a pure link-count ranking beats
    arm A. Test a count-normalised arm A (mean over links instead of sum) and an arm A with
    link count partialled out, which is what a reviewer will ask for.

(C) INFERENCE. 100 repeated stratified folds reuse every patient 20 times, so paired
    Wilcoxon over those 100 differences measures sign stability under resampling, not
    population evidence. Report instead: per-repeat means (n=20) with the Nadeau-Bengio
    corrected t, and a patient-level bootstrap of the cross-wave difference.
"""
import os, sys, json
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]
NSPLIT, NREP, TOP_M, NBOOT = 5, 20, 200, 2000
REFS = {'BMMC': 'programs', 'AML-mal': 'programs_aml_malignant_v2', 'AML-all': 'programs_aml_all_v2'}

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
rskf = RepeatedStratifiedKFold(n_splits=NSPLIT, n_repeats=NREP, random_state=0)
folds = list(rskf.split(np.zeros((okm.sum(), 1)), y))
print(f'labelled {okm.sum()} | xwave train {tr.sum()} test {te.sum()} '
      f'| excluded from xwave (cohort="Both") {okm.sum()-tr.sum()-te.sum()}')

def fold_scores(sel):
    """returns {C: (array of 100 fold AUROCs, xwave AUROC, xwave decision values)}"""
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

def nadeau_bengio(d_folds, n_tr, n_te):
    """Corrected resampled t-test for repeated k-fold cross-validation.

    Uses the individual fold differences with the variance inflated by
    1/(k*r) + n_test/n_train (Nadeau & Bengio 2003; Bouckaert & Frank 2004), which is the
    form used for repeated k-fold designs. An earlier version of this code averaged the
    folds within each repeat and then applied 1/r + n_test/n_train to the variance of those
    r means; that variant has no derivation behind it and understates the uncertainty,
    because averaging within a repeat removes the fold-to-fold variation that the
    correction term is meant to account for.
    """
    d = np.asarray(d_folds).ravel()
    m, v = d.mean(), d.var(ddof=1)
    corr = (1.0 / len(d)) + (n_te / n_tr)
    se = np.sqrt(v * corr) if v > 0 else np.nan
    t = m / se if se and np.isfinite(se) and se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df=len(d) - 1) if np.isfinite(t) else np.nan
    return m, d, t, p

def boot_xwave(dvA, dvR, yte, rng, n=NBOOT):
    """patient-level bootstrap of the paired cross-wave AUROC difference"""
    idx = np.arange(len(yte)); d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(yte[b])) < 2: continue
        d.append(roc_auc_score(yte[b], dvA[b]) - roc_auc_score(yte[b], dvR[b]))
    d = np.array(d)
    return d.mean(), np.percentile(d, [2.5, 97.5]), (d <= 0).mean()

rng = np.random.default_rng(5)
yte = yall[te]
n_tr_f, n_te_f = okm.sum() * (NSPLIT - 1) / NSPLIT, okm.sum() / NSPLIT
rows = []
for rname, tag in REFS.items():
    pr = np.load(f'{ROOT}/data/programs/{tag}.npz', allow_pickle=True)
    pgenes = pr['genes'].astype(str); cts = list(pr['cell_types'].astype(str))
    elig = pr['eligible']; jH = cts.index('HSC')
    # W_A/W_R are top-m MASKED; re-ranking them returns the same 200 genes and fakes a
    # null. Use the unthresholded scores: stored in the npz for the AML references, and
    # recomputed from the saved peak specificity for BMMC.
    if 'S_chrom' in pr.files:
        WA = np.array(pr['S_chrom'][:, jH]); WR = np.array(pr['S_gene'][:, jH])
    else:
        sp_ = np.load(f'{ROOT}/data/programs/specificity_bmmc.npz', allow_pickle=True)
        assert list(sp_['genes'].astype(str)) == list(pgenes)
        assert list(sp_['peaks'].astype(str)) == list(pr['peaks'].astype(str))
        Sc = np.zeros((len(pgenes), len(cts)), np.float32)
        np.add.at(Sc, pr['link_g'], pr['link_r'][:, None] * sp_['S_peak'][pr['link_p']])
        WA = np.array(Sc[:, jH]); WR = np.array(sp_['S_gene'][:, jH])
    chk = set(np.flatnonzero(pr['W_A'][:, jH]))
    reb = set(np.argsort(-np.where(elig, WA, -np.inf))[:TOP_M])
    print(f'  [{rname}] unthresholded score reproduces {len(chk & reb)}/{len(chk)} of stored arm A')
    nlink = np.bincount(pr['link_g'], minlength=len(pgenes)).astype(float)
    # (B) count-normalised and count-partialled variants of the arm A score
    WA_mean = np.divide(WA, nlink, out=np.zeros_like(WA), where=nlink > 0)
    m = elig & (nlink > 0) & np.isfinite(WA)
    sl, ic = np.polyfit(np.log1p(nlink[m]), WA[m], 1)
    WA_resid = np.where(m, WA - (sl * np.log1p(nlink) + ic), -np.inf)
    def top(score):
        s = np.where(elig, np.nan_to_num(score, nan=-np.inf, neginf=-np.inf), -np.inf)
        idx = np.argsort(-s)[:TOP_M]
        return np.array([EX[pgenes[i]] for i in idx if np.isfinite(s[i]) and pgenes[i] in EX])
    ARMS = {'armA_sum': top(WA), 'armA_mean_per_link': top(WA_mean),
            'armA_count_partialled': top(WA_resid), 'armR': top(WR), 'n_links_only': top(nlink)}
    sc = {k: fold_scores(v) for k, v in ARMS.items()}
    print(f'\n########## {rname} ##########')
    print(f'{"arm":24s}{"n":>4s}' + ''.join(f'{f"CV C={C:g}":>11s}{"xw":>8s}' for C in C_GRID))
    for k, v in ARMS.items():
        print(f'  {k:22s}{len(v):4d}' + ''.join(f'{sc[k][C][0].mean():11.3f}{sc[k][C][1]:8.3f}' for C in C_GRID))
    for k in ['armA_sum', 'armA_mean_per_link', 'armA_count_partialled', 'n_links_only']:
        print(f'\n  --- {k} minus armR ---')
        for C in C_GRID:
            d = sc[k][C][0] - sc['armR'][C][0]
            mean_d, per_rep, t, p = nadeau_bengio(d, n_tr_f, n_te_f)
            wp = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
            bm, bci, bp = boot_xwave(sc[k][C][2], sc['armR'][C][2], yte, rng)
            print(f'   C={C:<6g} CV {mean_d:+.4f} | naive Wilcoxon p={wp:8.3g} | '
                  f'NB-corrected t={t:+.2f} p={p:6.3f} | xwave {sc[k][C][1]-sc["armR"][C][1]:+.4f} '
                  f'boot 95% CI [{bci[0]:+.3f},{bci[1]:+.3f}] P(d<=0)={bp:.3f}')
            rows.append(dict(reference=rname, arm=k, C=C, cv_diff=mean_d, wilcoxon_p=wp,
                             nb_t=t, nb_p=p, xwave_diff=sc[k][C][1]-sc['armR'][C][1],
                             boot_lo=bci[0], boot_hi=bci[1], boot_p=bp,
                             cv_positive=bool(mean_d > 0), xwave_positive=bool(sc[k][C][1] > sc['armR'][C][1])))
D = pd.DataFrame(rows)
D.to_csv(f'{ROOT}/results/robustness_check.csv', index=False)
print('\n================ grid tally over all four penalties ================')
for k in ['armA_sum', 'armA_mean_per_link', 'armA_count_partialled', 'n_links_only']:
    s = D[D.arm == k]
    print(f'  {k:24s} CV positive {s.cv_positive.sum():2d}/{len(s)} | '
          f'xwave positive {s.xwave_positive.sum():2d}/{len(s)} | '
          f'NB p<0.05 {(s.nb_p < 0.05).sum():2d}/{len(s)} | '
          f'bootstrap CI excludes 0 {(s.boot_lo > 0).sum():2d}/{len(s)}')
print('\nwrote results/robustness_check.csv')
