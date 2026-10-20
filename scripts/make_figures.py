"""Figures 1-4."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT='/home/holiday01/2026_cm'; OUT=f'{ROOT}/manuscript/figures'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':.7,'xtick.major.width':.7,
                     'ytick.major.width':.7,'legend.frameon':False})
CA,CR_='#1b5e8c','#b4634a'; CG='#999999'
REFS=['BMMC','AML-mal','AML-all']
def panel(ax,letter):
    ax.text(-0.12,1.06,letter,transform=ax.transAxes,fontsize=10,fontweight='bold',va='top')
LBL={'BMMC':'Healthy marrow','AML-mal':'AML, malignant cells','AML-all':'AML, all cells'}
SHORT={'BMMC':'Healthy','AML-mal':'AML-M','AML-all':'AML-A'}

# ---------------- Figure 1: the 24 comparisons ----------------
rb=pd.read_csv(f'{ROOT}/results/robustness_check.csv'); a=rb[rb.arm=='armA_sum']
fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),sharey=True)
for ax,(col,ttl) in zip(axes,[('cv_diff','Cross-validation'),('xwave_diff','Held-out waves')]):
    ypos=[]; lbls=[]; k=0
    for ref in REFS:
        sub=a[a.reference==ref].sort_values('C')
        for _,r in sub.iterrows():
            ax.plot(r[col],k,'o',ms=4.5,color=CA if r[col]>0 else CR_)
            if col=='cv_diff':
                lo,hi=r[col]-1.96*abs(r[col]/max(r.nb_t,1e-9)),r[col]+1.96*abs(r[col]/max(r.nb_t,1e-9))
                ax.plot([lo,hi],[k,k],'-',color=CA,lw=1,alpha=.55)
            else:
                ax.plot([r.boot_lo,r.boot_hi],[k,k],'-',color=CG,lw=1,alpha=.8)
            lbls.append(f'{SHORT[ref]}  C={r.C:g}'); ypos.append(k); k+=1
        k+=0.8
    ax.axvline(0,color='k',lw=.7,ls='--'); ax.set_title(ttl,fontsize=8.5)
    ax.set_xlabel('arm A − arm R (ΔAUROC)')
axes[0].set_yticks(ypos); axes[0].set_yticklabels(lbls,fontsize=6.5)
axes[0].invert_yaxis()
for ax,L in zip(axes,'ab'): panel(ax,L)
axes[1].legend(handles=[Line2D([],[],color=CA,lw=1,label='Nadeau–Bengio 95% CI'),
                        Line2D([],[],color=CG,lw=1,label='patient bootstrap 95% CI')],
               loc='lower right',fontsize=6.5)
fig.tight_layout(); fig.savefig(f'{OUT}/Figure1.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 2: gene budget ----------------
bs=pd.read_csv(f'{ROOT}/results/budget_sweep.csv')
fig,axes=plt.subplots(1,3,figsize=(7.2,2.6),sharey=True)
for ax,ref in zip(axes,REFS):
    s=bs[bs.reference==ref].sort_values('k')
    for c,st,lab in [(0.01,'-','C=0.01'),(0.1,'--','C=0.1')]:
        ax.plot(s.k,s[f'd_cv_{c:g}'],st,color=CA,lw=1.4,marker='o',ms=3,label=f'CV {lab}')
        ax.plot(s.k,s[f'd_xw_{c:g}'],st,color=CG,lw=1.1,marker='s',ms=2.5,label=f'waves {lab}')
    ax.axhline(0,color='k',lw=.7,ls=':'); ax.axvline(100,color=CR_,lw=.8,alpha=.5)
    ax.set_xscale('log'); ax.set_xticks([10,25,60,150]); ax.set_xticklabels([10,25,60,150])
    ax.set_title(LBL[ref],fontsize=8); ax.set_xlabel('genes per programme')
axes[0].set_ylabel('arm A − arm R (ΔAUROC)'); axes[0].legend(fontsize=6,ncol=1)
for ax,L in zip(axes,'abc'): panel(ax,L)
fig.tight_layout(); fig.savefig(f'{OUT}/Figure2.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 3: versus the published set ----------------
L=json.load(open(f'{ROOT}/results/baseline_lsc104.json'))
names=[('HSC arm A top-99 (chromatin)','Arm A',CA),('HSC arm R top-99 (RNA)','Arm R',CR_),
       ('Ng2016 LSC genes (published, experimental)','Ng et al. LSC genes','#6a6a6a'),
       ('top-99 by variance','Top variance',CG)]
rand=[k for k in L['cv'] if k.startswith('var-matched random')]
fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),sharey=True)
for ax,(key,ttl) in zip(axes,[('cv','Cross-validation'),('xwave','Held-out waves')]):
    for j,C in enumerate(['0.01','0.1']):
        rv=[L[key][k][C] for k in rand]
        ax.scatter(np.full(len(rv),j)+np.random.default_rng(1).normal(0,.045,len(rv)),rv,
                   s=7,color=CG,alpha=.5,zorder=1)
        for nm,lab,col in names:
            ax.plot(j,L[key][nm][C],'o',ms=7,color=col,zorder=3,
                    label=lab if j==0 else None)
    ax.set_xticks([0,1]); ax.set_xticklabels(['C=0.01','C=0.1']); ax.set_title(ttl,fontsize=8.5)
    ax.set_xlim(-.4,1.4)
axes[0].scatter([],[],s=7,color=CG,alpha=.6,label='variance-matched random')
axes[0].set_ylabel('AUROC'); axes[0].legend(fontsize=6.3,loc='lower left',handletextpad=.5)
for ax,L in zip(axes,'ab'): panel(ax,L)

fig.tight_layout(); fig.savefig(f'{OUT}/Figure3.png',dpi=300,bbox_inches='tight'); plt.close(fig)

# ---------------- Figure 4: what carries the signal ----------------
loc=json.load(open(f'{ROOT}/results/locus_arm.json'))
pg=json.load(open(f'{ROOT}/results/link_permutation.json'))
pw=json.load(open(f'{ROOT}/results/link_permutation_window100.json'))
dt=pd.read_csv(f'{ROOT}/results/distributed_test.csv')
fig,axes=plt.subplots(1,3,figsize=(7.2,2.8))
ax=axes[0]
bars=[('Arm A',loc['cv']['arm A (correlation-linked)']['0.01'],CA),
      ('Window permuted',pw['window_C0.01']['null_cv_mean'],'#8fb4cc'),
      ('Window mean,\nno linking',loc['cv']['locus_mean (no linking)']['0.01'],CG),
      ('Arm R',loc['cv']['arm R (expression)']['0.01'],CR_),
      ('Link count\nalone',loc['cv']['n_links only']['0.01'],'#c9c9c9'),
      ('Links permuted\ngenome-wide',pg['global_C0.01']['null_cv_mean'],'#e0e0e0')]
for i,(nm,v,c) in enumerate(bars):
    ax.plot([.68,v],[i,i],'-',color=c,lw=1,alpha=.45)
    ax.plot(v,i,'o',ms=6,color=c)
ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in bars],fontsize=6.5)
ax.invert_yaxis(); ax.set_xlim(.68,.79); ax.set_xlabel('AUROC (cross-validation)')
ax.set_title('Score variants',fontsize=8)
ax=axes[1]
for nm,lab,col in [('top-k','Top-ranked',CA),('random-in-A','Random within arm A',CG),
                   ('bottom-k','Lowest-ranked','#8fb4cc'),('random-in-pool','Random from pool',CR_)]:
    sub=dt[dt.arm==nm].sort_values('size')
    axes[1].plot(sub['size'],sub['cv_0.01'],'-o',ms=3,lw=1.3,color=col,label=lab)
axes[1].set_xlabel('subset size'); axes[1].set_ylabel('AUROC'); axes[1].legend(fontsize=6)
axes[1].set_title('Subsets of the arm A gene set',fontsize=8)
ax=axes[2]
rb2=pd.read_csv(f'{ROOT}/results/robustness_check.csv')
lbl4=[('armA_sum','Arm A',CA),('armA_count_partialled','Count removed','#8fb4cc'),
      ('armA_mean_per_link','Per-link mean',CG),('n_links_only','Count only','#c9c9c9')]
for i,(nm,lab,col) in enumerate(lbl4):
    ax.bar(i,(rb2[rb2.arm==nm].cv_diff>0).sum(),color=col,width=.62)
ax.set_xticks(range(len(lbl4))); ax.set_xticklabels([l[1] for l in lbl4],fontsize=6.5,rotation=30,ha='right')
ax.set_ylabel('cells favouring the arm (of 12)',fontsize=7); ax.axhline(6,color='k',lw=.7,ls=':')
ax.set_title('Connectivity controls',fontsize=8)
for a_,L in zip(axes,'abc'): panel(a_,L)
fig.tight_layout(); fig.savefig(f'{OUT}/Figure4.png',dpi=300,bbox_inches='tight'); plt.close(fig)
print('wrote Figure1-4.png to manuscript/figures/')
for f in sorted(os.listdir(OUT)): print(' ',f,os.path.getsize(f'{OUT}/{f}')//1024,'KB')
