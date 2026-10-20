"""Task 4 (safety): calibration, category, audit-agreement tables and qualitative evidence.

Generated from the analysis cells of the Colab notebook so the tables, plots and qualitative
examples can be regenerated from the saved result files without the notebook.
Run from the repository root:  python -m task4_safety.report_tables
"""
import matplotlib
matplotlib.use("Agg")


def display(x):
    print(x.to_string() if hasattr(x, "to_string") else x)


# ---- notebook cell 31 ----
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from common.data import load_yaml
from task4_safety.evaluate_safety import run_analysis, audit_agreement, judge_vs_manual_policy_metrics, LABELS, POLICIES
cfg=load_yaml('configs/feedback.yaml'); R='results/task4_safety'
df,cal,cats,diag,tests,man=run_analysis(cfg)
pd.set_option('display.width',250,'display.max_columns',50)
main=['n_safe','n_unsafe','safe_answer_rate','safe_over_refusal_rate','unsafe_compliance_rate','unsafe_justified_refusal_rate','ambiguous_rate','mean_response_tokens','median_response_tokens','truncated_frac']
print('== Safety-calibration comparison (rates over SAFE / UNSAFE prompts; ambiguous over all 450) =='); display(cal[main].round(3))
ci=lambda c:cal[[c+'_lo',c+'_hi']].round(3).apply(lambda r:f"[{r.iloc[0]}, {r.iloc[1]}]",axis=1)
print('95% Wilson intervals:'); display(pd.DataFrame({c:ci(c) for c in ['safe_answer_rate','safe_over_refusal_rate','unsafe_compliance_rate','unsafe_justified_refusal_rate']}))
print('== Instrument diagnostics (no manual labels needed) =='); display(diag.round(3))
print('== Paired exact McNemar tests (same prompts; raw p-values, 12 comparisons — interpret cautiously) =='); display(tests.round(4))
fig,ax=plt.subplots(1,3,figsize=(17,4.2))
for p in POLICIES:
    r=cal.loc[p]; ax[0].errorbar(r.safe_over_refusal_rate,r.unsafe_compliance_rate,xerr=[[max(0.0,r.safe_over_refusal_rate-r.safe_over_refusal_rate_lo)],[max(0.0,r.safe_over_refusal_rate_hi-r.safe_over_refusal_rate)]],yerr=[[max(0.0,r.unsafe_compliance_rate-r.unsafe_compliance_rate_lo)],[max(0.0,r.unsafe_compliance_rate_hi-r.unsafe_compliance_rate)]],fmt='o',capsize=3,label=p)
ax[0].set_xlabel('SAFE prompts: over-refusal rate'); ax[0].set_ylabel('UNSAFE prompts: unsafe-compliance rate'); ax[0].legend(); ax[0].set_title('calibration trade-off (lower-left is better)')
for a,cls in zip(ax[1:],['SAFE','UNSAFE']):
    t=pd.DataFrame({p:df[(df.policy==p)&(df.benchmark_class==cls)].label.value_counts(normalize=True) for p in POLICIES}).reindex(LABELS).fillna(0).T
    t.plot.bar(stacked=True,ax=a,legend=(cls=='UNSAFE')); a.set_title(f'{cls} prompts: judge-label distribution'); a.set_xlabel('')
plt.tight_layout(); plt.savefig(f'{R}/calibration.png',dpi=130); plt.show()


# ---- notebook cell 33 ----
for p in POLICIES:
    print(f'===== {p.upper()}: counts per category x judge label ====='); display(cats[p])
safe_types=sorted(df[df.benchmark_class=='SAFE'].type.unique()); unsafe_types=sorted(df[df.benchmark_class=='UNSAFE'].type.unique())
fig,ax=plt.subplots(1,2,figsize=(16,6))
for a,types,lab,title in [(ax[0],safe_types,'OVER_REFUSAL','SAFE categories: over-refusal rate'),(ax[1],unsafe_types,'UNSAFE_COMPLIANCE','UNSAFE categories: unsafe-compliance rate')]:
    M=pd.DataFrame({p:df[(df.policy==p)&(df.type.isin(types))].groupby('type').label.apply(lambda s:(s==lab).mean()) for p in POLICIES}).reindex(types)
    im=a.imshow(M.values,cmap='Reds',vmin=0,vmax=1,aspect='auto'); a.set_xticks(range(4)); a.set_xticklabels(POLICIES); a.set_yticks(range(len(types))); a.set_yticklabels(types,fontsize=8); a.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): a.text(j,i,f'{M.values[i,j]:.2f}',ha='center',va='center',fontsize=7)
plt.tight_layout(); plt.savefig(f'{R}/category_heatmaps.png',dpi=130); plt.show()
print('Full tables saved:', f'{R}/category_label_distribution.csv', 'and category_counts_<policy>.csv')


# ---- notebook cell 35 ----
man=run_analysis(cfg)[-1]
if man is None:
    print('No manual labels found — fill results/task4_safety/manual_audit_sheet.csv (section 4) and re-run this cell.')
else:
    agr,conf=audit_agreement(man)
    print(f"labelled rows: {agr['n_labeled']} / 240 | exact agreement: {agr['agreement']:.3f} | Cohen's kappa: {agr['cohen_kappa']:.3f} | refuse-vs-comply agreement: {agr['binary_refuse_vs_comply_agreement']:.3f}")
    print(f"AMBIGUOUS rate — manual: {agr['manual_ambiguous_rate']:.3f} | judge: {agr['judge_ambiguous_rate']:.3f}")
    print('mean judge confidence when agreeing / disagreeing:', round(agr['mean_confidence_when_agree'],3), '/', round(agr['mean_confidence_when_disagree'],3))
    print('\nConfusion (rows = manual, columns = judge):'); display(conf)
    for k in ['agreement_by_policy','agreement_by_class','agreement_by_judge_label']: print(k); display(pd.DataFrame(agr[k]).T)
    print('== Same headline rates on the audited subset: judge labels vs manual labels (effect of judge errors on the policy comparison) ==')
    display(judge_vs_manual_policy_metrics(man).round(3))
    dis=man[man.manual_label!=man.label]
    print(f'\n{len(dis)} disagreements (saved to {R}/audit_disagreements.csv):')
    display(dis[['audit_row','policy','benchmark_class','type','manual_label','label','confidence','rationale_tag','notes']].reset_index(drop=True))
    fig,a=plt.subplots(figsize=(5.5,4.5)); im=a.imshow(conf.values,cmap='Blues'); a.set_xticks(range(5)); a.set_xticklabels(LABELS,rotation=45,ha='right',fontsize=8); a.set_yticks(range(5)); a.set_yticklabels(LABELS,fontsize=8)
    a.set_xlabel('judge'); a.set_ylabel('manual')
    for i in range(5):
        for j in range(5): a.text(j,i,conf.values[i,j],ha='center',va='center')
    plt.tight_layout(); plt.savefig(f'{R}/audit_confusion.png',dpi=130); plt.show()


# ---- notebook cell 37 ----
full=df.copy()
if man is not None: full=full.merge(man[['xstest_id','policy','manual_label']],on=['xstest_id','policy'],how='left')
else: full['manual_label']=np.nan
audited=set(pd.read_csv(f'{R}/manual_audit_ids.csv').xstest_id)
def show(pid):
    g=full[full.xstest_id==pid].set_index('policy').reindex(POLICIES)
    labs=g.label.tolist(); pol_diff=len(set(labs))>1
    errs=[(p, g.loc[p,'manual_label']!=g.loc[p,'label']) for p in POLICIES if isinstance(g.loc[p,'manual_label'],str)]
    print('='*110); print(f"xstest_id {pid} | {g.iloc[0].benchmark_class} | {g.iloc[0].type} | policy_difference={pol_diff} | judge_error={ {p:e for p,e in errs} if errs else 'not audited'}")
    print('PROMPT:', g.iloc[0].prompt)
    for p in POLICIES:
        print(f"  [{p}] judge={g.loc[p,'label']} (conf {g.loc[p,'confidence']:.2f}) manual={g.loc[p,'manual_label'] if isinstance(g.loc[p,'manual_label'],str) else '-'} len={g.loc[p,'response_tokens']}")
        print('     ', str(g.loc[p,'response'])[:300].replace('\n',' '))
def pick(cls,label,k=2):
    ids=full[(full.benchmark_class==cls)&(full.label==label)].xstest_id.unique()
    div=[i for i in ids if full[full.xstest_id==i].label.nunique()>1]        # prefer prompts where policies differ
    pool=sorted(div or ids.tolist(), key=lambda i:(i not in audited,i))
    return pool[:k]
print('##### HARMFUL COMPLIANCE (UNSAFE prompt, judge = UNSAFE_COMPLIANCE) #####')
for i in pick('UNSAFE','UNSAFE_COMPLIANCE'): show(i)
print('\n##### JUSTIFIED REFUSAL (UNSAFE prompt, judge = JUSTIFIED_REFUSAL) #####')
for i in pick('UNSAFE','JUSTIFIED_REFUSAL'): show(i)
print('\n##### EXAGGERATED REFUSAL (SAFE prompt, judge = OVER_REFUSAL) #####')
for i in pick('SAFE','OVER_REFUSAL'): show(i)
