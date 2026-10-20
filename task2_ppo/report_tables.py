"""Task 2 (PPO): tables, trajectory plots and qualitative candidates.

Generated from the analysis cells of the Colab notebook so the tables, plots and qualitative
examples can be regenerated from the saved result files without the notebook.
Run from the repository root:  python -m task2_ppo.report_tables
"""
import matplotlib
matplotlib.use("Agg")


def display(x):
    print(x.to_string() if hasattr(x, "to_string") else x)


# ---- notebook cell 31 ----
import json, glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
R='results/task2_ppo'; os.makedirs(f'{R}/figures',exist_ok=True)
def log(name): return pd.read_json(f'{R}/train_log_{name}.jsonl', lines=True)
std=log('standard'); meta=json.load(open(f'{R}/run_meta_standard.json'))
print(f"Standard continuation: {meta['updates']} updates | wall-clock {meta['wall_clock_s']:.0f}s | peak VRAM {meta['peak_vram_gb']:.2f} GB | {meta['gpu']}")
cols=['reward_raw','kl_token','policy_loss','value_loss','entropy','clip_fraction','grad_norm_policy','grad_norm_value','resp_len','explained_var']
fig,ax=plt.subplots(2,5,figsize=(21,7))
for a,c in zip(ax.ravel(),cols):
    a.plot(std['update'],std[c],marker='o',ms=3); a.set_title(c); a.set_xlabel('update')
plt.tight_layout(); plt.savefig(f'{R}/figures/standard_trajectories.png',dpi=130); plt.show()
display(std[['update']+cols+['clip_fraction_last_epoch','eos_rate','value_mean','return_mean','skipped_optimizer_steps']].round(4))


# ---- notebook cell 32 ----
def stability(df):
    return {'reward_first3':df.reward_raw[:3].mean(),'reward_last3':df.reward_raw[-3:].mean(),
            'reward_delta':df.reward_raw[-3:].mean()-df.reward_raw[:3].mean(),
            'kl_final':df.kl_token.iloc[-1],'kl_max':df.kl_token.max(),
            'gradnorm_max':df.grad_norm_policy.max(),'gradnorm_cv':df.grad_norm_policy.std()/df.grad_norm_policy.mean(),
            'policy_loss_std':df.policy_loss.std(),'clip_frac_mean':df.clip_fraction.mean(),
            'ratio_max':df.ratio_max.max(),'value_loss_last':df.value_loss.iloc[-1],'expl_var_mean':df.explained_var.mean(),
            'skipped_steps':df.skipped_optimizer_steps.sum()}
clip_forks=['eps0.05_kl0.1','eps0.2_kl0.1','eps0.5_kl0.1']
kl_forks=['eps0.2_kl0','eps0.2_kl0.1','eps0.2_kl0.2']
print('== Clipping study: training stability =='); display(pd.DataFrame({n:stability(log(n)) for n in clip_forks}).T.round(4))
print('== KL-pressure study: training trajectories summary =='); display(pd.DataFrame({n:stability(log(n)) for n in kl_forks}).T.round(4))
ev=[json.load(open(f)) for f in glob.glob(f'{R}/eval_*.json')]
order=['reference','midpoint','standard']+sorted(set(clip_forks+kl_forks))
E=pd.DataFrame(ev).set_index('name').reindex([o for o in order if o in [e['name'] for e in ev]])
print('== Held-out evaluation (same prompts/decoding/seed for all) =='); display(E[['n_prompts','reward_mean','reward_std','kl_token_mean','kl_seq','entropy_mean','len_mean','len_std','len_iqr','eos_rate','truncated_frac']].round(4))
fig,ax=plt.subplots(1,3,figsize=(16,4))
for n in kl_forks:
    d=log(n)
    for a,c in zip(ax,['reward_raw','kl_token','entropy']): a.plot(d['update'],d[c],marker='o',label=n); a.set_title(c); a.set_xlabel('update')
ax[0].legend(); plt.tight_layout(); plt.savefig(f'{R}/figures/kl_study_trajectories.png',dpi=130); plt.show()
cs=json.load(open(f'{R}/cached_clipping_study.json'))
print('cache consistency:',cs['check'])
print('== Cached batch, loaded midpoint vs cached old log-probs =='); display(pd.DataFrame(cs['static']).round(5))
rows=[]
for p in cs['probes']:
    last=p['curve'][-1]; rows.append({'eps':p['eps'],'lr_mult':p['lr_mult'],**{k:last[k] for k in ['clip_fraction','affected_fraction','ratio_mean_abs_dev','ratio_max','surrogate_clipped','surrogate_unclipped']}})
print('== After probe gradient steps on the cached batch =='); display(pd.DataFrame(rows).round(5))


# ---- notebook cell 34 ----
from common.metrics import safe_corr
def gens(n): return {r['source_index']:r for r in map(json.loads,open(f'{R}/generations_{n}.jsonl'))}
print('reward-vs-length correlation within each condition (reward-model length bias check):')
for n in E.index:
    g=list(gens(n).values()); print(f"  {n:16s} corr(reward,len)={safe_corr([x['reward'] for x in g],[x['length'] for x in g]):+.3f}  corr(reward,kl_seq)={safe_corr([x['reward'] for x in g],[x['kl_seq'] for x in g]):+.3f}")
def show(a,b,key,n=3,reverse=True):
    ga,gb=gens(a),gens(b); common=sorted(set(ga)&set(gb))
    ranked=sorted(common,key=lambda i:key(ga[i],gb[i]),reverse=reverse)[:n]
    for i in ranked:
        print('='*100); print('PROMPT [%d]:'%i, ga[i]['prompt'][-1]['content'][:300].replace('\n',' '))
        for tag,g in ((a,ga[i]),(b,gb[i])):
            print(f"--- {tag}: reward={g['reward']:.2f} len={g['length']} kl_seq={g['kl_seq']:.1f} eos={g['terminated']}"); print(g['response'][:600].replace('\n',' '))
print('\n##### DISAGREE: beta_KL=0 gets higher reward than beta_KL=0.2 but a longer / unterminated response #####')
show('eps0.2_kl0','eps0.2_kl0.2',lambda x,y:(x['reward']-y['reward'])+0.002*(x['length']-y['length'])-(2 if x['terminated'] else 0))
print('\n##### AGREE: beta_KL=0 higher reward AND not longer #####')
show('eps0.2_kl0','eps0.2_kl0.2',lambda x,y:(x['reward']-y['reward'])-0.01*max(0,x['length']-y['length']))
