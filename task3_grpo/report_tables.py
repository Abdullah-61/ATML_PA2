"""Task 3 (GRPO): tables, plots and qualitative examples.

Generated from the analysis cells of the Colab notebook so the tables, plots and qualitative
examples can be regenerated from the saved result files without the notebook.
Run from the repository root:  python -m task3_grpo.report_tables
"""
import matplotlib
matplotlib.use("Agg")


def display(x):
    print(x.to_string() if hasattr(x, "to_string") else x)


# ---- notebook cell 32 ----
import json, glob, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
R='results/task3_grpo'; os.makedirs(f'{R}/figures',exist_ok=True)
def log(name): return pd.read_json(f'{R}/train_log_{name}.jsonl', lines=True)
std=log('standard'); meta=json.load(open(f'{R}/run_meta_standard.json'))
print(f"Standard continuation: {meta['updates']} updates | wall-clock {meta['wall_clock_s']:.0f}s | peak VRAM {meta['peak_vram_gb']:.2f} GB | total generated tokens {meta['total_generated_tokens']} | {meta['gpu']}")
cols=['reward','kl_k3','group_reward_std','frac_uninformative_groups','policy_loss','grad_norm','entropy','resp_len','masked_frac','clip_fraction']
fig,ax=plt.subplots(2,5,figsize=(21,7))
for a,c in zip(ax.ravel(),cols):
    a.plot(std['update'],std[c],marker='o',ms=3); a.set_title(c); a.set_xlabel('update')
plt.tight_layout(); plt.savefig(f'{R}/figures/standard_trajectories.png',dpi=130); plt.show()
display(std[['update']+cols+['kl_token','sample_entropy','eos_rate','scaler_retries']].round(4))
print('Run averages:'); display(std[cols].mean().round(4).to_frame('mean'))


# ---- notebook cell 33 ----
gs=json.load(open(f'{R}/group_size_study.json'))
print('Rules:', gs['rule']['informative'], '|', gs['rule']['difficulty'], '| total generations:', gs['rule']['total_generations'])
rows=[]
for k,parts in gs['random_mean'].items():
    for part,m in parts.items(): rows.append({'K':int(k),'bin':part,**m})
G=pd.DataFrame(rows)
print('== Equal-generation group-size study (random-partition means) ==')
display(G[['K','bin','n_groups','n_generations','informative_rate','mean_group_reward_std','signal_variance','advantage_variance']+[c for c in G if c.startswith('informative_rate_std>')]].round(4))
print('== Contiguous split ==')
display(pd.DataFrame([{'K':int(k),'bin':p,**m} for k,ps in gs['contiguous'].items() for p,m in ps.items()])[['K','bin','informative_rate','mean_group_reward_std','signal_variance']].round(4))
print('Spearman(prompt difficulty rank, mean group std) per K:', gs['spearman_difficulty_vs_group_std'])
fig,ax=plt.subplots(1,3,figsize=(16,4))
for a,c in zip(ax,['informative_rate','mean_group_reward_std','signal_variance']):
    P=G.pivot(index='bin',columns='K',values=c).loc[['overall','hard','medium','easy']]
    P.plot.bar(ax=a); a.set_title(c); a.set_xlabel('')
plt.tight_layout(); plt.savefig(f'{R}/figures/group_size.png',dpi=130); plt.show()


# ---- notebook cell 34 ----
forks=['canonical_grpo','dr_grpo']
ev=[json.load(open(f)) for f in glob.glob(f'{R}/eval_*.json')]
order=['reference','midpoint','standard']+forks
E=pd.DataFrame(ev).set_index('name').reindex([o for o in order if o in [e['name'] for e in ev]])
print('== Held-out evaluation (same prompts / decoding / seed for all) =='); display(E[['n_prompts','reward_mean','reward_std','kl_token_mean','kl_seq','entropy_mean','len_mean','len_std','len_iqr','corr_reward_len','eos_rate','truncated_frac']].round(4))
print('== Fork training summaries ==')
display(pd.DataFrame({n:{'reward_mean':log(n).reward.mean(),'kl_k3_last':log(n).kl_k3.iloc[-1],'len_mean':log(n).resp_len.mean(),
    'gen_tokens_total':json.load(open(f'{R}/run_meta_{n}.json'))['total_generated_tokens'],'grad_norm_mean':log(n).grad_norm.mean(),
    'group_std_mean':log(n).group_reward_std.mean(),'uninformative_frac':log(n).frac_uninformative_groups.mean(),
    'masked_frac':log(n).masked_frac.mean()} for n in forks}).T.round(4))
ns=json.load(open(f'{R}/normalization_summary.json'))
print('== Length-conditioned gradient statistics (short/long split at pooled median length = %.1f tokens) ==' % ns['pooled_median_length'])
display(pd.DataFrame({n:ns[n] for n in forks}).round(4))
fig,ax=plt.subplots(1,2,figsize=(12,4))
for n in forks:
    d=pd.DataFrame([json.loads(l) for l in open(f'{R}/seqstats_{n}.jsonl')]); d=d[(~d.masked)&(d.adv.abs()>0)]
    ax[0].scatter(d.length,d.grad_norm/d.adv.abs(),s=12,alpha=.6,label=n); ax[1].scatter(d.length,d.mass,s=12,alpha=.6,label=n)
ax[0].set_xlabel('completion length'); ax[0].set_ylabel('grad norm / |A|'); ax[1].set_xlabel('completion length'); ax[1].set_ylabel('policy-term weight mass |A|*T/norm/N')
ax[0].legend(); plt.tight_layout(); plt.savefig(f'{R}/figures/normalization_gradients.png',dpi=130); plt.show()


# ---- notebook cell 36 ----
cache=[json.loads(l) for l in open('cached/grpo_k_cache.jsonl')]
by={}
for r in cache: by.setdefault(r['source_index'],[]).append(r)
sp={p:np.std([x['reward'] for x in g]) for p,g in by.items()}
def show_group(p):
    g=sorted(by[p],key=lambda x:x['reward'])
    print('='*100); print(f"prompt {p}: reward std={sp[p]:.3f} | rewards={[round(x['reward'],2) for x in g]} | lengths={[x['completion_tokens'] for x in g]}")
    for x in (g[0],g[-1]): print(f"--- reward={x['reward']:.2f} len={x['completion_tokens']} eos={x['terminated_with_eos']}:", x['completion'][:350].replace('\n',' '))
print('##### MOST informative groups (large within-prompt spread) #####')
for p in sorted(sp,key=sp.get,reverse=True)[:2]: show_group(p)
print('\n##### LEAST informative groups (near-identical rewards) #####')
for p in sorted(sp,key=sp.get)[:2]: show_group(p)
def gens(n): return {r['source_index']:r for r in map(json.loads,open(f'{R}/generations_{n}.jsonl'))}
a,b=gens('canonical_grpo'),gens('dr_grpo'); common=sorted(set(a)&set(b))
print('\n##### NORMALISATION: prompts with the biggest length difference (Dr.-GRPO vs canonical) #####')
for i in sorted(common,key=lambda i:abs(b[i]['length']-a[i]['length']),reverse=True)[:3]:
    print('='*100); print('PROMPT [%d]:'%i, a[i]['prompt'][-1]['content'][:250].replace('\n',' '))
    for tag,g in (('canonical_grpo',a[i]),('dr_grpo',b[i])):
        print(f"--- {tag}: reward={g['reward']:.2f} len={g['length']} kl_seq={g['kl_seq']:.1f} eos={g['terminated']}"); print(g['response'][:500].replace('\n',' '))
print('\n##### reward and length move together (first) / disagree (second) #####')
sc=lambda i:(b[i]['reward']-a[i]['reward'])*(b[i]['length']-a[i]['length'])
for i in sorted(common,key=sc,reverse=True)[:1]+sorted(common,key=sc)[:1]:
    print('='*100); print('PROMPT [%d]:'%i, a[i]['prompt'][-1]['content'][:250].replace('\n',' '))
    for tag,g in (('canonical_grpo',a[i]),('dr_grpo',b[i])): print(f"--- {tag}: reward={g['reward']:.2f} len={g['length']}"); print(g['response'][:400].replace('\n',' '))
