"""Regenerate the two report figures (report/figs/*.pdf) from the saved logs.  Run: python -m scripts.make_figures"""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D='results'; O='report/figs/'
C=['#2a78d6','#eb6834','#1baf7a','#eda100']
plt.rcParams.update({'font.size':7,'axes.titlesize':7.5,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.color':'#e6e6e6','grid.linewidth':0.5,'axes.edgecolor':'#888','lines.linewidth':1.4,'font.family':'serif'})
rd=lambda f:[json.loads(l) for l in open(f)]
P=rd(f'{D}/task2_ppo/train_log_standard.jsonl')
G=rd(f'{D}/task3_grpo/train_log_standard.jsonl')
u=np.arange(1,21)
f,ax=plt.subplots(2,5,figsize=(7.0,2.7))
def pl(a,y,t,c,mask=None):
    y=np.array(y,float); a.plot(u,y,'-o',ms=2.5,color=c); a.set_title(t)
    if mask is not None:
        for i in np.where(mask)[0]: a.axvspan(i+0.5,i+1.5,color='#bbbbbb',alpha=0.35,lw=0)
noeos=np.array([x['eos_rate']==0 for x in P])
pl(ax[0,0],[x['reward_raw'] for x in P],'PPO reward (raw RM)',C[0],noeos)
pl(ax[0,1],[x['kl_token']*1e3 for x in P],'PPO KL to ref (×1e-3)',C[0])
pl(ax[0,2],[x['value_loss'] for x in P],'PPO value loss',C[0])
pl(ax[0,3],[x['explained_var'] for x in P],'PPO explained var.',C[0]); ax[0,3].set_ylim(-120,5)
pl(ax[0,4],[x['resp_len'] for x in P],'PPO response length',C[0],noeos)
msk=np.array([x['masked_frac']==1 for x in G])
pl(ax[1,0],[x['reward'] for x in G],'GRPO reward',C[1],msk)
pl(ax[1,1],[x['group_reward_std'] for x in G],'GRPO group reward std',C[1],msk)
pl(ax[1,2],[x['grad_norm'] for x in G],'GRPO grad norm',C[1],msk)
pl(ax[1,3],[x['entropy'] if x['masked_frac']<1 else np.nan for x in G],'GRPO entropy',C[1],msk)
pl(ax[1,4],[x['resp_len'] for x in G],'GRPO mean length',C[1],msk)
for a in ax[1]: a.set_xlabel('update')
for a in ax.flat: a.set_xticks([1,5,10,15,20])
plt.tight_layout(pad=0.3); plt.savefig(O+'online_traj.pdf'); plt.close()

# group size + normalization
gs=json.load(open(f'{D}/task3_grpo/group_size_study.json'))['random_mean']
f,ax=plt.subplots(1,3,figsize=(7.0,1.9))
Ks=[2,4,8]; bins=['hard','medium','easy']
for j,b in enumerate(bins):
    ax[0].plot(Ks,[gs[str(k)][b]['mean_group_reward_std'] for k in Ks],'-o',ms=3,color=C[j],label=b)
    ax[1].plot(Ks,[gs[str(k)][b]['informative_rate_std>0.5'] for k in Ks],'-o',ms=3,color=C[j],label=b)
ax[0].set_title('Mean within-group reward std'); ax[1].set_title('Frac. groups with std > 0.5')
for a in ax[:2]: a.set_xticks(Ks); a.set_xlabel('K (192 generations fixed)'); a.set_xscale('log',base=2); a.set_xticks(Ks); a.set_xticklabels(Ks)
ax[0].legend(frameon=False,title='difficulty',fontsize=6,title_fontsize=6)
for j,(n,lab) in enumerate([('canonical_grpo','canonical (1/$T_k$)'),('dr_grpo','Dr. GRPO (1/$T_{max}$)')]):
    S=[r for r in rd(f'{D}/task3_grpo/seqstats_{n}.jsonl') if not r['masked'] and abs(r['adv'])>1e-6]
    ax[2].scatter([r['length'] for r in S],[r['grad_norm']/abs(r['adv']) for r in S],s=9,color=C[j],label=lab,edgecolor='white',linewidth=0.4)
ax[2].set_yscale('log'); ax[2].set_xlabel('completion length (tokens)'); ax[2].set_title('Per-completion grad norm / |A|'); ax[2].legend(frameon=False,fontsize=6)
plt.tight_layout(pad=0.3); plt.savefig(O+'grpo_k_norm.pdf'); plt.close()
print('ok')
