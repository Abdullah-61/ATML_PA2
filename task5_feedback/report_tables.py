"""Task 5 (RLVR vs RLAIF): all tables, plots, qualitative evidence and cost.

Generated from the analysis cells of the Colab notebook so the tables, plots and qualitative
examples can be regenerated from the saved result files without the notebook.
Run from the repository root:  python -m task5_feedback.report_tables
"""
import matplotlib
matplotlib.use("Agg")


def display(x):
    print(x.to_string() if hasattr(x, "to_string") else x)


# ---- notebook cell 23 ----
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from common.data import load_yaml
from task5_feedback.compare_feedback import run_all, POLICIES
cfg = load_yaml('configs/feedback.yaml'); R = 'results/task5_feedback'
out = run_all(cfg)
pd.set_option('display.width', 250, 'display.max_columns', 50)
G = out['gsm']
cols = ['n', 'exact_acc', 'exact_acc_lo', 'exact_acc_hi', 'win_vs_sft', 'win_vs_sft_lo', 'win_vs_sft_hi', 'W', 'T', 'L',
        'has_final_rate', 'strict_format_rate', 'truncated_rate', 'mean_tokens', 'median_tokens', 'mean_words']
print('== GSM8K: policy comparison (has_final = verifier-readable `#### n`; strict = exactly one marker, on the last line) ==')
display(G['policy'][cols].round(3))
print('Head-to-head under the judge (A = RLVR, B = RLAIF):', G['h2h'])
print('\n== Paired exact McNemar tests on accuracy (same prompts) =='); display(G['tests'].round(4))
print('== Verifier vs judge agreement on the same pairs ==')
print('decisive = exactly one of the two responses is correct; verifier_tie = both correct or both wrong (verifier cannot rank)')
display(G['agreement'].round(3))
print('K=3 direct-RLAIF group reward vs binary verifier on the same responses:', json.dumps(G['group_summary'], indent=1))
print('Judge position diagnostic (picked_first_shown ≈ 0.5 means no position bias):', G['position'])

t = G['policy']
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].bar(POLICIES, t.exact_acc, yerr=[t.exact_acc - t.exact_acc_lo, t.exact_acc_hi - t.exact_acc], capsize=4, color=['grey', 'tab:blue', 'tab:orange'])
ax[0].set_title('GSM8K exact accuracy (95% Wilson)'); ax[0].set_ylim(0, 1)
w = t.loc[['rlvr', 'rlaif']]
ax[1].bar(['rlvr', 'rlaif'], w.win_vs_sft, yerr=[(w.win_vs_sft - w.win_vs_sft_lo).clip(lower=0), (w.win_vs_sft_hi - w.win_vs_sft).clip(lower=0)], capsize=4, color=['tab:blue', 'tab:orange'])
ax[1].axhline(0.5, ls='--', c='k', lw=1); ax[1].set_ylim(0, 1); ax[1].set_title('judge win rate vs SFT (0.5 = parity)')
gg = pd.concat([pd.read_json(f'{R}/gen_gsm_{p}.jsonl', lines=True) for p in POLICIES])
for p, c in zip(POLICIES, ['grey', 'tab:blue', 'tab:orange']):
    ax[2].hist(gg[gg.policy == p].response_tokens, bins=30, alpha=.5, label=p, color=c)
ax[2].axvline(cfg['math_max_new_tokens'], c='r', ls=':', lw=1); ax[2].legend(); ax[2].set_title('response length (tokens)')
plt.tight_layout(); plt.savefig(f'{R}/gsm_comparison.png', dpi=130); plt.show()


# ---- notebook cell 25 ----
Dg = out['diag']
pt = Dg['pair_table']
gt = pt[pt.has_ground_truth][['family', 'n', 'verifier_better', 'verifier_tie', 'verifier_wrong', 'judge_better', 'judge_tie', 'judge_wrong']]
print('== Controlled pairs WITH a diagnostically better response (better / tie / wrong preference) =='); display(gt.round(3))
ng = pt[~pt.has_ground_truth][['family', 'n', 'verifier_prefers_first', 'verifier_tie', 'verifier_prefers_second', 'judge_prefers_first', 'judge_tie', 'judge_prefers_second']]
print('== Pairs WITHOUT ground truth: which side each mechanism favours =='); display(ng.round(3))
print('== Sensitivities =='); display(Dg['sensitivity'].round(3))
print('== Per perturbation category: verifier reward, RLAIF K=5 group reward, length =='); display(Dg['variant_table'].round(3))
print('Judge position diagnostic on the diagnostic set:', Dg['position'])

main4 = ['clean>reason_corrupt', 'clean>wrong_final', 'clean>gold_distractor', 'clean>filler']
fig, ax = plt.subplots(1, 3, figsize=(18, 4.3))
for a, mech in zip(ax[:2], ['verifier', 'judge']):
    pt.loc[main4, [f'{mech}_better', f'{mech}_tie', f'{mech}_wrong']].set_axis(['better', 'tie', 'wrong'], axis=1).plot.barh(
        stacked=True, ax=a, color=['tab:green', 'lightgrey', 'tab:red'], legend=(mech == 'judge'))
    a.set_title(f'{mech}: controlled pairs'); a.set_xlim(0, 1); a.invert_yaxis()
vt = Dg['variant_table']
x = np.arange(len(vt)); ax[2].bar(x - .2, vt.verifier_reward, .4, label='verifier reward'); ax[2].bar(x + .2, vt.rlaif_group_reward, .4, label='RLAIF K=5 group reward')
ax[2].set_xticks(x); ax[2].set_xticklabels(vt.index, rotation=20); ax[2].legend(); ax[2].set_title('reward by perturbation category'); ax[2].set_ylim(0, 1.05)
plt.tight_layout(); plt.savefig(f'{R}/diagnostic_sensitivity.png', dpi=130); plt.show()

v = Dg['variants']
print('Within-problem correlation of RLAIF group reward with length (words):',
      round(v.groupby('problem_id').apply(lambda d: d.rlaif_group_reward.corr(d.response_words)).mean(), 3),
      '| with verifier reward:', round(v.groupby('problem_id').apply(lambda d: d.rlaif_group_reward.corr(d.verifier_reward)).mean(), 3))


# ---- notebook cell 27 ----
T = out['transfer']
print('== SVAMP-100: policy comparison =='); display(T['policy'][cols].round(3))
print('Head-to-head (A = RLVR, B = RLAIF):', T['h2h'])
print('\n== Drop from in-domain (GSM8K → SVAMP) =='); display(out['drop'].round(3))
print('== Paired McNemar on SVAMP accuracy =='); display(T['tests'].round(4))
print('== Verifier vs judge agreement on SVAMP =='); display(T['agreement'].round(3))
print('== Failure types (share of responses) ==')
display(pd.concat({'GSM8K': G['failures'], 'SVAMP': T['failures']}).round(3))
st = pd.read_json(f'{R}/gen_transfer_sft.jsonl', lines=True)
if 'svamp_type' in st:
    sv = pd.concat([pd.read_json(f'{R}/gen_transfer_{p}.jsonl', lines=True) for p in POLICIES])
    print('== SVAMP accuracy by problem type =='); display(sv.pivot_table(index='svamp_type', columns='policy', values='exact', aggfunc=['mean', 'size']).round(3))

fig, ax = plt.subplots(1, 3, figsize=(17, 4))
dr = out['drop']; x = np.arange(3)
ax[0].bar(x - .2, dr.acc_gsm, .4, label='GSM8K'); ax[0].bar(x + .2, dr.acc_svamp, .4, label='SVAMP-100')
ax[0].set_xticks(x); ax[0].set_xticklabels(POLICIES); ax[0].legend(); ax[0].set_ylim(0, 1); ax[0].set_title('exact accuracy in- vs out-of-domain')
ax[1].bar(x - .2, dr.win_vs_sft_gsm, .4, label='GSM8K'); ax[1].bar(x + .2, dr.win_vs_sft_svamp, .4, label='SVAMP-100')
ax[1].axhline(.5, ls='--', c='k', lw=1); ax[1].set_xticks(x); ax[1].set_xticklabels(POLICIES); ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].set_title('judge win rate vs SFT')
ft = pd.concat({'GSM8K': G['failures'], 'SVAMP': T['failures']}).drop(columns='correct')
ft.index = [f'{d}/{p}' for d, p in ft.index]; ft.plot.bar(stacked=True, ax=ax[2]); ax[2].set_title('failure types (share of all responses)'); ax[2].tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.savefig(f'{R}/transfer_comparison.png', dpi=130); plt.show()


# ---- notebook cell 29 ----
import textwrap
P, V = Dg['pairs'], Dg['variants'].set_index(['problem_id', 'variant'])
def show_pair(row, n=700):
    pid = row.problem_id
    prob, gold = V.loc[(pid, row['first'])].problem, V.loc[(pid, row['first'])].gold_final
    print('=' * 110)
    print(f"problem {pid} | pair {row.pair} | gold {gold} | verifier: {row.verifier_first:.0f} vs {row.verifier_second:.0f} -> {row.verifier_pref} ({row.verifier_outcome})"
          f" | judge -> {row.judge_pref} ({row.judge_outcome})")
    print('PROBLEM:', textwrap.shorten(prob, 350))
    for side in ('first', 'second'):
        r = V.loc[(pid, row[side])]
        print(f"\n  [{side}: {row[side]} | final={r.pred_final} | {r.response_words} words]\n  " + r.response[:n].replace('\n', '\n  '))
def pick(pair, prefer=('wrong', 'tie', 'better'), k=1):
    d = P[P.pair == pair]
    for o in prefer:
        s = d[d.judge_outcome == o]
        if len(s): return s.head(k)
    return d.head(k)
print('########## REASONING-ONLY CORRUPTION (final held correct) — one the judge catches, one it misses (if both exist) ##########')
d = P[P.pair == 'clean>reason_corrupt']
for sub in (d[d.judge_outcome == 'better'], d[d.judge_outcome != 'better']):
    for _, r in sub.head(1).iterrows(): show_pair(r)
print('\n########## PERSUASIVE FILLER (same correct answer + irrelevant persuasive text) ##########')
for _, r in pick('clean>filler', ('wrong', 'tie', 'better')).iterrows(): show_pair(r)
print('\n########## WRONG FINAL / GOLD DISTRACTOR ##########')
for _, r in pick('clean>gold_distractor', ('wrong', 'tie', 'better')).iterrows(): show_pair(r)
for _, r in pick('clean>wrong_final', ('wrong', 'tie', 'better')).iterrows(): show_pair(r)
print('\n########## PROCESS vs OUTCOME (corrupt reasoning + right answer  vs  good reasoning + wrong answer) ##########')
for _, r in P[P.pair == 'reason_corrupt vs wrong_final'].head(1).iterrows(): show_pair(r)


# ---- notebook cell 30 ----
# In-domain verifier–judge disagreements on real policy outputs
J = pd.read_json(f'{R}/judge_gsm.jsonl', lines=True, dtype={'source_index': str})
gens = {p: pd.read_json(f'{R}/gen_gsm_{p}.jsonl', lines=True, dtype={'source_index': str}).set_index('source_index') for p in POLICIES}
def show_real(r, n=600):
    a, b = gens[r.a_policy].loc[r.source_index], gens[r.b_policy].loc[r.source_index]
    print('=' * 110); print(f"{r.source_index} | {r.a_policy} vs {r.b_policy} | gold {a.gold_final} | verifier {r.verifier_pref} | judge {r.judge_pref}")
    print('PROBLEM:', textwrap.shorten(a.question, 300))
    for name, x in ((r.a_policy, a), (r.b_policy, b)):
        print(f"\n  [{name} | final={x.get('pred_final')} | exact={x.exact:.0f} | {x.response_tokens} tok]\n  " + x.response[:n].replace('\n', '\n  '))
cases = [('judge prefers the INCORRECT response', J[(J.verifier_pref != 'TIE') & (J.judge_pref != 'TIE') & (J.judge_pref != J.verifier_pref)]),
         ('both wrong (verifier ties) but judge ranks them', J[(J.verifier_pref == 'TIE') & (J.a_exact == 0) & (J.judge_pref != 'TIE')]),
         ('both correct (verifier ties) but judge ranks them', J[(J.verifier_pref == 'TIE') & (J.a_exact == 1) & (J.judge_pref != 'TIE')])]
for title, d in cases:
    print(f'\n########## {title}: {len(d)} cases ##########')
    for _, r in d.head(1).iterrows(): show_real(r)


# ---- notebook cell 32 ----
C = out['cost']
print(json.dumps(C, indent=1, default=float))
print(f"\nOne exact-verifier check ≈ {C['verifier_us_per_check']:.1f} µs; one pairwise judge call ≈ {C['judge_s_per_pairwise_call']:.2f} s "
      f"(≈{C['judge_over_verifier_ratio']:.0f}× slower). A direct-RLAIF group of K responses needs K(K−1)/2 judge calls "
      f"(K=8 → 28 calls ≈ {C['est_judge_s_per_group(K=8)']:.0f} s) vs K verifier checks.")
print('Judge tie / position diagnostics — GSM8K:', G['position'], '| SVAMP:', T['position'], '| diagnostics:', Dg['position'])
