"""Task 1 (DPO): summary tables.

Generated from the analysis cells of the Colab notebook so the tables, plots and qualitative
examples can be regenerated from the saved result files without the notebook.
Run from the repository root:  python -m task1_dpo.report_tables
"""
import matplotlib
matplotlib.use("Agg")


def display(x):
    print(x.to_string() if hasattr(x, "to_string") else x)


# ---- notebook cell 20 ----
import json,glob,pandas as pd
rows=[json.load(open(f)) for f in sorted(glob.glob('results/task1_dpo/eval_*.json'))]
cols=['name','n','n_skipped_overlong_prompt','dpo_loss','pref_acc','kl_token_mean','kl_seq_sum','reward_mean','len_mean','len_std','len_iqr','wordlimit_compliance']
display(pd.DataFrame(rows)[cols].round(4))
for f in sorted(glob.glob('results/task1_dpo/stratified_*.json')):
    r=json.load(open(f)); print(r['name'],'| stratum field:',r['stratum_key']); display(pd.DataFrame(r['strata']).T.round(4))
