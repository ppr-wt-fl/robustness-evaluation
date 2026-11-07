# round-0 logit margin of the two attacked targets vs. all ad/tracker requests
# run: ../poisoning-eval/.venv/bin/python adhoc/backdoors/margin.py (loads the cached model / splits of poisoning-eval)
import os, sys
os.chdir(os.path.expanduser('~/projects/uniprojects/masterthesis-code/robustness-evaluation/poisoning-eval'))
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import setup as s
model = s.model.eval()
def logits(x):
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(x, np.float32)).to(s.device))[:, 0].cpu().numpy()
test_all = logits(s.test_features); train_all = logits(s.train_features)
ad_test = test_all[s.test_labels == 1]
for host in ['stats.g.doubleclick.net', 'ad-delivery.net']:
    e = np.load(f'edge_sets/{host}_hidden.npz')
    held = test_all[e['test_rows']]; tr = train_all[e['exclude_rows']]
    print(host, 'target label', e['target_label'].item())
    for name, v in [('held-out', held), ('train', tr)]:
        q = np.percentile(v, [0, 5, 25, 50, 75])
        print(f'  {name:8s} n={len(v):4d} logit min/p5/p25/median/p75 = ' + ' / '.join(f'{x:.2f}' for x in q),
              f' prob median {1/(1+np.exp(-np.median(v))):.3f}', f' share logit<2: {(v<2).mean()*100:.1f} %')
        print(f'           median logit is above {(ad_test < np.median(v)).mean()*100:.0f} % of all ad/tracker test logits')
