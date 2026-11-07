# reproduce Fang vs EPPRFL on a partition, dump per-round Krum diagnostics of the attackers' honest updates
import os, sys, json, time
import numpy as np
os.chdir(os.path.expanduser('~/projects/uniprojects/masterthesis-code/robustness-evaluation/poisoning-eval'))
sys.path.insert(0, os.getcwd())
from setup import *
src = open('experiment.py').read().splitlines()
exec(compile('\n'.join(src[:192]), 'experiment.py', 'exec'), globals())
from epprfl import EPPRFL
from attacks.fangLocalModelPoisoning2020 import FangKrumAttack, krum_lambda_upper_bound, krum_index_crafted, pairwise_distances, estimated_direction

PART, C, NR = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
OUT = sys.argv[4]
parts = CLIENT_PARTITIONS[PART]
reseed()
agg = EPPRFL(random_generator, filter_lambda=0.005)
att = FangKrumAttack(C, random_generator)

def diag(own_updates):
    d = own_updates.shape[1]; sd = np.sqrt(d)
    s = estimated_direction(own_updates)
    own = own_updates.astype(np.float64)
    D = pairwise_distances(own); dots = own @ s; sq = (own**2).sum(1); norms = np.sqrt(sq)
    B = krum_lambda_upper_bound(D, norms.max(), 1, d)
    def ok(l):
        cd = np.sqrt(np.maximum(l*l*d + 2*l*dots + sq, 0))
        return krum_index_crafted(D, cd, 1) < 1, cd
    # threshold lamda* by bisection on (1e-6, B) (pass is monotone below threshold here)
    lo, hi = 1e-7, B
    assert ok(lo)[0] or True
    for _ in range(60):
        mid = (lo+hi)/2
        if ok(mid)[0]: lo = mid
        else: hi = mid
    lstar = lo
    sortedD = np.sort(D, 1)[:, 1:C-1]
    own_scores = sortedD.sum(1)
    cd0 = norms
    return dict(B_sd=B*sd, maxnorm=norms.max(), meannorm=norms.mean(), minscore_mean=own_scores.min()/(C-2),
                medscore_mean=np.median(own_scores)/(C-2), lstar_sd=lstar*sd, ok_at_0=bool(ok(1e-9)[0]),
                gamma_mean=float((dots/(sd*norms)).mean()), crafted_score_at0=np.sort(cd0)[:C-2].sum()/(C-2),
                pair_mean=float(D[np.triu_indices(C,1)].mean()), k=float(np.log2(B/lstar)))

g = initial_weights
for rnd in range(1, NR+1):
    t = time.time()
    ups = train_all_clients(g, parts)
    info = diag(ups[:C])
    ups2, mal = att.perform_attack(ups, g)
    agg_up, acc = agg.aggregate(ups2, g)
    g = g + agg_up
    info.update(round=rnd, lamda_sd=att.last_info['lamda']*np.sqrt(ups.shape[1]), passed=bool(0 in set(acc.tolist())),
                att_ratio=agg.last_info['epprfl_distances'][0]/agg.last_info['epprfl_m_h'], secs=time.time()-t)
    print(json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in info.items()}), flush=True)
    with open(OUT, 'a') as f: f.write(json.dumps(info)+'\n')
