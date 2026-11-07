import sys, numpy as np
sys.path.insert(0,'.')
from load import load
r=load('data/twosplit-lambda0.005')
L=0.005
for part in ['iid','natural','dirichlet_1.0']:
  for nm in [10,20,40,60,80]:
    x=r[(r.aggregator=='EPPRFL')&(r.name=='MinMax')&(r.partition==part)&(r.n_malicious==nm)].iloc[0]
    print(f'== {part} n={nm} drop={x.drop_f1:.2f} recall={x.detection_recall}')
    prev=None
    for q in x.rounds:
      if 'epprfl_distances' not in q: prev=q['f1']; continue
      d=np.array(q['epprfl_distances']); mh=q['epprfl_m_h']; w=np.sqrt(L/mh)
      da=d[0]/mh; ben=d[nm:]/mh
      acc=set(q['accepted_clients']); passed=0 in acc
      clip=min(1,1/da)
      cn=q['client_norms']; 
      print(f"r{q['round']:2d} mh={mh:.4f} band=[{1-w:.3f},{1+w:.3f}] att={da:.3f} {'PASS' if passed else 'rej '} ben_med={np.median(ben):.3f} ben_q10/90={np.quantile(ben,.1):.2f}/{np.quantile(ben,.9):.2f} nben_acc={len(acc)-(nm if passed else 0)} clip={clip:.2f} lam={q.get('lamda',0):.3f} attnorm={cn[0]:.3f} bennorm={q['benign_norm_mean']:.3f} cos={q['client_cosines_to_benign_mean'][0]:.2f} dF1={(q['f1']-prev)*100:+.2f}")
      prev=q['f1']
