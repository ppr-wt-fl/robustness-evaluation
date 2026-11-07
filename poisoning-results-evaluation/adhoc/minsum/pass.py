import sys; sys.path.insert(0,'.')
from load import load
import numpy as np
r=load('data/twosplit-lambda0.005'); lam=0.005
s=r[(r.aggregator=='EPPRFL')&(r.name=='MinSum')]
for p in ['iid','natural','dirichlet']:
  acc=[];rej=[];edge=[];ben=[]
  for _,x in s[s.partition.str.lower().str.startswith(p[:3])].iterrows():
    nm=x.n_malicious
    for rd in x.rounds:
      if 'epprfl_distances' not in rd: continue
      q=np.array(rd['epprfl_distances'])/rd['epprfl_m_h']; lo=1-np.sqrt(lam/rd['epprfl_m_h'])
      (acc if rd['n_malicious_accepted']>0 else rej).append((q[0],lo,nm))
      edge.append(lo); ben.append(np.median(q[nm:]))
  f=lambda a:f"n={len(a)} q med={np.median([t[0] for t in a]):.3f} min={min(t[0] for t in a):.3f} max={max(t[0] for t in a):.3f} q-edge med={np.median([t[0]-t[1] for t in a]):+.3f}" if a else "n=0"
  print(p,'edge med %.3f'%np.median(edge),'| ACCEPTED',f(acc),'| shares',sorted({t[2] for t in acc}),'| REJECTED',f(rej))
