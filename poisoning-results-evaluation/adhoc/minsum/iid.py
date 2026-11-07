import sys; sys.path.insert(0,'.')
from load import load
import numpy as np
r=load('data/twosplit-lambda0.005'); lam=0.005
s=r[(r.name=='MinSum')]
for _,x in s[(s.aggregator=='EPPRFL')&(s.partition.str.lower()=='iid')].sort_values('n_malicious').iterrows():
  nm=x.n_malicious; q=[];pct=[];acc=0;n=0;below_hon=[]
  for rd in x.rounds:
    if 'epprfl_distances' not in rd: continue
    d=np.array(rd['epprfl_distances'])/rd['epprfl_m_h']; h=d[nm:]
    q.append(d[0]); pct.append((h<d[0]).mean()*100); acc+=rd['n_malicious_accepted']>0; n+=1
  print(f"nm={nm} drop={x.drop_f1:.2f} q med={np.median(q):.2f} honest-closer-than-attacker med={np.median(pct):.0f}% acc={acc}/{n} max q accepted<1: {max(q)<1}")
