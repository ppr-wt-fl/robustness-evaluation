import numpy as np
from load import load; r=load('data/twosplit-lambda0.005')
s=r[(r.name=='MinMax-std')&(r.aggregator=='FedAvg')&(r.n_malicious>=60)]
for _,x in s.iterrows():
    print(x.partition,x.n_malicious,' '.join(f"{t['round']}:{t['fpr']:.2f}/{t['recall']:.2f}" for t in x.rounds[1::5]))
# lower-edge rejections IID/natural F&C
lam=0.005
s=r[(r.name=='MinMax-std')&(r.aggregator=='EPPRFL')&(r.partition!='dirichlet_1.0')]
for _,x in s.iterrows():
    lo=hi=0
    for t in x.rounds[1:]:
        d=np.array(t['epprfl_distances']);mh=t['epprfl_m_h']
        if (d[0]-mh)**2>=lam*mh: lo+=d[0]<mh; hi+=d[0]>mh
    print(x.partition,x.n_malicious,'att rejected rounds below/above',lo,hi)
