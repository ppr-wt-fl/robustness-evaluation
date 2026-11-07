import numpy as np
from load import load; r=load('data/twosplit-lambda0.005')
s=r[(r.name=='MinMax-std')&(r.aggregator=='EPPRFL')]
x=s.iloc[0]; print(x.rounds[1].keys()); print(len(x.rounds))
lam=0.005
for _,x in s.sort_values(['partition','n_malicious']).iterrows():
    nm=x.n_malicious; print('==',x.partition,nm,'drop',round(x.drop_f1,2),'det',x.detection_recall)
    for t in x.rounds:
        d=t.get('epprfl_distances'); mh=t.get('epprfl_m_h')
        if d is None: continue
        d=np.array(d); band=np.sqrt(lam/mh)
        a=d[0]/mh; h=d[nm:]/mh
        if t['round'] in (1,2,3,5,10,20,30,40,49,50) :
            print(f"r{t['round']:2d} mh={mh:.4g} band=±{band:.3f} att d/mh={a:.3f} honest d/mh med={np.median(h):.3f} p10={np.percentile(h,10):.3f} p90={np.percentile(h,90):.3f} rejB={t['n_benign_rejected']} rejM={t['n_malicious_rejected']} lamda={t.get('lamda')} f1={t['f1']:.3f}")
