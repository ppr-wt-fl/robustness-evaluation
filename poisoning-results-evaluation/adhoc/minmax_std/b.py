import numpy as np
from load import load; r=load('data/twosplit-lambda0.005')
s=r[(r.name=='MinMax-std')]
for _,x in s.sort_values(['aggregator','partition','n_malicious']).iterrows():
    nm=x.n_malicious
    cs=np.array(x.client_sizes); pr=np.array(x.client_positive_rates)
    out=[]
    for k in (1,10,25,50):
        t=x.rounds[k]
        n=np.array(t['client_norms']); c=np.array(t['client_cosines_to_benign_mean'])
        out.append(f"r{k}: lam={t['lamda']:.2f} attN/honN_med={n[0]/np.median(n[nm:]):.2f} honN max/med={n[nm:].max()/np.median(n[nm:]):.2f} hon cos med={np.median(c[nm:]):.2f} att cos={c[0]:.2f} f1={t['f1']:.3f} rec={t['recall']:.2f} prec={t['precision']:.2f}")
    print('==',x.aggregator,x.partition,nm,'drop',round(x.drop_f1,1),'| sizes cv',round(cs.std()/cs.mean(),2),'pos rate sd',round(pr.std(),3), 'att pos mean',round(pr[:nm].mean(),3),'all',round(pr.mean(),3))
    for o in out: print('  ',o)
