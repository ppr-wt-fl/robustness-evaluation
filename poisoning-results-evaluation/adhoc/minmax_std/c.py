import numpy as np
from load import load; r=load('data/twosplit-lambda0.005')
lam=0.005
s=r[(r.name=='MinMax-std')]
for _,x in s.sort_values(['partition','n_malicious','aggregator']).iterrows():
    nm=x.n_malicious; sh=[];rejB=[];hf=[];aw=[]
    for t in x.rounds[1:]:
        aw.append(t.get('attacker_weight') or np.nan)
        if x.aggregator!='EPPRFL': continue
        d=np.array(t['epprfl_distances']); mh=t['epprfl_m_h']
        acc=np.abs(d-mh)**2<lam*mh*1.0
        acc=(d-mh)**2<lam*mh
        f=np.where(acc,np.minimum(1,mh/np.maximum(d,1e-30)),0)
        sh.append(f[:nm].sum()/f.sum()); rejB.append((~acc[nm:]).sum()); hf.append(f[nm:][acc[nm:]].mean())
    f1=[t['f1'] for t in x.rounds]
    print(x.partition,nm,x.aggregator,f"drop={x.drop_f1:.2f} attacker_weight_logged={np.nanmean(aw):.3f}",
          f"Fc_att_share={np.mean(sh):.3f} hon_rej/round={np.mean(rejB):.1f} hon_mean_factor={np.mean(hf):.3f}" if sh else '',
          f"f1 r10/25/50={f1[10]:.3f}/{f1[25]:.3f}/{f1[50]:.3f} rec50={x.rounds[50]['recall']:.2f} lamda_mean={np.mean([t['lamda'] for t in x.rounds[1:]]):.2f} malN/benN={np.mean([t['malicious_norm_mean']/t['benign_norm_mean'] for t in x.rounds[1:]]):.2f}")
