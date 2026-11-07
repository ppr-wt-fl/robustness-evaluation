import sys, numpy as np
sys.path.insert(0,'.')
from load import load
r=load('data/twosplit-lambda0.005')
for part in ['iid','natural','dirichlet_1.0']:
  for nm in [10,20,40,60,80]:
    x=r[(r.aggregator=='EPPRFL')&(r.name=='MinMax')&(r.partition==part)&(r.n_malicious==nm)].iloc[0]
    R=[q for q in x.rounds if 'epprfl_distances' in q]
    da=np.array([q['epprfl_distances'][0]/q['epprfl_m_h'] for q in R])
    P=np.array([0 in set(q['accepted_clients']) for q in R])
    mn=np.array([q['client_norms'][0] for q in R]); m=np.array([q['lamda'] for q in R])-mn
    agg=np.array([q['aggregate_norm'] for q in R])
    pp=P[1:][P[:-1]].mean() if P[:-1].any() else np.nan; pr=P[1:][~P[:-1]].mean() if (~P[:-1]).any() else np.nan
    # attacker weight in aggregate when passed: n*1 / (n + nb) (clip~1)
    nacc=np.array([len(q['accepted_clients']) for q in R])
    share=(nm/nacc[P]).mean() if P.any() else np.nan
    print(f"{part:13s} n={nm:2d} P(pass|prev pass)={pp:.2f} P(pass|prev rej)={pr:.2f} corr(d/mh,malnorm)={np.corrcoef(da,mn)[0,1]:+.2f} |m| after pass={m[1:][P[:-1]].mean() if P[:-1].any() else np.nan:.3f} after rej={m[1:][~P[:-1]].mean():.3f} att share of accepted when pass={share:.2f} aggnorm pass={agg[P].mean() if P.any() else np.nan:.3f} rej={agg[~P].mean():.3f}")
