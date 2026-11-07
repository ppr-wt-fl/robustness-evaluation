import sys, numpy as np
sys.path.insert(0,'.')
from load import load
r=load('data/twosplit-lambda0.005')
L=0.005
for part in ['iid','natural','dirichlet_1.0']:
  for nm in [10,20,40,60,80]:
    x=r[(r.aggregator=='EPPRFL')&(r.name=='MinMax')&(r.partition==part)&(r.n_malicious==nm)].iloc[0]
    rows=[];prev=None
    for q in x.rounds:
      if 'epprfl_distances' not in q: prev=q['f1']; continue
      d=np.array(q['epprfl_distances']); mh=q['epprfl_m_h']; w=np.sqrt(L/mh)
      da=d[0]/mh; acc=set(q['accepted_clients']); p=0 in acc
      assert all((i in acc)==p for i in range(nm))
      below=da<1-w; above=da>1+w
      nb=len(acc)-(nm if p else 0)
      rows.append((q['round'],da,1-w,p,below,above,q['lamda'],q['client_norms'][0],q['lamda']-q['client_norms'][0],(q['f1']-prev)*100,nb,min(1,1/da), np.median(d[nm:])/mh, q['client_cosines_to_benign_mean'][0]))
      prev=q['f1']
    a=np.array(rows,dtype=float); P=a[:,3]==1
    late=a[:,0]>10
    print(f"{part:13s} n={nm:2d} pass={P.mean()*100:3.0f}% rej_below={a[:,4].sum():.0f} rej_above={a[:,5].sum():.0f} att/mh med={np.median(a[:,1]):.2f} [q10 {np.quantile(a[:,1],.1):.2f} q90 {np.quantile(a[:,1],.9):.2f}] lower={a[:,2].mean():.3f} lam={a[:,6].mean():.3f} malnorm={a[:,7].mean():.3f} |m|={a[:,8].mean():.3f} cos={a[:,13].mean():.2f} minclip={a[P,11].min() if P.any() else float('nan'):.2f} nbenacc={a[:,10].mean():.0f} dF1 pass={a[P,9].mean() if P.any() else np.nan:+.2f} rej={a[~P,9].mean() if (~P).any() else np.nan:+.2f} | late(r>10) pass={P[late].mean()*100:.0f}% dF1p={a[P&late,9].mean() if (P&late).any() else np.nan:+.2f} dF1r={a[~P&late,9].mean() if (~P&late).any() else np.nan:+.2f}")
