import json,glob,numpy as np,collections
R='/home/felipetrost/projects/uniprojects/masterthesis-code/robustness-evaluation/poisoning-eval/results/'
for f in sorted(glob.glob(R+'20261023-14*shard*of12.jsonl')):
  for l in open(f):
    r=json.loads(l)
    if r['attack']!='FangKrum' or r['aggregator']!='EPPRFL' or r['aggregator_params']['filter_lambda']!=0.005: continue
    N=r['n_attackers']; rr=r['round_results'][1:]
    lam=np.array([x['lamda'] for x in rr]); nc=collections.Counter(x['num_crafted'] for x in rr)
    dm=np.array([np.mean(x['epprfl_distances'][:N]) for x in rr]); mh=np.array([x['epprfl_m_h'] for x in rr])
    sp=np.array([np.ptp(x['epprfl_distances'][:N]) for x in rr])
    norm=np.array([x['malicious_norm_mean'] for x in rr]); D=(norm/lam)**2
    hw=np.sqrt(0.005*mh)
    R_=0.06*D.mean(); fit=lam**2*R_
    lo=np.sqrt((mh-hw)/R_); hi=np.sqrt((mh+hw)/R_); mid=np.sqrt(mh/R_)
    print(r['config']['partition_scheme'],N,'d=%.3g'%D.mean(),'nc',dict(nc),'lamda set',sorted(set(np.round(lam,6))) if len(set(np.round(lam,5)))<5 else '%.2e..%.2e'%(lam.min(),lam.max()),
      '| dmal/(lam^2|R|) %.2f'%np.mean(dm/fit),'| copies spread %.1e'%sp.max(),'| band in lamda [%.2e,%.2e] mid %.2e'%(lo.mean(),hi.mean(),mid.mean()),'| lamda/lo %.2f..%.2f'%((lam/lo).min(),(lam/lo).max()))
