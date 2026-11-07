import json,glob,numpy as np
R='/home/felipetrost/projects/uniprojects/masterthesis-code/robustness-evaluation/poisoning-eval/results/'
runs=[]
for f in sorted(glob.glob(R+'20261023-14*shard*of12.jsonl')):
  for l in open(f):
    r=json.loads(l)
    if r['aggregator']=='EPPRFL' and r['config']['n_rounds']==50 and r['attack'] in ('None','FangKrum'): runs.append(r)
print('== accepted rounds: first, F1 delta, attacker weight, max lambda that rejects')
for r in runs:
  if r['attack']!='FangKrum' or r['config']['partition_scheme']!='dirichlet_1.0': continue
  N=r['n_attackers']; lam=r['aggregator_params']['filter_lambda']; rr=r['round_results']
  acc=[i for i in range(1,51) if rr[i]['n_malicious_accepted']>0]
  if not acc: print(N,lam,'none'); continue
  dacc=[rr[i]['f1']-rr[i-1]['f1'] for i in acc]; dno=[rr[i]['f1']-rr[i-1]['f1'] for i in range(2,51) if i not in acc]
  lstar=[rr[i]['epprfl_m_h']*(1-np.mean(rr[i]['epprfl_distances'][:N])/rr[i]['epprfl_m_h'])**2 for i in acc]
  w=[N/len(rr[i]['accepted_clients']) for i in acc]
  print('N=%d lam=%.3f first %d n %d | dF1 acc %+.2fpp vs other %+.2fpp | mal share of aggregate %.2f | lambda to reject all: < %.4f'%(N,lam,acc[0],len(acc),100*np.mean(dacc),100*np.mean(dno),np.mean(w),min(lstar)))
print('== first-order honest rejection (%) under alternative lambda, from logged distances & m_h (no feedback)')
for r in runs:
  N=r['n_attackers']; lam=r['aggregator_params']['filter_lambda']
  if lam!=0.005 or (r['attack']=='FangKrum' and N!=40): continue
  rr=r['round_results'][1:]
  out=[]
  for L in [0.002,0.003,0.0035,0.004,0.005,0.006,0.01]:
    rej=np.mean([np.mean((np.array(x['epprfl_distances'][N:])-x['epprfl_m_h'])**2>=L*x['epprfl_m_h']) for x in rr])
    macc=np.mean([np.any((np.array(x['epprfl_distances'][:N])-x['epprfl_m_h'])**2<L*x['epprfl_m_h']) for x in rr]) if N else 0
    out.append('%.4f:%4.1f%%/%2.0f%%'%(L,100*rej,100*macc))
  print(r['config']['partition_scheme'],r['attack'],N,' '.join(out))
