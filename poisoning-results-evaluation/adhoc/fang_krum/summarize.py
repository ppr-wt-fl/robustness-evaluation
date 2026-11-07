import json,glob,numpy as np,sys
R='/home/felipetrost/projects/uniprojects/masterthesis-code/robustness-evaluation/poisoning-eval/results/'
runs=[]
for f in sorted(glob.glob(R+'20261023-14*shard*of12.jsonl')):
  for l in open(f):
    r=json.loads(l)
    if r['attack'] in ('FangKrum','None'): runs.append(r)
def part(r):
  c=r['config']; return c['partition_scheme']+('' if c['dirichlet_alpha'] is None else str(c['dirichlet_alpha']))
base={}
for r in runs:
  if r['attack']=='None' and r['config']['n_rounds']==50:
    base[(r['aggregator'],r['aggregator_params'].get('filter_lambda'),part(r))]=r['test']['f1']
print('baselines',base)
print('part lam nmal | F1drop | atk rounds acc | mean d_mal/m_h | band rel half-width | mean benign rej | ben rej (atk-accepted rounds) | mean lamda | med hon d/m_h | hon d CV')
for r in sorted(runs,key=lambda r:(part(r),str(r['aggregator_params'].get('filter_lambda')),r['n_attackers'])):
  if r['attack']!='FangKrum': continue
  lam=r['aggregator_params'].get('filter_lambda'); p=part(r); n=r['n_attackers']
  fed=base.get(('FedAvg',None,p)); f1=r['test']['f1']
  b=base.get((r['aggregator'],lam,p))
  if r['aggregator']!='EPPRFL':
    print(p,'FedAvg',n,'F1',round(f1*100,2),'drop vs fedavg clean',round((fed-f1)*100,2)); continue
  rows=[x for x in r['round_results'] if x.get('attack_round')]
  acc=[x['n_malicious_accepted'] for x in rows]
  ratio=[np.mean(x['epprfl_distances'][:n])/x['epprfl_m_h'] for x in rows]
  hw=[np.sqrt(lam*x['epprfl_m_h'])/x['epprfl_m_h'] for x in rows]
  hon=[np.median(x['epprfl_distances'][n:])/x['epprfl_m_h'] for x in rows]
  cv=[np.std(x['epprfl_distances'][n:])/np.mean(x['epprfl_distances'][n:]) for x in rows]
  br=[x['n_benign_rejected'] for x in rows]
  bra=[x['n_benign_rejected'] for x in rows if x['n_malicious_accepted']>0]
  print(p,lam,n,'| drop',round((b-f1)*100,2),'| acc rounds',sum(a>0 for a in acc),'/',len(rows),'| d/m_h %.3f'%np.mean(ratio),'| hw %.3f'%np.mean(hw),'| benrej %.1f'%np.mean(br),'| benrej@acc %s'%(round(np.mean(bra),1) if bra else '-'),'| lamda %.2e'%np.mean([x['lamda'] for x in rows]),'| hon %.3f'%np.mean(hon),'| cv %.3f'%np.mean(cv))
