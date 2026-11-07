import json,glob,numpy as np
R='/home/felipetrost/projects/uniprojects/masterthesis-code/robustness-evaluation/poisoning-eval/results/'
T={}
for f in sorted(glob.glob(R+'20261023-14*shard*of12.jsonl')):
  for l in open(f):
    r=json.loads(l)
    if r['config']['partition_scheme']!='dirichlet_1.0' or r['config']['n_rounds']!=50: continue
    k=(r['attack'],r['aggregator'],r['aggregator_params'].get('filter_lambda'),r['n_attackers'])
    if k[0] in('None','FangKrum') and k[3] in (0,40,None) or (k[0]=='None'): T[k]=r
for k in T: print(k)
c=T[('None','EPPRFL',0.005,0)] if ('None','EPPRFL',0.005,0) in T else [v for k,v in T.items() if k[0]=='None' and k[2]==0.005][0]
a=T[('FangKrum','EPPRFL',0.005,40)]; fa=T[('FangKrum','FedAvg',None,40)]; fc=[v for k,v in T.items() if k[0]=='None' and k[1]=='FedAvg'][0]
print('rnd cleanF&C(f1,benrej,honCV) | atkF&C f1 | FedAvg40 f1 | FedAvg clean f1 | atk_round flag')
for i in range(1,51):
  x=c['round_results'][i]; d=np.array(x['epprfl_distances'])
  print(i,'%.4f %3d %.2f | %.4f | %.4f | %.4f | %s'%(x['f1'],x['n_benign_rejected'],d.std()/d.mean(),a['round_results'][i]['f1'],fa['round_results'][i]['f1'],fc['round_results'][i]['f1'],a['round_results'][i].get('attack_round')))
print('test', c['test']['f1'],a['test']['f1'],fa['test']['f1'],fc['test']['f1'])
