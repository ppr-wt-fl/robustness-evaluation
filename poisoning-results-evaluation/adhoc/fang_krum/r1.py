import json,glob,numpy as np
R='/home/felipetrost/projects/uniprojects/masterthesis-code/robustness-evaluation/poisoning-eval/results/'
for f in sorted(glob.glob(R+'20261023-14*shard*of12.jsonl')):
  for l in open(f):
    r=json.loads(l)
    if r['aggregator']!='EPPRFL' or r['config']['n_rounds']!=50 or r['attack'] not in ('None','FangKrum'): continue
    x=r['round_results'][1]; d=np.array(x['epprfl_distances'])[80:]
    n=np.array(x.get('client_norms') or [np.nan]*200)[80:]
    print(r['config']['partition_scheme'],r['attack'],r['aggregator_params']['filter_lambda'],r['n_attackers'],'r1 d[80:5] ',d[:5].round(4),'norms',n[:3].round(4),'CV %.2f'%(d.std()/d.mean()))
