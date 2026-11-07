import json,glob,numpy as np
R='/home/felipetrost/projects/uniprojects/masterthesis-code/robustness-evaluation/poisoning-eval/results/'
T={}
for f in sorted(glob.glob(R+'20261023-14*shard*of12.jsonl')):
  for l in open(f):
    r=json.loads(l)
    if r['config']['partition_scheme']!='dirichlet_1.0' or r['config']['n_rounds']!=50 or r['aggregator']!='EPPRFL': continue
    T[(r['attack'],r['aggregator_params']['filter_lambda'],r['n_attackers'])]=r
def stats(r,N):
  acc=np.zeros(200)
  for x in r['round_results'][1:]: acc[x['accepted_clients']]+=1
  hon=acc[N:]/50
  rr=r['round_results']
  cv=np.mean([np.std(np.array(x['epprfl_distances'])[40:])/np.mean(np.array(x['epprfl_distances'])[40:]) for x in rr[1:]])
  hr=np.mean([sum(1 for i in range(40,200) if i not in set(x['accepted_clients'])) for x in rr[1:]])
  return 'clients40-199: mean acc rate %.3f, never acc %d, always acc %d, mean rej/round %.1f, CV(d,40:) %.2f, m_h r1 %.4f r50 %.4f'%(hon.mean(),(hon==0).sum(),(hon==1).sum(),hr,cv,rr[1]['epprfl_m_h'],rr[50]['epprfl_m_h'])
for k in sorted(T,key=str):
  if k[0] in ('None','FangKrum'): print(k, stats(T[k],40))
# client sizes
r=T[('None',0.005,0)]
s=np.array(r['config']['client_sizes']); print('sizes 0-39 mean %.0f, 40-199 mean %.0f, CV all %.2f'%(s[:40].mean(),s[40:].mean(),s.std()/s.mean()))
acc=np.zeros(200)
for x in r['round_results'][1:]: acc[x['accepted_clients']]+=1
print('clean: acc rate of 0-39: %.3f; corr(acc rate,size) %.2f'%(acc[:40].mean()/50,np.corrcoef(acc,s)[0,1]))
print('---')
p=np.array(r['config']['client_positive_rates'])
for a,b in [(0,10),(10,20),(20,40),(40,60),(60,80),(80,200)]: print('clients %d-%d pos rate mean %.3f, size-weighted pos %.3f, size sum %d'%(a,b,p[a:b].mean(),(p[a:b]*s[a:b]).sum()/s[a:b].sum(),s[a:b].sum()))
print('overall pos %.3f'%((p*s).sum()/s.sum()))
for k in sorted(T,key=str):
  if k[0] in ('None','FangKrum'):
    rr=T[k]['round_results']
    cv=[np.std(np.array(x['epprfl_distances'])[80:])/np.mean(np.array(x['epprfl_distances'])[80:]) for x in rr[1:]]
    rej=np.mean([sum(1 for i in range(80,200) if i not in set(x['accepted_clients'])) for x in rr[1:]])
    print(k,'CV(d,80:) r1-10 %.2f all %.2f, rej among 80-199/round %.1f'%(np.mean(cv[:10]),np.mean(cv),rej))
