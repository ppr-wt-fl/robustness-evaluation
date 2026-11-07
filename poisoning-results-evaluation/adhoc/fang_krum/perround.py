import json,glob,numpy as np,sys
R='/home/felipetrost/projects/uniprojects/masterthesis-code/robustness-evaluation/poisoning-eval/results/'
P,LAM,N=sys.argv[1],float(sys.argv[2]),int(sys.argv[3])
for f in sorted(glob.glob(R+'20261023-14*shard*of12.jsonl')):
  for l in open(f):
    r=json.loads(l)
    if r['attack']=='FangKrum' and r['aggregator']=='EPPRFL' and r['config']['partition_scheme']==P and r['aggregator_params']['filter_lambda']==LAM and r['n_attackers']==N:
      print('rnd  f1    m_h     dmal/m_h  lo/hi edge   macc benrej clipped aw   lamda   honCV  hon<lo hon>hi')
      for x in r['round_results'][1:]:
        d=np.array(x['epprfl_distances']); m=x['epprfl_m_h']; hw=np.sqrt(LAM*m)
        hon=d[N:]
        print('%3d %.4f %.4f %.3f  %.3f/%.3f  %3d  %3d  %3s  %.3f %s %.2f %3d %3d'%(x['round'],x['f1'],m,d[:N].mean()/m,1-hw/m,1+hw/m,x['n_malicious_accepted'],x['n_benign_rejected'],x.get('epprfl_n_clipped'),x.get('attacker_weight') or 0,('%.2e'%x['lamda']) if x.get('lamda') else '-',hon.std()/hon.mean(),(hon<m-hw).sum(),(hon>m+hw).sum()))
