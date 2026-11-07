from load import load
import numpy as np, sys
r=load('data/twosplit-lambda0.005')
lam=0.005
for name in ['MinSum','MinMax']:
  s=r[(r.aggregator=='EPPRFL')&(r.name==name)]
  for _,x in s.sort_values(['partition','n_malicious']).iterrows():
    nm=x.n_malicious; rows=[]
    for rd in x.rounds:
        if 'epprfl_distances' not in rd: continue
        d=np.array(rd['epprfl_distances']); mh=rd['epprfl_m_h']
        q=d/mh; w=np.sqrt(lam/mh)
        acc=rd['n_malicious_accepted']>0
        ben=q[nm:]
        rows.append((rd['round'],q[0],w,acc,rd['lamda'],np.median(ben),np.percentile(ben,5),np.percentile(ben,95),rd['n_benign_rejected'],rd['f1'],rd['malicious_norm_mean'],rd['benign_norm_mean'],rd['client_cosines_to_benign_mean'][0]))
    a=np.array(rows,dtype=float)
    print(f"{name} {x.partition} nm={nm} det={x.detection_recall:.2f} drop={x.drop_f1:.2f} | att q med={np.median(a[:,1]):.3f} [{a[:,1].min():.3f},{a[:,1].max():.3f}] halfband med={np.median(a[:,2]):.3f} acc_rounds={int(a[:,3].sum())} lamda med={np.median(a[:,4]):.3f} benq med={np.median(a[:,5]):.3f} benrej med={np.median(a[:,8]):.0f} malnorm/bennorm={np.median(a[:,10]/a[:,11]):.2f} cos={np.median(a[:,12]):.3f} below={int((a[:,1]<1-a[:,2]).sum())} above={int((a[:,1]>1+a[:,2]).sum())}")
    if len(sys.argv)>1 and x.partition==sys.argv[1] and nm==int(sys.argv[2]) and name==sys.argv[3]:
        np.set_printoptions(precision=3,suppress=True,linewidth=200)
        print(' rd  q_att  hw  acc lamda benq_med p5 p95 benrej f1 malnorm bennorm cos'); print(a)
