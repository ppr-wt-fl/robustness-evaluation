import sys; sys.path.insert(0,'.')
from load import load
import numpy as np, pandas as pd
r=load('data/thesis-lambda0.005'); lam=0.005
b=r[r.host.apply(lambda h:isinstance(h,str)) & (r.schedule=='every')]
rows=[]
for x in b.itertuples():
    peak=x.peak_asr*100
    if x.aggregator!='EPPRFL':
        rows.append(dict(name=x.name,host=x.host[:5],part=x.partition[:3],nm=x.n_malicious,agg='FA',peak=peak)); continue
    q=[];lo=[];hi=[];hon=[];acc=0;n=0;spread=[];below=0;above=0;share=[];q10=[];q40=[]
    for rd in x.rounds:
        if 'epprfl_distances' not in rd or not rd.get('attack_round',False) or rd.get('n_attackers_active',0)==0: continue
        d=np.array(rd['epprfl_distances'])/rd['epprfl_m_h']; w=np.sqrt(lam/rd['epprfl_m_h']); k=x.n_malicious
        q.append(np.median(d[:k])); spread.append(d[:k].max()-d[:k].min()); lo.append(1-w); hi.append(1+w); hon.append(np.median(d[k:])); n+=1
        acc+=rd['n_malicious_accepted']
        # rejected attackers below / above the band; attackers' share of the accepted updates (FedAvg: k / 200)
        rejected=~np.isin(np.arange(k),rd['accepted_clients'])
        below+=(rejected&(d[:k]<1)).sum(); above+=(rejected&(d[:k]>1)).sum()
        share.append(rd['n_malicious_accepted']/max(len(rd['accepted_clients']),1))
        # attackers' median distance in the first and last 10 attack rounds
        (q10 if rd['round']<=10 else q40 if rd['round']>40 else []).append(np.median(d[:k]))
    rows.append(dict(name=x.name,host=x.host[:5],part=x.partition[:3],nm=x.n_malicious,agg='FC',peak=peak,det=x.detection_recall*100,
        q=np.median(q) if q else np.nan,qmin=min(q) if q else np.nan,qmax=max(q) if q else np.nan,lo=np.median(lo) if lo else np.nan,hi=np.median(hi) if hi else np.nan,hon=np.median(hon) if hon else np.nan,spr=np.median(spread) if spread else np.nan,nr=n,
        below=below,above=above,att_share=np.mean(share)*100,fa_share=x.n_malicious/2,q_r1_10=np.median(q10) if q10 else np.nan,q_r41_50=np.median(q40) if q40 else np.nan))
d=pd.DataFrame(rows)
fa=d[d['agg']=='FA'].set_index(['name','host','part','nm']).peak.rename('FA')
fc=d[d['agg']=='FC'].set_index(['name','host','part','nm'])
out=fc.join(fa).drop(columns='agg')
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
print(out.round(2).to_string())
