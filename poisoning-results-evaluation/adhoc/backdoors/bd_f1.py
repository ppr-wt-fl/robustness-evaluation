import sys; sys.path.insert(0,'.')
from load import load
import numpy as np, pandas as pd
r=load('data/thesis-lambda0.005')
b=r[r.host.apply(lambda h:isinstance(h,str)) & (r.schedule=='every')].copy()
clean=r[r.n_malicious==0].set_index(['aggregator','partition','n_rounds'])
dips=[]
for x in b.itertuples():
    c=clean.loc[(x.aggregator,x.partition,x.n_rounds),'rounds']
    cf={rd['round']:rd['f1'] for rd in c if 'f1' in rd}
    d=[(cf[rd['round']]-rd['f1'])*100 for rd in x.rounds if 'f1' in rd and rd['round'] in cf and rd['round']>0]
    dips.append(max(d) if d else np.nan)
b['max_round_dip']=dips
print('schedules',b.schedule.unique(),'rounds',b.n_rounds.unique(), 'n runs',len(b))
pd.set_option('display.width',250)
for col in ['drop_f1','drop_non_target_f1','max_round_dip']:
    print('\n==',col,'(pp, max over partitions)')
    print(b.pivot_table(index=['host','name'],columns=['n_malicious','aggregator'],values=col,aggfunc='max').round(2))
print('\nworst 10 final drop_f1:'); print(b.nlargest(10,'drop_f1')[['aggregator','name','host','partition','n_malicious','drop_f1','drop_non_target_f1','max_round_dip']].round(2).to_string())
print('\nworst 10 per-round dip:'); print(b.nlargest(10,'max_round_dip')[['aggregator','name','host','partition','n_malicious','drop_f1','max_round_dip']].round(2).to_string())
print('\nmax per aggregator:'); print(b.groupby('aggregator')[['drop_f1','drop_non_target_f1','max_round_dip']].max().round(2))
print(b.groupby('aggregator')[['drop_f1']].min().round(2))
e=b[b.aggregator=='EPPRFL']; print(e.nlargest(3,'max_round_dip')[['name','host','partition','n_malicious','max_round_dip']].to_string())
