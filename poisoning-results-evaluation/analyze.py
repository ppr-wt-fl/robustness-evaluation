# numbers for the results section, printed as text tables (figures.py draws the plots)
# usage: python analyze.py [results_dir]
import sys
import numpy as np
import pandas as pd
from load import load, N_CLIENTS, RESULTS_DIR

# same partitions as figures.py (dirichlet_0.5 skipped)
PARTITIONS = ['iid', 'natural', 'dirichlet_1.0']
# G3: attack success within this many pp of the run without attack
G3_BAND = 10

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)

runs = load(sys.argv[1] if len(sys.argv) > 1 else RESULTS_DIR)
runs = runs[runs['partition'].isin(PARTITIONS)]
clean = runs[runs['n_malicious'] == 0]
untargeted = runs[(runs['n_malicious'] > 0) & runs['host'].isna()]
# every-round grid (attacks in rounds 1-50 of 70); the old every_k runs (every 10th round of 100) are left out
backdoors = runs[runs['host'].notna() & (runs['schedule'] == 'every')]


def section(title):
    print(f'\n=== {title} ===')


# G4: test numbers after the last round, EPPRFL - FedAvg, and what EPPRFL rejects per round
section('G4: no attack, test metrics')
for n_rounds in [50, 100]:
    part = clean[clean['n_rounds'] == n_rounds]
    table = part.pivot_table(index='partition', columns='aggregator', values=['f1', 'precision', 'recall', 'fpr'])
    for metric in ['f1', 'precision', 'recall', 'fpr']:
        table[(metric, 'diff_pp')] = (table[(metric, 'EPPRFL')] - table[(metric, 'FedAvg')]) * 100
    print(f'-- {n_rounds} rounds')
    print(table.sort_index(axis=1).round(4))

section('G4: round-0, round-10/25 and final test F1')
rows = []
for run in clean[clean['n_rounds'] == 50].itertuples():
    f1 = [result['f1'] for result in run.rounds]
    rows.append({'aggregator': run.aggregator, 'partition': run.partition, 'f1_round0': f1[0], 'f1_final': f1[-1],
                 'f1_max': max(f1), 'f1_round10': f1[10], 'f1_round25': f1[25]})
print(pd.DataFrame(rows).set_index(['partition', 'aggregator']).sort_index().round(4))
print('-- largest gap EPPRFL - FedAvg in test F1 over all rounds (pp)')
for (partition, n_rounds), part in clean.groupby(['partition', 'n_rounds']):
    f1 = {run.aggregator: np.array([result['f1'] for result in run.rounds]) for run in part.itertuples()}
    print(partition, n_rounds, round(np.abs(f1['EPPRFL'] - f1['FedAvg']).max() * 100, 2))

section('G4: EPPRFL honest clients rejected per round (% of 200), clipped, skipped rounds')
rows = []
for run in clean[clean['aggregator'] == 'EPPRFL'].itertuples():
    rejected = [result.get('n_benign_rejected') or 0 for result in run.rounds[1:]]
    clipped = [result.get('epprfl_n_clipped') or 0 for result in run.rounds[1:]]
    rows.append({'partition': run.partition, 'n_rounds': run.n_rounds,
                 'rejected_mean_pct': np.mean(rejected) * 100 / N_CLIENTS,
                 'rejected_min_pct': min(rejected) * 100 / N_CLIENTS,
                 'rejected_max_pct': max(rejected) * 100 / N_CLIENTS,
                 'rejected_last10_pct': np.mean(rejected[-10:]) * 100 / N_CLIENTS,
                 'clipped_mean_pct': np.mean(clipped) * 100 / N_CLIENTS,
                 'clipped_min_pct': min(clipped) * 100 / N_CLIENTS,
                 'clipped_max_pct': max(clipped) * 100 / N_CLIENTS,
                 'rounds_all_rejected': sum(r == N_CLIENTS for r in rejected)})
print(pd.DataFrame(rows).set_index(['partition', 'n_rounds']).round(2))

# checks the text's reason for the honest rejections: on non-IID partitions the clients' updates differ more from
# each other and from the historical update, and fluctuate more between rounds, so more fall outside the band
section('G4: why EPPRFL rejects honest clients, 100-round runs')
rows = []
for run in clean[(clean['aggregator'] == 'EPPRFL') & (clean['n_rounds'] == 100)].itertuples():
    rounds = [result for result in run.rounds if result.get('epprfl_distances')]
    # client x round: distance to the historical update relative to its median m_h, cosine to the mean update
    ratios = np.array([np.array(result['epprfl_distances']) / result['epprfl_m_h'] for result in rounds]).T
    cosines = np.array([result['client_cosines_to_benign_mean'] for result in rounds]).T
    accepted = np.zeros_like(ratios, dtype=bool)
    for column, result in enumerate(rounds):
        accepted[result['accepted_clients'], column] = True
    rejected_share = pd.Series(1 - accepted.mean(1))
    positive_rates = pd.Series(run.client_positive_rates)
    norms = pd.Series(np.array([result['client_norms'] for result in rounds]).mean(0))
    rows.append({'partition': run.partition,
                 # rejected for being farther from the historical update than typical (above m_h) or closer
                 'rejected_above_pct': (~accepted & (ratios > 1)).sum() * 100 / max((~accepted).sum(), 1),
                 'cosine_to_mean': cosines.mean(),
                 # spread of the distances within a round (coefficient of variation)
                 'distance_cv': (ratios.std(0) / ratios.mean(0)).mean(),
                 # how much one client's distance moves between rounds
                 'distance_fluctuation': ratios.std(1).mean(),
                 # how often a client switches between accepted and rejected
                 'status_flips_per_client': (accepted[:, 1:] != accepted[:, :-1]).sum(1).mean(),
                 'clients_never_rejected_pct': (rejected_share == 0).mean() * 100,
                 # are the rejected clients the dissimilar ones? (< 0: the less similar, the more often rejected)
                 'corr_rejected_vs_cosine': rejected_share.corr(pd.Series(cosines.mean(1)), method='spearman'),
                 'corr_rejected_vs_fluctuation': rejected_share.corr(pd.Series(ratios.std(1)), method='spearman'),
                 # are they the clients with unusual data? ad/tracker share far from the median, data size, update norm
                 'corr_rejected_vs_unusual_ad_share': rejected_share.corr((positive_rates - positive_rates.median()).abs(), method='spearman'),
                 'corr_rejected_vs_size': rejected_share.corr(pd.Series(run.client_sizes), method='spearman'),
                 'corr_rejected_vs_norm': rejected_share.corr(norms, method='spearman')})
print(pd.DataFrame(rows).set_index('partition').round(3))

section('G4: same-level band, std of test F1 / FPR over the last 10 rounds (pp)')
rows = []
for run in clean.itertuples():
    last = run.rounds[-10:]
    rows.append({'aggregator': run.aggregator, 'partition': run.partition, 'n_rounds': run.n_rounds,
                 'f1_std_pp': np.std([result['f1'] for result in last]) * 100,
                 'fpr_std_pp': np.std([result['fpr'] for result in last]) * 100})
band = pd.DataFrame(rows)
print(band.set_index(['partition', 'n_rounds', 'aggregator']).sort_index().round(3))
print('max f1 std pp:', band['f1_std_pp'].max().round(3), ' max fpr std pp:', band['fpr_std_pp'].max().round(3))

# G2: drop vs the same rule's run without attack
section('G2: test F1 drop (pp), worst partition, FedAvg / EPPRFL')
for value in ['drop_f1', 'rise_fpr']:
    table = untargeted.pivot_table(index='name', columns=['n_malicious', 'aggregator'], values=value, aggfunc='max')
    print(f'-- {value}')
    print(table.round(2))

section('G2: worst partition per cell (which partition gives the max)')
for value in ['drop_f1', 'rise_fpr']:
    worst = untargeted.loc[untargeted.groupby(['name', 'n_malicious', 'aggregator'])[value].idxmax()]
    print(f'-- {value}')
    print(worst.pivot_table(index='name', columns=['n_malicious', 'aggregator'], values='partition', aggfunc='first'))

section('G2: per partition, drop_f1 / rise_fpr')
for partition in PARTITIONS:
    part = untargeted[untargeted['partition'] == partition]
    for value in ['drop_f1', 'rise_fpr']:
        print(f'-- {partition} {value}')
        print(part.pivot_table(index='name', columns=['n_malicious', 'aggregator'], values=value).round(2))

section('G2: worst case over attacks and partitions per share')
worst = untargeted.groupby(['n_malicious', 'aggregator'])[['drop_f1', 'rise_fpr']].max().unstack()
print(worst.round(2))
# G2 as stated: EPPRFL's drop smaller than FedAvg's (per attack and partition, and in the worst case)
paired = untargeted.pivot_table(index=['name', 'partition', 'n_malicious'], columns='aggregator', values=['drop_f1', 'rise_fpr'])
better = pd.DataFrame({'f1': paired[('drop_f1', 'EPPRFL')] < paired[('drop_f1', 'FedAvg')],
                       'fpr': paired[('rise_fpr', 'EPPRFL')] < paired[('rise_fpr', 'FedAvg')]})
print('-- settings where EPPRFL drops less / FPR rises less than FedAvg, per share')
print(better.groupby('n_malicious').agg(['sum', 'count']))
print('-- settings where EPPRFL is worse in F1 drop (name, partition, share)')
print(paired[~better['f1']].round(2))

section('G2: test precision / recall under attack (EPPRFL and FedAvg), worst-drop setting per attack')
for aggregator in ['FedAvg', 'EPPRFL']:
    part = untargeted[untargeted['aggregator'] == aggregator]
    worst = part.loc[part.groupby('name')['drop_f1'].idxmax()]
    print(f'-- {aggregator}')
    print(worst[['name', 'partition', 'n_malicious', 'f1', 'precision', 'recall', 'fpr', 'clean_f1', 'drop_f1', 'rise_fpr']].round(4).to_string(index=False))

section('G2: EPPRFL detection, attackers rejected (%) / honest rejected (%), mean over attack rounds')
epprfl = untargeted[untargeted['aggregator'] == 'EPPRFL'].copy()
epprfl['detection_pct'] = epprfl['detection_recall'] * 100
epprfl['benign_rejected_pct'] = epprfl['benign_rejected'] * 100
for value in ['detection_pct', 'benign_rejected_pct']:
    print(f'-- {value}')
    print(epprfl.pivot_table(index=['name', 'partition'], columns='n_malicious', values=value).round(1))

section('G2 over the rounds: EPPRFL rounds with every attacker rejected, and the test F1 change (pp) in rounds where attackers pass / are all rejected')
rows = []
for run in epprfl[epprfl['name'].isin(['MinMax', 'MinMax-std', 'FangKrum', 'MinSum'])].itertuples():
    rejected_all = np.array([result.get('n_malicious_rejected') == result.get('n_attackers_active') for result in run.rounds[1:]])
    change = np.diff([result['f1'] for result in run.rounds]) * 100
    # first round from which every later round rejects every attacker
    streak = next((index + 1 for index in range(len(rejected_all)) if rejected_all[index:].all()), None)
    rows.append({'name': run.name, 'partition': run.partition, 'n_malicious': run.n_malicious,
                 'rounds_all_rejected': rejected_all.sum(), 'all_rejected_from_round': streak,
                 'df1_passed_pp': change[~rejected_all].mean() if (~rejected_all).any() else np.nan,
                 'df1_rejected_pp': change[rejected_all].mean() if rejected_all.any() else np.nan})
print(pd.DataFrame(rows).set_index(['name', 'partition', 'n_malicious']).sort_index().round(2))

section('G2: EPPRFL rounds with no client kept (zero update) under attack')
rows = []
for run in epprfl.itertuples():
    kept = [result.get('epprfl_n_benign') for result in run.rounds[1:]]
    rows.append({'name': run.name, 'partition': run.partition, 'n_malicious': run.n_malicious,
                 'rounds_none_kept': sum(k == 0 for k in kept if k is not None)})
skipped = pd.DataFrame(rows)
print(skipped[skipped['rounds_none_kept'] > 0].to_string(index=False) or 'none')

section('G2 by share: EPPRFL attackers / honest rejected (%), mean and range over attacks and partitions')
by_share = epprfl.groupby('n_malicious')[['detection_pct', 'benign_rejected_pct']].agg(['mean', 'min', 'max'])
print(by_share.round(1))
print('-- without Gaussian noise and sign flipping (the attacks F&C always catches)')
crafted = epprfl[~epprfl['name'].isin(['Gaussian', 'SignFlipping'])]
print(crafted.groupby('n_malicious')[['detection_pct', 'benign_rejected_pct']].agg(['mean', 'min', 'max']).round(1))
print('-- per attack, mean over partitions')
print(epprfl.pivot_table(index='name', columns='n_malicious', values='detection_pct').round(1))
print('-- settings where FedAvg drops more than 1 pp / F&C drops more than 1 pp, per share (of 18)')
print(untargeted.assign(over=untargeted['drop_f1'] > 1).pivot_table(index='n_malicious', columns='aggregator', values='over', aggfunc='sum'))
print('-- median drop over attacks and partitions, per share')
print(untargeted.pivot_table(index='n_malicious', columns='aggregator', values=['drop_f1', 'rise_fpr'], aggfunc='median').round(2))

# G3: highest test attack success from the first attack round on (peak_asr), floor = clean miss rate
section('G3: peak test attack success (%), worst partition, FedAvg / EPPRFL')
table = backdoors.pivot_table(index=['host', 'name'], columns=['n_malicious', 'aggregator'], values='peak_asr', aggfunc='max') * 100
print(table.round(1))
section('G3: floor (clean miss rate of the target, %), max over partitions')
print((backdoors.pivot_table(index='host', columns='aggregator', values='floor_asr', aggfunc='max') * 100).round(2))

section('G3: per partition peak attack success (%)')
for partition in PARTITIONS:
    part = backdoors[backdoors['partition'] == partition]
    print(f'-- {partition}')
    print((part.pivot_table(index=['host', 'name'], columns=['n_malicious', 'aggregator'], values='peak_asr') * 100).round(1))

section(f'G3 verdict: settings with peak above floor + {G3_BAND} pp, per share and aggregator')
over = backdoors.assign(over=(backdoors['peak_asr'] - backdoors['floor_asr']) * 100 > G3_BAND)
print(over.pivot_table(index='n_malicious', columns='aggregator', values='over', aggfunc=['sum', 'count']))
print(over.pivot_table(index=['host', 'name'], columns=['n_malicious', 'aggregator'], values='over', aggfunc='sum'))

section('G3: test attack success right after the last attack and 20 rounds later (%), worst partition')
for value in ['asr_last_attack', 'asr_after_decay']:
    print(f'-- {value}')
    print((backdoors.pivot_table(index=['host', 'name'], columns=['n_malicious', 'aggregator'], values=value, aggfunc='max') * 100).round(1))

section('G3: final test attack success (round 70, %), worst partition')
backdoors = backdoors.assign(final_asr=[run.rounds[-1]['attack_success_rate'] for run in backdoors.itertuples()])
print((backdoors.pivot_table(index=['host', 'name'], columns=['n_malicious', 'aggregator'], values='final_asr', aggfunc='max') * 100).round(1))

# attack rounds 1-50, then 20 rounds without attack (attack_round False)
section('G3: rounds with test attack success above 50 %, of the 50 attack rounds / of the 20 rounds after, worst partition')
backdoors = backdoors.assign(
    attack_rounds_over_half=[sum(result['attack_success_rate'] > 0.5 for result in run.rounds if result.get('attack_round'))
                             for run in backdoors.itertuples()],
    after_rounds_over_half=[sum(result['attack_success_rate'] > 0.5 for result in run.rounds if result.get('attack_round') is False)
                            for run in backdoors.itertuples()])
for value in ['attack_rounds_over_half', 'after_rounds_over_half']:
    print(f'-- {value}')
    print(backdoors.pivot_table(index=['host', 'name'], columns=['n_malicious', 'aggregator'], values=value, aggfunc='max'))

section('G3: non-target test F1 drop (pp), worst partition')
print(backdoors.pivot_table(index=['host', 'name'], columns=['n_malicious', 'aggregator'], values='drop_non_target_f1', aggfunc='max').round(2))

section('G3: EPPRFL attackers rejected in attack rounds (%), mean over partitions')
bd_epprfl = backdoors[backdoors['aggregator'] == 'EPPRFL']
print((bd_epprfl.pivot_table(index=['host', 'name'], columns='n_malicious', values='detection_recall') * 100).round(1))
print('-- per partition')
print((bd_epprfl.pivot_table(index=['host', 'name', 'partition'], columns='n_malicious', values='detection_recall') * 100).round(1))
