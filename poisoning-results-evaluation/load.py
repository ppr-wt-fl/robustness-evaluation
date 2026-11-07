# one row per run from the results/*.jsonl files experiment.py writes, plus the matched clean run's numbers
import os
import sys
import json
import glob
import pickle
import hashlib
import numpy as np
import pandas as pd

# thesis grid: two-split untargeted runs + every-round backdoor grid, FedAvg + EPPRFL lambda 0.005 only
# (one lambda per folder, load.py doesn't split by lambda)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'thesis-lambda0.005')
PLOTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
PLOTS_DIR = PLOTS_ROOT
# other aggregators in the result files are dropped
AGGREGATORS = ['FedAvg', 'EPPRFL']
N_CLIENTS = 200
# draft: low-res PNG only; set False for the final 400 dpi PNG + PDF
DRAFT = False
# draw the figure title above the plot (drafts always do); off for the thesis, where the caption gives it
TITLES = os.environ.get('TITLES', '1') == '1'


def first_attack_index(rounds):
    for index, result in enumerate(rounds):
        if result.get('attack_round'):
            return index
    return len(rounds)


def mean_over(rounds, field, divide_by=1):
    values = [(result.get(field) or 0) / divide_by for result in rounds]
    return np.mean(values) if values else np.nan


def load(results_dir=None):
    global PLOTS_DIR
    results_dir = results_dir or (sys.argv[1] if len(sys.argv) > 1 else RESULTS_DIR)
    # plots/<backup folder> for <backup folder>/poisoning-eval/results, else plots/<results folder>
    name = os.path.basename(os.path.abspath(results_dir))
    if name == 'results':
        name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(results_dir))))
    PLOTS_DIR = os.path.join(PLOTS_ROOT, name)

    # cache: hash of every result file and of this file (code changes also reload), stored before the table
    paths = sorted(glob.glob(os.path.join(results_dir, '*.jsonl')))
    digest = hashlib.sha256()
    for path in paths + [os.path.abspath(__file__)]:
        digest.update(os.path.basename(path).encode())
        with open(path, 'rb') as handle:
            digest.update(hashlib.file_digest(handle, 'sha256').digest())
    cache_path = os.path.join(results_dir, 'runs.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as handle:
            if pickle.load(handle) == digest.hexdigest():
                return pickle.load(handle)

    rows = []
    for path in paths:
        with open(path) as results_file:
            for line in results_file:
                run = json.loads(line)
                if not run.get('completed') or run['aggregator'] not in AGGREGATORS:
                    continue
                config = run['config']
                rounds = run['round_results']
                # attack rounds, or every round after round 0 in a clean run
                attacked = rounds[first_attack_index(rounds):] if config['n_malicious'] else rounds[1:]
                attacked = [result for result in attacked if result.get('attack_round', True)]
                row = {
                    'aggregator': run['aggregator'],
                    'attack': run['attack_display_name'],
                    'attack_key': run['attack'],
                    'host': run['target_host'],
                    'schedule': run['attack_rounds'],
                    'n_malicious': config['n_malicious'],
                    'partition': config['partition_scheme'],
                    'n_rounds': config['n_rounds'],
                    'asr': run.get('test_attack_success_rate'),
                    'asr_last_attack': run.get('test_attack_success_rate_at_last_attack'),
                    'asr_after_decay': run.get('test_attack_success_rate_after_decay'),
                    'benign_rejected': mean_over(attacked, 'n_benign_rejected', N_CLIENTS - config['n_malicious']),
                    'detection_recall': mean_over(attacked, 'detection_recall'),
                    'client_sizes': config['client_sizes'],
                    'client_positive_rates': config['client_positive_rates'],
                    'rounds': rounds,
                }
                for metric, value in run['test'].items():
                    row[metric] = value
                for metric, value in run['test_non_target'].items():
                    row['non_target_' + metric] = value
                rows.append(row)
    runs = pd.DataFrame(rows)

    # short name: backdoor variant without the host (MR-naive, Tails-pgd), the two Min-Max kept apart
    names = []
    for key, host, attack in zip(runs['attack_key'], runs['host'], runs['attack']):
        name = key.removesuffix('-' + host) if isinstance(host, str) else key
        if 'standard deviation' in attack:
            name += '-std'
        names.append(name)
    runs['name'] = names
    runs['share'] = runs['n_malicious'] / N_CLIENTS

    # matched clean run: same aggregator, partition and length, no attackers
    clean = runs[runs['n_malicious'] == 0].set_index(['aggregator', 'partition', 'n_rounds'])
    assert clean.index.is_unique, 'two clean runs for one aggregator / partition / length'
    keys = list(zip(runs['aggregator'], runs['partition'], runs['n_rounds']))
    for metric in ['f1', 'fpr', 'non_target_f1']:
        # NaN when the matched clean run hasn't been run
        runs['clean_' + metric] = [clean.loc[key, metric] if key in clean.index else np.nan for key in keys]
    runs['drop_f1'] = (runs['clean_f1'] - runs['f1']) * 100
    runs['rise_fpr'] = (runs['fpr'] - runs['clean_fpr']) * 100
    runs['drop_non_target_f1'] = (runs['clean_non_target_f1'] - runs['non_target_f1']) * 100

    # backdoors: highest test ASR from the first attack round to the end, and the clean run's miss rate
    # of the same host over the same rounds (the floor)
    peaks = []
    floors = []
    for run, key in zip(runs.itertuples(), keys):
        if not isinstance(run.host, str):
            peaks.append(np.nan)
            floors.append(np.nan)
            continue
        start = first_attack_index(run.rounds)
        peaks.append(max(result['attack_success_rate'] for result in run.rounds[start:]))
        if key not in clean.index:
            floors.append(np.nan)
            continue
        clean_rounds = clean.loc[key, 'rounds'][start:]
        floors.append(max(result['clean_success_rates'][run.host] for result in clean_rounds))
    runs['peak_asr'] = peaks
    runs['floor_asr'] = floors
    with open(cache_path, 'wb') as handle:
        pickle.dump(digest.hexdigest(), handle)
        pickle.dump(runs, handle)
    return runs


def per_round(runs, *fields):
    # long table: one row per (run, round)
    rows = []
    for run in runs.itertuples():
        for result in run.rounds:
            row = {
                'aggregator': run.aggregator, 'name': run.name, 'host': run.host, 'schedule': run.schedule,
                'n_malicious': run.n_malicious, 'partition': run.partition, 'n_rounds': run.n_rounds,
                'round': result['round'],
            }
            for field in fields:
                row[field] = result.get(field)
            rows.append(row)
    return pd.DataFrame(rows)


def save(fig, name, title, always_title=False):
    # name may contain a subfolder
    os.makedirs(os.path.dirname(os.path.join(PLOTS_DIR, name)), exist_ok=True)
    # always_title: figures whose title stays in the thesis too
    if DRAFT or TITLES or always_title:
        # bottom edge 0.05 in above the figure, so tall and flat figures get the same gap
        fig.suptitle(title, y=1 + 0.05 / fig.get_figheight(), va='bottom', fontsize='large')
    if not DRAFT:
        fig.savefig(os.path.join(PLOTS_DIR, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, name + '.png'), dpi=200 if DRAFT else 400, bbox_inches='tight')


def save_table(table, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(PLOTS_DIR, name + '.csv'))
