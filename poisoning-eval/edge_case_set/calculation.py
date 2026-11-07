# FIXME: rename (edge_case_set/ -> target_set/, edge_sets/ -> target_sets/): targets are whole hosts of any rarity,
# not only edge cases (density <= p, Definition 1)
#
# Source: Wang et al., "Attack of the Tails: Yes, You Really Can Backdoor Federated Learning" (NeurIPS 2020),
# Sec. 2 (Fig. 2), Definition 1 and Remark 1: fit a Gaussian mixture (one component per class) to a pre-trained
# model's last-layer outputs on normal (non-edge-case) samples and measure the density of candidate samples. The official code
# (https://github.com/kamikazekartik/OOD_Federated_Learning) has no implementation of this step.
#
# Departures from the paper:
#   - candidates: all rows of one unit (host, or host + first path segment with --unit host_path)
#   - classes: requests are labelled ad/tracker or benign
#   - both directions: ad/tracker units get target_label 0, benign units target_label 1
#   - layer: last hidden activation (4096-d), reduced by PCA before the mixture (--layer logit = paper-literal)
#   - p: a quantile of the log-densities of all train rows
#   - no density filter: a target keeps all of its train rows; its rarity is its median_percentile
#   - three density buckets (rare / common / very common) instead of one edge-case set
#   - held-out set: the unit's test rows, minus copies of train rows and rows of the other class
#   - Fig. 2 analogue (host_plots/<id>_<layer>_fig2.png): number of train requests per bin, bins holding target
#     requests coloured, a strip of those bins below, bucket bands and the target's median request
#
# Commands: rank, density (also stores the model's logits), select (targets, per-target .npz/.json).
# plot.py: density_, ad_recall_, and after select targets_density_, all_targets_, stability_
# <layer>[_host_path].png/.csv and host_plots/ fig2; all_targets_: every target's strip under one histogram of all train requests

import argparse
import json
import os
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ('train', 'test')
HELD_OUT_SPLITS = ('test',)
MODEL_CACHE_PATH = os.path.join(HERE, '.model.pt')
OUT_DIR = os.path.join(HERE, 'edge_sets')

CHUNK_SIZE = 8192
SEED = 0
PCA_DIMENSIONS = 16
N_COMPONENTS = 2  # paper: number of classes
PCA_FIT_ROWS = 30000
MIN_EDGE_TRAIN_ROWS = 100  # the attack samples 100 edge-case rows
MIN_ACCURACY = 0.85  # share of the unit's train rows classified correctly
MIN_LABEL_PURITY = 0.98

# on median_percentile; the gaps are a margin for other density settings
BUCKETS = {'rare': (0.0, 0.10), 'common': (0.15, 0.55), 'very_common': (0.65, 1.0)}
DIRECTIONS = {'ad/tracker': 0.0, 'benign': 1.0}  # unit's class -> target_label
TARGETS_PER_CELL = 3
TRAIN_ROWS_BAND = (150, 1000)
MIN_DISTINCT_TRAIN_ROWS = 100
MIN_HELD_OUT_ROWS = 30
MIN_HELD_OUT_SITES = 5
MAX_MISS_RATE = 0.05  # misclassification rate on the held-out rows

# setup.py's FEATURE_COLUMNS, in the model's order
FEATURE_COLUMNS = [
    'content_policy_type', 'url_length', 'brackettodot', 'is_third_party', 'keyword_raw_present',
    'num_get_storage', 'num_set_storage', 'num_get_cookie', 'num_requests_sent_static',
    'req_url_33', 'req_url_135', 'req_url_179',
    'fqdn_4', 'fqdn_13', 'fqdn_14', 'fqdn_15', 'fqdn_23', 'fqdn_26', 'fqdn_27',
    'ng_0_0_2', 'ng_0_15_15', 'ng_2_13_2', 'ng_15_0_3', 'ng_15_0_15', 'ng_15_15_15',
    'avg_ident', 'avg_charperline',
]


# same as setup.py's Rescale / build_model
class Rescale(nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.register_buffer('scale', scale)
        self.register_buffer('offset', offset)

    def forward(self, x):
        return x * self.scale + self.offset


def build_model(specs):
    layers = []
    for kind, *values in specs:
        if kind == 'rescale':
            layers.append(Rescale(*values))
        elif kind == 'linear':
            weight, bias = values
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data = weight
            linear.bias.data = bias
            layers.append(linear)
        elif kind == 'relu':
            layers.append(nn.ReLU())
        elif kind == 'dropout':
            layers.append(nn.Dropout(values[0]))
    return nn.Sequential(*layers)


# host, or host + first path segment (www.google.de/ads/x -> www.google.de/ads)
def unit_of(url, unit):
    parts = urlsplit(url)
    if unit == 'host' or parts.hostname is None:
        return parts.hostname
    if '/' in parts.path:
        first_segment = parts.path.split('/')[1]
    else:
        first_segment = ''
    return parts.hostname + '/' + first_segment


def load_frames(unit, columns=None):
    frames = {}
    for split in SPLITS:
        frame = pd.read_parquet(os.path.join(HERE, f'.{split}.parquet'), columns=columns)
        frame['host'] = frame['name'].map(lambda url: unit_of(url, 'host'))
        frame['unit'] = frame['name'].map(lambda url: unit_of(url, unit))
        frames[split] = frame
    return frames


def feature_matrices(frames):
    features = {}
    for split in SPLITS:
        features[split] = frames[split][FEATURE_COLUMNS].to_numpy(np.float32)
    return features


# per held-out split: True for rows whose feature vector is not in the train split
def novel_masks(features):
    train_set = set()
    for row in features['train']:
        train_set.add(bytes(row))

    masks = {}
    for split in HELD_OUT_SPLITS:
        masks[split] = np.array([bytes(row) not in train_set for row in features[split]], dtype=bool)
    return masks


# chunked, so the 4096-wide activations never all sit in memory
@torch.no_grad()
def run_chunked(network, features, reduce=None):
    outputs = []
    for start in range(0, len(features), CHUNK_SIZE):
        chunk = torch.from_numpy(features[start:start + CHUNK_SIZE])
        output = network(chunk).numpy().astype(np.float64)
        if reduce is not None:
            output = reduce(output)
        outputs.append(output)
    return np.concatenate(outputs)


def settings_suffix(pca_dimensions, n_components, seed=SEED):
    suffix = ''
    if pca_dimensions != PCA_DIMENSIONS or n_components != N_COMPONENTS:
        suffix = f'_pca{pca_dimensions}_k{n_components}'
    if seed != SEED:
        suffix += f'_s{seed}'
    return suffix


# hosts_<layer>*.csv or host_paths_<layer>*.csv
def units_path(layer, unit, pca_dimensions=PCA_DIMENSIONS, n_components=N_COMPONENTS, seed=SEED):
    return os.path.join(OUT_DIR, f'{unit}s_{layer}{settings_suffix(pca_dimensions, n_components, seed)}.csv')


def target_file_prefix(unit_id, layer):
    return os.path.join(OUT_DIR, unit_id.replace('/', '_') + '_' + layer)


# the model trained on AdFlush's dataset without any poisoning (the thesis should say so)
def load_model():
    return build_model(torch.load(MODEL_CACHE_PATH)['layers']).eval()


# the density model, fitted on all train rows. Returns the model, a log-density function and the train log-densities
def fit_density(model, train_features, layer, pca_dimensions, n_components, seed):
    print('warning: using the cached .model.pt and .train/.test.parquet as they are, '
          'rerun setup.py first if the model, the split or FEATURE_COLUMNS changed', flush=True)
    if layer == 'hidden':
        linear_indices = [index for index, module in enumerate(model) if isinstance(module, nn.Linear)]
        network = model[:linear_indices[-1]]
        pca_rows = np.random.default_rng(seed).permutation(len(train_features))[:PCA_FIT_ROWS]
        pca = PCA(n_components=pca_dimensions, random_state=seed)
        pca.fit(run_chunked(network, train_features[pca_rows]))
        reduce = pca.transform
    else:
        network = model
        reduce = None

    gaussian_mixture = GaussianMixture(n_components=n_components, covariance_type='full', random_state=seed)
    train_outputs = run_chunked(network, train_features, reduce)
    gaussian_mixture.fit(train_outputs)
    train_log_density = gaussian_mixture.score_samples(train_outputs)

    def log_density(features):
        if len(features) == 0:
            return np.zeros(0)
        return gaussian_mixture.score_samples(run_chunked(network, features, reduce))

    return log_density, train_log_density


# rank: one row per candidate unit with its density and what select's eligibility needs
def rank_units(layer, quantile, pca_dimensions, n_components, seed, unit):
    frames = load_frames(unit)
    features = feature_matrices(frames)
    model = load_model()
    _, reference = fit_density(model, features['train'], layer, pca_dimensions, n_components, seed)

    threshold = float(np.quantile(reference, quantile))
    sorted_reference = np.sort(reference)

    labels = {}
    correct = {}
    for split in SPLITS:
        labels[split] = frames[split]['CLASS'].to_numpy()
        predicted_ad = run_chunked(model, features[split])[:, 0] > 0
        correct[split] = predicted_ad == labels[split]

    novel = novel_masks(features)
    train_keys = [bytes(row) for row in features['train']]

    train_units = frames['train']['unit'].to_numpy()
    accuracy_per_unit = pd.Series(correct['train']).groupby(train_units).mean()
    train_counts = frames['train']['unit'].value_counts()
    candidates = []
    for name, accuracy in accuracy_per_unit.items():
        if train_counts[name] >= MIN_EDGE_TRAIN_ROWS and accuracy >= MIN_ACCURACY:
            candidates.append(name)

    table = []
    for name in candidates:
        rows = {}
        for split in SPLITS:
            rows[split] = np.flatnonzero(frames[split]['unit'].to_numpy() == name)
        train_rows = rows['train']
        train_log_density = reference[train_rows]
        # share of train rows at most as dense, per row of the unit
        train_percentile = np.searchsorted(sorted_reference, train_log_density, side='right') / len(sorted_reference)

        # Ensure for each split that MIN_LABEL_PURITY % of samples have the same label
        is_ad = np.mean(labels['train'][train_rows]) >= 0.5
        split_purities = []
        for split in SPLITS:
            if len(rows[split]) > 0:
                split_purities.append(float(np.mean(labels[split][rows[split]] == is_ad)))
        purity = min(split_purities)

        if purity < MIN_LABEL_PURITY:
            target_label = None
        elif is_ad:
            target_label = 0.0
        else:
            target_label = 1.0

        other_rows = np.ones(len(reference), dtype=bool)
        other_rows[train_rows] = False
        other_minimum = reference[other_rows].min()

        entry = {
            'unit': name,
            'host': frames['train']['host'].iloc[train_rows[0]],
            'train_rows': len(train_rows),
            'test_rows': len(rows['test']),
            'ad_share': float(np.mean(labels['train'][train_rows])),
            'label_purity': purity,
            'target_label': target_label,
            'clean_accuracy': float(accuracy_per_unit[name]),
            'train_distinct_rows': len(set(train_keys[row] for row in train_rows)),
            'train_sites': int(frames['train']['visit_id'].iloc[train_rows].nunique()),
            'share_below_p': float(np.mean(train_log_density <= threshold)),
            'median_percentile': float(np.mean(reference <= np.median(train_log_density))),
        }
        for bucket, (low, high) in BUCKETS.items():
            in_bucket = (low <= train_percentile) & (train_percentile <= high)
            entry[f'share_{bucket}'] = float(np.mean(in_bucket))
        # Remark 1: less dense than every other train row = out-of-distribution
        entry['share_below_all_benign'] = float(np.mean(train_log_density < other_minimum))

        for split in HELD_OUT_SPLITS:
            split_rows = rows[split]
            split_novel = novel[split][split_rows]
            same_class = labels[split][split_rows] == is_ad
            held_out = split_rows[split_novel & same_class]
            if len(split_rows) > 0:
                entry[f'{split}_duplicate_share'] = float(np.mean(~split_novel))
            else:
                entry[f'{split}_duplicate_share'] = None
            entry[f'{split}_held_out_rows'] = len(held_out)
            entry[f'{split}_held_out_sites'] = int(frames[split]['visit_id'].iloc[held_out].nunique())
            if len(held_out) > 0:
                entry[f'{split}_clean_miss_rate'] = float(1 - np.mean(correct[split][held_out]))
            else:
                entry[f'{split}_clean_miss_rate'] = None
        table.append(entry)

    table = pd.DataFrame(table).sort_values('share_below_p', ascending=False)
    table.to_csv(units_path(layer, unit, pca_dimensions, n_components, seed), index=False)
    print(table.to_string(index=False))


# density: log-density of every row of every split, in parquet row order
def save_log_densities(layer, quantile, pca_dimensions, n_components, seed):
    frames = load_frames('host')
    features = feature_matrices(frames)
    model = load_model()
    log_density, train_log_density = fit_density(model, features['train'], layer, pca_dimensions,
                                                 n_components, seed)
    # the model's logit, for the plot of correctly classified requests
    train_logit = run_chunked(model, features['train'])[:, 0]
    threshold = float(np.quantile(train_log_density, quantile))
    path = os.path.join(OUT_DIR, f'log_density_{layer}{settings_suffix(pca_dimensions, n_components, seed)}.npz')
    np.savez(path,
             train_log_density=train_log_density,
             test_log_density=log_density(features['test']),
             log_density_threshold=np.array(threshold),
             quantile=np.array(quantile),
             pca=np.array(pca_dimensions),
             components=np.array(n_components),
             seed=np.array(seed),
             train_logit=train_logit)
    print(path)


def targets_path(layer, unit):
    if unit == 'host':
        return os.path.join(OUT_DIR, f'targets_{layer}.csv')
    return os.path.join(OUT_DIR, f'targets_{layer}_{unit}.csv')


def bucket_of(median_percentile):
    for name, (low, high) in BUCKETS.items():
        if low <= median_percentile <= high:
            return name
    return None


def bucket_or_gap(median_percentile):
    bucket = bucket_of(median_percentile)
    if bucket is None:
        return 'gap'
    return bucket


def direction_of(target_label):
    for direction, label in DIRECTIONS.items():
        if target_label == label:
            return direction
    return np.nan


def is_eligible(table):
    test_ok = ((table['test_held_out_rows'] >= MIN_HELD_OUT_ROWS)
               & (table['test_held_out_sites'] >= MIN_HELD_OUT_SITES)
               & (table['test_clean_miss_rate'] <= MAX_MISS_RATE))
    train_ok = (table['train_rows'].between(*TRAIN_ROWS_BAND)
                & (table['train_distinct_rows'] >= MIN_DISTINCT_TRAIN_ROWS))
    return table['target_label'].notna() & train_ok & test_ok


# select: up to TARGETS_PER_CELL eligible, stable units per bucket and direction, closest to the bucket's middle.
# Stable: in no other bucket (a gap is fine) in any --compare table
def select_targets(layer, unit, compare, ids):
    table = pd.read_csv(units_path(layer, unit))
    table['bucket'] = table['median_percentile'].map(bucket_of)
    table['direction'] = table['target_label'].map(direction_of)
    table['stable'] = True
    compare_columns = []
    for path in compare:
        other = pd.read_csv(path).set_index('unit')['median_percentile']
        column = f'percentile_{os.path.basename(path)}'
        compare_columns.append(column)
        table[column] = table['unit'].map(other)
        other_bucket = table['unit'].map(other.map(bucket_or_gap))
        table['stable'] &= (other_bucket == table['bucket']) | (other_bucket == 'gap')
    table['eligible'] = is_eligible(table)

    columns = (['unit', 'host', 'bucket', 'direction', 'target_label', 'median_percentile']
               + compare_columns
               + ['share_below_p', 'train_rows', 'train_distinct_rows', 'train_sites',
                  'test_held_out_rows', 'test_held_out_sites', 'test_clean_miss_rate', 'test_duplicate_share',
                  'eligible', 'stable'])

    if ids:
        missing = sorted(set(ids) - set(table['unit']))
        assert not missing, f'not candidates in rank\'s table: {missing}'
        targets = table.set_index('unit').loc[ids].reset_index()
    else:
        chosen_rows = []
        for bucket, (low, high) in BUCKETS.items():
            middle = (low + high) / 2
            for direction in DIRECTIONS:
                cell = table[(table['bucket'] == bucket) & (table['direction'] == direction)]
                chosen = cell[cell['eligible'] & cell['stable']].copy()
                chosen['distance'] = (chosen['median_percentile'] - middle).abs()
                chosen = chosen.sort_values('distance', kind='stable')
                print(f'\n{bucket} {BUCKETS[bucket]}, {direction}: {len(chosen)} eligible and stable of {len(cell)} '
                      f'in the cell ({cell["eligible"].sum()} eligible)')
                print(chosen[columns].head(5).to_string(index=False))
                for _, row in chosen.head(TARGETS_PER_CELL).iterrows():
                    chosen_rows.append(row)
        if chosen_rows:
            targets = pd.DataFrame(chosen_rows)
        else:
            targets = pd.DataFrame(columns=columns)

    targets = targets[columns]
    targets.to_csv(targets_path(layer, unit), index=False)
    print('\ntargets:')
    print(targets.to_string(index=False))
    write_target_sets(layer, unit, targets)


# per target: <id>_<layer>.npz (row indices into the cached parquets) and .json summary
def write_target_sets(layer, unit, targets):
    saved = np.load(os.path.join(OUT_DIR, f'log_density_{layer}.npz'))
    assert int(saved['pca']) == PCA_DIMENSIONS and int(saved['components']) == N_COMPONENTS, \
        'the density file has non-default settings, rerun density'
    threshold = float(saved['log_density_threshold'])
    quantile = float(saved['quantile'])

    frames = load_frames(unit, columns=['name', 'CLASS', 'visit_id', *FEATURE_COLUMNS])
    novel = novel_masks(feature_matrices(frames))

    for split in SPLITS:
        assert len(frames[split]) == len(saved[f'{split}_log_density']), \
            f'{split}: the density file does not match the cached parquet, rerun density'

    for target in targets.itertuples():
        target_label = float(target.target_label)
        assert target_label in DIRECTIONS.values(), f'{target.unit}: mixed labels, no target_label'

        rows = {}
        for split in SPLITS:
            rows[split] = np.flatnonzero(frames[split]['unit'].to_numpy() == target.unit)
        # held out: not a copy of a train row, true label not target_label
        for split in HELD_OUT_SPLITS:
            split_rows = rows[split]
            other_class = frames[split]['CLASS'].to_numpy()[split_rows] != target_label
            rows[split] = split_rows[novel[split][split_rows] & other_class]
        assert len(rows['train']) >= MIN_EDGE_TRAIN_ROWS, f'{target.unit}: only {len(rows["train"])} train rows'

        out_prefix = target_file_prefix(target.unit, layer)
        # exclude_rows = train_rows, both read by experiment.py's load_edge_set
        np.savez(f'{out_prefix}.npz',
                 train_rows=rows['train'], test_rows=rows['test'],
                 exclude_rows=rows['train'], target_label=np.array(target_label),
                 log_density_threshold=np.array(threshold))

        split_summaries = {}
        for split in SPLITS:
            split_rows = rows[split]
            log_densities = saved[f'{split}_log_density'][split_rows]
            if len(split_rows) > 0:
                ad_share = float(np.mean(frames[split]['CLASS'].to_numpy()[split_rows]))
            else:
                ad_share = None
            split_summaries[split] = {
                'rows': len(split_rows),
                'ad_share': ad_share,
                'rows_below_p': int((log_densities <= threshold).sum()),
                'sites': int(frames[split]['visit_id'].iloc[split_rows].nunique()),
            }
        bucket = target.bucket if isinstance(target.bucket, str) else None
        summary = {
            'unit': target.unit,
            'unit_kind': unit,
            'host': target.host,
            'bucket': bucket,
            'target_label': target_label,
            'median_percentile': float(target.median_percentile),
            'settings': {'layer': layer, 'quantile': quantile, 'pca': int(saved['pca']),
                         'components': int(saved['components']), 'pca_fit_rows': PCA_FIT_ROWS, 'seed': SEED},
            'log_density_threshold': threshold,
            'splits': split_summaries,
        }
        with open(f'{out_prefix}.json', 'w') as summary_file:
            json.dump(summary, summary_file, indent=2)

        print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description='p-edge-case example set (Wang et al. 2020, Sec. 2)')
    parser.add_argument('command', choices=['rank', 'select', 'density'])
    parser.add_argument('--layer', choices=['hidden', 'logit'], default='hidden')
    parser.add_argument('--unit', choices=['host', 'host_path'], default='host', help='rank / select: the target unit')
    parser.add_argument('--quantile', type=float, default=0.05, help='p = this quantile of the log-densities of all train rows')
    parser.add_argument('--pca', type=int, default=PCA_DIMENSIONS, help='PCA dimensions before the mixture')
    parser.add_argument('--components', type=int, default=N_COMPONENTS, help='Gaussian mixture components')
    parser.add_argument('--seed', type=int, default=SEED, help='PCA rows and mixture initialisation')
    parser.add_argument('--compare', nargs='*', default=[], help='select only: other rank tables for the stability check')
    parser.add_argument('--ids', nargs='*', default=[], help='select only: these units instead of the automatic choice')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.command == 'rank':
        rank_units(args.layer, args.quantile, args.pca, args.components, args.seed, args.unit)
    elif args.command == 'select':
        select_targets(args.layer, args.unit, args.compare, args.ids)
    else:
        save_log_densities(args.layer, args.quantile, args.pca, args.components, args.seed)


if __name__ == '__main__':
    main()
