import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from calculation import (BUCKETS, DIRECTIONS, HERE, MIN_EDGE_TRAIN_ROWS, N_COMPONENTS, OUT_DIR, PCA_DIMENSIONS, SEED,
                         bucket_or_gap, load_frames, settings_suffix, target_file_prefix, targets_path, unit_of)

PLOT_DIR = os.path.join(OUT_DIR, 'host_plots')
PLOT_LOW_QUANTILE = 0.01
# print quality for the thesis
DPI = 600

# plots: dataviz reference palette, categorical slots 1-2, neutral ink
BLUE = '#2a78d6'
ORANGE = '#eb6834'
INK = '#0b0b0b'
MUTED = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'
BAND = '#9b9a96'
DIRECTION_COLORS = {'ad/tracker': BLUE, 'benign': ORANGE}
DIRECTION_NAMES = {'ad/tracker': 'ad/tracker', 'benign': 'benign'}
# (label, rank table suffix) for the stability plot
STABILITY_SETTINGS = [
    ('PCA 8', '_pca8_k2'),
    ('PCA 16', ''),
    ('PCA 32', '_pca32_k2'),
    ('PCA 64', '_pca64_k2'),
    ('PCA 16\nseed 1', '_s1'),
    ('PCA 16\n5 components', '_pca16_k5'),
]


def style_plots():
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})


def unit_suffix(unit):
    if unit == 'host':
        return ''
    return '_' + unit


# bucket ranges on the log-density axis: quantiles of the train log-densities
def log_density_buckets(reference):
    ranges = {}
    for bucket, (low, high) in BUCKETS.items():
        ranges[bucket] = (float(np.quantile(reference, low)), float(np.quantile(reference, high)))
    return ranges


# shaded bucket bands, gaps unshaded, named at the top (or right) edge. Call after the axis limits are set
def shade_buckets(axis, ranges, horizontal=False, labels=True):
    if horizontal:
        axis_low, axis_high = axis.get_ylim()
    else:
        axis_low, axis_high = axis.get_xlim()
    previous_center = None
    label_height = 1.0
    for bucket, (low, high) in ranges.items():
        low = max(low, axis_low)
        high = min(high, axis_high)
        if low >= high:
            continue
        name = bucket.replace('_', ' ')
        center = (low + high) / 2
        if horizontal:
            axis.axhspan(low, high, color=BAND, alpha=0.15, linewidth=0, zorder=0)
            if labels:
                axis.text(1.0, center, ' ' + name, transform=axis.get_yaxis_transform(),
                          ha='left', va='center', fontsize=8, color=MUTED)
        else:
            axis.axvspan(low, high, color=BAND, alpha=0.15, linewidth=0, zorder=0)
            if labels:
                # a label too close to the previous one goes one line higher
                fraction = (center - axis_low) / (axis_high - axis_low)
                if previous_center is not None and fraction - previous_center < 0.15 and label_height == 1.0:
                    label_height = 1.06
                else:
                    label_height = 1.0
                axis.text(center, label_height, name, transform=axis.get_xaxis_transform(),
                          ha='center', va='bottom', fontsize=8, color=MUTED)
                previous_center = fraction


# 100 bins from the 1st percentile of all train rows and the target rows together to the densest row
def target_bins(reference, unit_log_density):
    both = np.concatenate([reference, unit_log_density])
    return np.linspace(np.quantile(both, 0.01), both.max(), 100)


# y-axis ticks as 10k, 20k, ...
def thousands(value, position):
    if abs(value) >= 1000:
        return f'{value / 1000:g}k'
    return f'{value:g}'


# grey histogram: number of train requests per bin
def draw_all_requests(axis, bins, reference):
    reference_count = np.histogram(reference, bins=bins)[0]
    axis.stairs(reference_count, bins, fill=True, color=BAND, label='all training requests')
    axis.set_xlim(bins[0], bins[-1])
    axis.set_ylabel('number of requests per bin')
    axis.yaxis.set_major_formatter(FuncFormatter(thousands))
    return reference_count


# one target's strip from bottom to bottom + height: the bins holding at least one of its train rows, and its median row
def draw_target_strip(axis, bins, unit_log_density, color, bottom=0.0, height=1.0):
    has_unit_rows = np.histogram(unit_log_density, bins=bins)[0] > 0
    spans = []
    for index in range(len(has_unit_rows)):
        if has_unit_rows[index]:
            spans.append((bins[index], bins[index + 1] - bins[index]))
    axis.broken_barh(spans, (bottom, height), color=color, linewidth=0)
    median = float(np.median(unit_log_density))
    axis.plot([median, median], [bottom, bottom + height], color=INK, linewidth=1.5)


# plot: overview, correctly classified requests by density, and (if select has run) targets and stability for --unit
def plot_all(layer, pca_dimensions, n_components, seed, unit):
    style_plots()
    suffix = f'{layer}{settings_suffix(pca_dimensions, n_components, seed)}'
    saved = np.load(os.path.join(OUT_DIR, f'log_density_{suffix}.npz'))
    plot_overview(saved, suffix)
    plot_ad_recall(saved, suffix)
    if os.path.exists(targets_path(layer, unit)) and suffix == layer:
        targets = pd.read_csv(targets_path(layer, unit))
        train_units = load_frames(unit, columns=['name'])['train']['unit'].to_numpy()
        plot_targets(saved, targets, train_units, layer, unit)
        plot_all_targets(saved, targets, train_units, layer, unit)
        plot_stability(targets, layer, unit)
        for target in targets.itertuples():
            plot_fig2(saved['train_log_density'], target, train_units, layer)


# share of train rows vs share of hosts (at their median row) per log-density bin
def plot_overview(saved, suffix):
    rows = pd.read_parquet(os.path.join(HERE, '.train.parquet'), columns=['name'])
    rows['host'] = rows['name'].map(lambda url: unit_of(url, 'host'))
    rows['log_density'] = saved['train_log_density']

    host_stats = rows.groupby('host')['log_density'].agg(['size', 'median'])
    hosts = host_stats.loc[host_stats['size'] >= MIN_EDGE_TRAIN_ROWS, 'median']

    # the long left tail is cut off and not drawn, as in all_targets
    low = rows['log_density'].quantile(PLOT_LOW_QUANTILE)
    high = rows['log_density'].max()
    bins = np.linspace(low, high, 100)
    rows_cut = (rows['log_density'] < low).sum()
    hosts_cut = (hosts < low).sum()
    cut_note = (f'x-axis starts at the {PLOT_LOW_QUANTILE:.0%} quantile of the training requests; '
                f'{rows_cut:,} requests and {hosts_cut} hosts left of it are not drawn')

    figure, axis = plt.subplots(figsize=(10, 4))
    row_share = np.histogram(rows['log_density'], bins=bins)[0] / len(rows)
    host_share = np.histogram(hosts, bins=bins)[0] / len(hosts)
    axis.stairs(row_share, bins, color=ORANGE, linewidth=2, label=f'share of training requests (n = {len(rows):,})')
    axis.stairs(host_share, bins, color=BLUE, linewidth=2,
                label=f'share of hosts with at least {MIN_EDGE_TRAIN_ROWS} training requests, at their median request (n = {len(hosts)})')
    axis.set_xlim(low, high)
    shade_buckets(axis, log_density_buckets(saved['train_log_density']))
    axis.grid(axis='y', color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_ylabel('share per bin')
    axis.set_xlabel('log-density ln p(x) (higher = more typical)')
    axis.legend(frameon=False, loc='upper left')
    axis.set_title(f'Training requests under the density model (PCA {int(saved["pca"])}, '
                   f'{int(saved["components"])} components)', color=INK, loc='left', pad=22)
    figure.text(0.01, 0.005, cut_note, color=MUTED, fontsize=8)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = os.path.join(OUT_DIR, f'density_{suffix}.png')
    figure.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    print(out_path)


# share of train rows classified correctly, per percentile bin, ad/tracker and benign rows separately
def plot_ad_recall(saved, suffix):
    assert 'train_logit' in saved.files, 'the density file has no train_logit, rerun density'
    labels = pd.read_parquet(os.path.join(HERE, '.train.parquet'), columns=['CLASS'])['CLASS'].to_numpy()
    reference = saved['train_log_density']
    sorted_reference = np.sort(reference)
    percentile = np.searchsorted(sorted_reference, reference, side='right') / len(reference)
    correct = (saved['train_logit'] > 0) == labels

    n_bins = 40
    edges = np.linspace(0, 1, n_bins + 1)
    bin_of_row = np.digitize(percentile, edges[1:-1], right=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ad_recall = np.full(n_bins, np.nan)
    benign_accuracy = np.full(n_bins, np.nan)
    for index in range(n_bins):
        in_bin = bin_of_row == index
        ad_rows = in_bin & labels
        benign_rows = in_bin & ~labels
        if ad_rows.sum() > 0:
            ad_recall[index] = np.mean(correct[ad_rows])
        if benign_rows.sum() > 0:
            benign_accuracy[index] = np.mean(correct[benign_rows])

    figure, axis = plt.subplots(figsize=(8, 3.6))
    axis.plot(centers, ad_recall, color=BLUE, linewidth=2, marker='o', markersize=4,
              label=f'ad/tracker requests: share classified as ad/tracker (n = {labels.sum():,})')
    axis.plot(centers, benign_accuracy, color=ORANGE, linewidth=2, marker='o', markersize=4,
              label=f'benign requests: share classified as benign (n = {(~labels).sum():,})')
    axis.set_xlim(0, 1)
    axis.set_ylim(-0.02, 1.02)
    shade_buckets(axis, BUCKETS)
    axis.grid(axis='y', color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_xlabel('percentile of the request (share of training requests at most as dense)')
    axis.set_ylabel('share classified correctly')
    axis.set_title('Correctly classified training requests by density', color=INK, loc='left', pad=22)
    axis.legend(frameon=False, loc='lower left')
    figure.tight_layout()
    out_path = os.path.join(OUT_DIR, f'ad_recall_{suffix}.png')
    figure.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    print(out_path)


# all targets on one log-density axis: all train requests on top, per target its median row and 25-75 % of its rows below
def plot_targets(saved, targets, train_units, layer, unit):
    reference = saved['train_log_density']
    low = float(np.quantile(reference, 0.01))
    high = float(reference.max())
    bins = np.linspace(low, high, 100)
    ranges = log_density_buckets(reference)

    ordered = []
    for direction in DIRECTIONS:
        in_direction = targets[targets['direction'] == direction].sort_values('median_percentile')
        for _, target in in_direction.iterrows():
            ordered.append(target)

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 2.2 + 0.32 * len(ordered)), sharex=True,
                                         gridspec_kw={'height_ratios': [2.2, 0.32 * len(ordered)]})
    row_count = np.histogram(np.clip(reference, low, None), bins=bins)[0]
    top.stairs(row_count, bins, color=MUTED, linewidth=1.5)
    top.set_ylabel('number of\nrequests per bin')
    top.set_title('Candidate targets on the training log-density', color=INK, loc='left', pad=22)
    top.yaxis.set_major_formatter(FuncFormatter(thousands))
    top.grid(axis='y', color=GRID, linewidth=0.6)
    top.set_axisbelow(True)
    top.set_xlim(low, high)
    shade_buckets(top, ranges)

    labels = []
    for position, target in enumerate(ordered):
        unit_log_density = reference[train_units == target['unit']]
        median = float(np.median(unit_log_density))
        quartile_low = float(np.quantile(unit_log_density, 0.25))
        quartile_high = float(np.quantile(unit_log_density, 0.75))
        color = DIRECTION_COLORS[target['direction']]
        bottom.plot([max(quartile_low, low), quartile_high], [position, position], color=color, linewidth=2,
                    solid_capstyle='round')
        bottom.plot([median], [position], marker='o', markersize=8, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=2)
        labels.append(f'{target["unit"]}  ({target["median_percentile"]:.3f})')
    bottom.set_yticks(range(len(ordered)))
    bottom.set_yticklabels(labels, fontsize=8)
    for tick, target in zip(bottom.get_yticklabels(), ordered):
        tick.set_color(INK)
    bottom.set_ylim(len(ordered) - 0.5, -0.5)
    bottom.set_xlim(low, high)
    shade_buckets(bottom, ranges, labels=False)
    bottom.spines['left'].set_visible(False)
    bottom.tick_params(axis='y', length=0)
    bottom.set_xlabel('log-density ln p(x) (higher = more typical)')
    handles = []
    for direction, color in DIRECTION_COLORS.items():
        handles.append(plt.Line2D([], [], color=color, marker='o', linewidth=2, label=DIRECTION_NAMES[direction]))
    bottom.legend(handles=handles, frameon=False, loc='lower left', fontsize=8)
    figure.text(0.01, 0.005, 'dot: median request (percentile in brackets); bar: middle 50 % of the host\'s training '
                'requests. x-axis starts at the 1 % quantile of the training requests', color=MUTED, fontsize=8)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = os.path.join(OUT_DIR, f'targets_density_{layer}{unit_suffix(unit)}.png')
    figure.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    print(out_path)


# every target in the fig2 style: one histogram of all train requests, below it one strip per target, grouped by
# bucket and direction
def plot_all_targets(saved, targets, train_units, layer, unit):
    reference = saved['train_log_density']
    ranges = log_density_buckets(reference)

    groups = []
    for bucket in list(BUCKETS) + ['gap']:
        for direction in DIRECTIONS:
            in_group = targets[(targets['median_percentile'].map(bucket_or_gap) == bucket)
                               & (targets['direction'] == direction)]
            if len(in_group) > 0:
                groups.append((bucket, direction, in_group.sort_values('median_percentile')))

    # strip positions from the top, an extra gap between groups
    ordered = []
    positions = []
    group_spans = []
    position = 0.0
    for bucket, direction, in_group in groups:
        first = position
        for _, target in in_group.iterrows():
            ordered.append(target)
            positions.append(position)
            position += 1.0
        group_spans.append((bucket, first, position - 1.0))
        position += 0.6
    bottom_end = position - 0.6

    unit_log_densities = []
    for target in ordered:
        unit_log_densities.append(reference[train_units == target['unit']])
    bins = target_bins(reference, np.concatenate(unit_log_densities))

    strips_height = 0.24 * bottom_end
    figure, (axis, strips) = plt.subplots(2, 1, figsize=(7.5, 2.4 + strips_height), sharex=True,
                                          gridspec_kw={'height_ratios': [1.8, strips_height]})
    draw_all_requests(axis, bins, reference)
    shade_buckets(axis, ranges)
    axis.set_title('Candidate targets on the training log-density', color=INK, loc='left', pad=22)

    labels = []
    for target, position in zip(ordered, positions):
        color = DIRECTION_COLORS[target['direction']]
        unit_log_density = reference[train_units == target['unit']]
        draw_target_strip(strips, bins, unit_log_density, color, bottom=position - 0.4, height=0.8)
        labels.append(f'{target["unit"]}  ({target["median_percentile"]:.3f})')
    strips.set_ylim(bottom_end + 0.6, -0.6)
    shade_buckets(strips, ranges, labels=False)
    strips.set_yticks(positions)
    strips.set_yticklabels(labels, fontsize=8, color=INK)
    strips.tick_params(axis='y', length=0)
    strips.spines['left'].set_visible(False)
    for bucket, first, last in group_spans:
        strips.text(1.0, (first + last) / 2, ' ' + bucket.replace('_', ' '), transform=strips.get_yaxis_transform(),
                    ha='left', va='center', fontsize=8, color=MUTED)
    strips.set_xlabel('log-density ln p(x) (higher = more typical)')

    handles = [plt.Rectangle((0, 0), 1, 1, color=BAND, label='all training requests')]
    for direction, color in DIRECTION_COLORS.items():
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color,
                                     label=f'{DIRECTION_NAMES[direction]} host: bins with at least one of its requests'))
    # vertical marker, like the median ticks in the strips
    handles.append(plt.Line2D([], [], color=INK, marker='|', linestyle='None', markersize=10, markeredgewidth=1.5,
                              label='host\'s median request (its percentile in brackets)'))
    axis.legend(handles=handles, frameon=False, loc='upper left', fontsize=8)
    figure.text(0.01, 0.005, 'x-axis starts at the 1 % quantile of the training requests',
                color=MUTED, fontsize=8)
    figure.tight_layout(rect=(0, 0.03, 1, 1), h_pad=0.4)
    out_path = os.path.join(OUT_DIR, f'all_targets_{layer}{unit_suffix(unit)}.png')
    figure.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    print(out_path)


# median_percentile of every target under the other density settings (rank tables), plus Spearman over all units
def plot_stability(targets, layer, unit):
    tables = {}
    for label, suffix in STABILITY_SETTINGS:
        path = os.path.join(OUT_DIR, f'{unit}s_{layer}{suffix}.csv')
        if os.path.exists(path):
            tables[label] = pd.read_csv(path).set_index('unit')
    default = tables['PCA 16']

    spearman_rows = []
    for label, table in tables.items():
        row = {'setting': label.replace('\n', ' ')}
        for name, units in [('all', default.index),
                            ('ad_tracker', default.index[default['target_label'] == 0.0]),
                            ('benign', default.index[default['target_label'] == 1.0])]:
            common = units.intersection(table.index)
            first = default.loc[common, 'median_percentile']
            second = table.loc[common, 'median_percentile']
            row[f'spearman_{name}'] = float(first.rank().corr(second.rank()))
            row[f'units_{name}'] = len(common)
        spearman_rows.append(row)
    spearman = pd.DataFrame(spearman_rows)
    csv_path = os.path.join(OUT_DIR, f'stability_{layer}{unit_suffix(unit)}.csv')
    spearman.to_csv(csv_path, index=False)
    print(spearman.round(3).to_string(index=False))
    print(csv_path)

    settings = list(tables.keys())
    figure, axis = plt.subplots(figsize=(8, 4.5))
    end_values = []
    for _, target in targets.iterrows():
        values = []
        for label in settings:
            values.append(tables[label].loc[target['unit'], 'median_percentile'])
        color = DIRECTION_COLORS[target['direction']]
        axis.plot(range(len(settings)), values, color=color, linewidth=2, marker='o', markersize=5,
                  markeredgecolor=SURFACE, markeredgewidth=1)
        end_values.append((values[-1], target['unit']))
    axis.set_xlim(-0.3, len(settings) - 1 + 0.3)
    axis.set_ylim(0, 1)
    shade_buckets(axis, BUCKETS, horizontal=True, labels=False)
    axis.set_xticks(range(len(settings)))
    axis.set_xticklabels(settings)
    axis.grid(axis='y', color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_ylabel('percentile of the median request')

    # direct labels at the right end, pushed apart so they don't overlap
    end_values.sort()
    previous = -1.0
    for value, name in end_values:
        position = max(value, previous + 0.04)
        axis.text(len(settings) - 1 + 0.35, position, name, va='center', fontsize=7, color=INK)
        previous = position
    for bucket, (low, high) in BUCKETS.items():
        axis.text(-0.25, (low + high) / 2, bucket.replace('_', ' '), rotation=90, ha='center', va='center',
                  fontsize=7, color=MUTED)
    handles = []
    for direction, color in DIRECTION_COLORS.items():
        handles.append(plt.Line2D([], [], color=color, marker='o', linewidth=2, label=DIRECTION_NAMES[direction]))
    axis.legend(handles=handles, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    axis.set_title('Candidate targets under other density settings (2 components unless noted)', color=INK,
                   loc='left', pad=30)
    figure.tight_layout()
    out_path = os.path.join(OUT_DIR, f'stability_{layer}{unit_suffix(unit)}.png')
    figure.savefig(out_path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    print(out_path)


# Fig. 2 analogue
def plot_fig2(reference, target, train_units, layer):
    unit_log_density = reference[train_units == target.unit]
    bins = target_bins(reference, unit_log_density)
    color = DIRECTION_COLORS[target.direction]
    median = float(np.median(unit_log_density))
    bucket_ranges = log_density_buckets(reference)

    figure, (axis, strip) = plt.subplots(2, 1, figsize=(6, 3.5), sharex=True,
                                         gridspec_kw={'height_ratios': [24, 1]})
    reference_count = draw_all_requests(axis, bins, reference)
    # bins of all train requests that hold at least one target request, in the target's colour
    has_unit_rows = np.histogram(unit_log_density, bins=bins)[0] > 0
    unit_bins_count = np.where(has_unit_rows, reference_count, 0)
    axis.stairs(unit_bins_count, bins, fill=True, color=color, label=f'bins with at least one {target.unit} request')
    axis.axvline(median, color=INK, linewidth=1.5,
                 label=f'median request (percentile {target.median_percentile:.3f})')
    shade_buckets(axis, bucket_ranges)
    axis.legend(frameon=False, loc='upper left', fontsize=8)

    draw_target_strip(strip, bins, unit_log_density, color)
    strip.set_ylim(0, 1)
    shade_buckets(strip, bucket_ranges, labels=False)
    strip.set_yticks([])
    strip.set_xlabel('log-density ln p(x)')
    figure.tight_layout(h_pad=0.3)
    plot_path = os.path.join(PLOT_DIR, os.path.basename(target_file_prefix(target.unit, layer)) + '_fig2.png')
    figure.savefig(plot_path, dpi=DPI, facecolor=SURFACE)
    plt.close(figure)
    print(plot_path)


def main():
    parser = argparse.ArgumentParser(description='p-edge-case example set (Wang et al. 2020, Sec. 2)')
    parser.add_argument('--layer', choices=['hidden', 'logit'], default='hidden')
    parser.add_argument('--unit', choices=['host', 'host_path'], default='host', help='the target unit')
    parser.add_argument('--pca', type=int, default=PCA_DIMENSIONS, help='PCA dimensions before the mixture')
    parser.add_argument('--components', type=int, default=N_COMPONENTS, help='Gaussian mixture components')
    parser.add_argument('--seed', type=int, default=SEED, help='PCA rows and mixture initialisation')
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_all(args.layer, args.pca, args.components, args.seed, args.unit)


if __name__ == '__main__':
    main()
