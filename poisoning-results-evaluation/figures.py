# every evaluation figure / table, ids from the thesis figure list (M = must, S = should)
# usage: python figures.py [results_dir] [--only m4,s1,attacker_count_curves]  -> plots/
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
# no GUI backend: faster, and safe in forked workers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
import seaborn as sns
import load as load_module
from load import load, per_round, save, save_table, N_CLIENTS, RESULTS_DIR


# facet titles without the 'name = ' prefix, legend entries with the thesis names
def save_grid(grid, name, title, titles=True, ncol=None):
    if titles:
        grid.set_titles(row_template='{row_name}', col_template='{col_name}')
    if grid.legend:
        # one row below the x labels, as in the hand-built figures; save's bbox_inches='tight' crops the old right margin;
        # ncol: wrap into several rows when one row is wider than the panels
        sns.move_legend(grid, 'upper center', bbox_to_anchor=(0.5, 0), ncol=ncol or len(grid.legend.legend_handles))
        for text in grid.legend.texts + [grid.legend.get_title()]:
            text.set_text(NAMES.get(text.get_text(), text.get_text()))
        # group headers of a legend with several variables (seaborn adds them as entries with an empty handle):
        # drop the handle space so the header starts left of its entries
        for column in grid.legend._legend_handle_box.get_children():
            for row in column.get_children():
                handle_box, text_box = row.get_children()
                if text_box.get_children()[0].get_text() in LEGEND_HEADERS:
                    handle_box.width = 0
                    row.sep = 0
    save(grid.figure, name, title)

# runs of other partitions are dropped (dirichlet_0.5 skipped)
# result name -> name shown in figures and tables
PARTITION_NAMES = {'iid': 'IID', 'natural': 'Natural', 'dirichlet_1.0': 'Dirichlet (α = 1.0)'}
PARTITIONS = list(PARTITION_NAMES.values())
LABELS = {'drop_f1': 'test F1 drop (pp)', 'rise_fpr': 'test FPR rise (pp)'}
# names in the thesis text, for legends, facet titles and heatmap rows (file names keep the result names)
NAMES = {
    'EPPRFL': 'F&C', 'aggregator': 'Rule', 'host': 'Target', 'partition': 'Partition', 'name': 'Attack', 'malicious': 'Malicious clients',
    'SignFlipping': 'Sign flipping', 'Gaussian': 'Gaussian noise', 'FangKrum': 'Fang (Krum)', 'MinMax': 'Min-Max (unit)',
    'MinMax-std': 'Min-Max (std)', 'MinSum': 'Min-Sum (unit)',
    'MR-naive': 'Naive', 'MR-replacement': 'Model replacement', 'MR-train_scale_max': 'Train-and-scale',
    'MR-constrain_scale_cosine': 'Constrain-and-scale', 'Tails-pgd': 'Edge-case (PGD)',
    'ad-delivery.net': 'common target', 'stats.g.doubleclick.net': 'rare target',
}
# legend group headers (variable names after NAMES), drawn without the empty handle space in save_grid
LEGEND_HEADERS = {'Rule', 'Target', 'Partition', 'Attack', 'Malicious clients'}
# same order as the thesis tables
UNTARGETED_ORDER = ['Gaussian noise', 'Sign flipping', 'Fang (Krum)', 'Min-Max (unit)', 'Min-Max (std)', 'Min-Sum (unit)']
BACKDOOR_ORDER = ['Naive', 'Model replacement', 'Train-and-scale', 'Constrain-and-scale', 'Edge-case (PGD)']
# the same colors as the attacker count figures, FedAvg first in every legend
RULE_COLORS = {'FedAvg': '#2a78d6', 'F&C': '#eb6834'}


def pretty(data):
    return data.replace({column: NAMES for column in ['aggregator', 'name', 'host']})


# False: skip every backdoor figure / table
BACKDOORS = True
# parallel figure functions; each worker slowly copies the loaded runs, so mind RAM
WORKERS = 16

parser = argparse.ArgumentParser()
parser.add_argument('results_dir', nargs='?', default=RESULTS_DIR)
# comma-separated function names or their id prefixes (m4, s1); all if unset
parser.add_argument('--only')
arguments = parser.parse_args()
runs = load(arguments.results_dir)
runs = runs[runs['partition'].isin(list(PARTITION_NAMES))].copy()
runs['partition'] = runs['partition'].map(PARTITION_NAMES)
clean = runs[runs['n_malicious'] == 0]
untargeted = runs[(runs['n_malicious'] > 0) & runs['host'].isna()]
# every-round grid (attacks in rounds 1-50 of 70); the old every_k runs (every 10th round of 100) are left out
backdoors = runs[runs['host'].notna() & (runs['schedule'] == 'every')] if BACKDOORS else runs.iloc[:0]


# M1: clean runs per partition and length
def m1_clean_table():
    table = clean.pivot_table(
        index=['n_rounds', 'partition'], columns='aggregator',
        values=['f1', 'precision', 'recall', 'fpr', 'benign_rejected'],
    )
    save_table(table, 'm1_clean')


# M2: test F1 over rounds without attack
def m2_clean_curves():
    # the 50-round runs are the first 50 rounds of the 100-round runs (deterministic), so only the latter are shown
    data = pretty(per_round(clean[clean['n_rounds'] == 100], 'f1', 'fpr', 'n_benign_rejected'))
    data['benign_rejected_of_all'] = data['n_benign_rejected'] * 100 / N_CLIENTS
    top = data['benign_rejected_of_all'].max() * 1.1
    for metric in ['f1', 'fpr']:
        # curve row, then F&C's honest rejections as a bar strip, as in the attacker count figures
        fig, axes = plt.subplots(2, len(PARTITIONS), figsize=(3.6 * len(PARTITIONS), 3.6), sharex=True, sharey='row',
                                 gridspec_kw={'height_ratios': [3, 1]})
        for column, partition in enumerate(PARTITIONS):
            part = data[data['partition'] == partition]
            for aggregator, color in RULE_COLORS.items():
                line = part[part['aggregator'] == aggregator]
                axes[0, column].plot(line['round'], line[metric], color=color, label=aggregator)
            axes[0, column].set_title(partition)
            line = part[part['aggregator'] == 'F&C']
            # no edge: seaborn's white bar edges hide the thin bars of the 100-round runs
            axes[1, column].bar(line['round'], line['benign_rejected_of_all'], width=1, linewidth=0, color='#1baf7a',
                                label='F&C honest rejected')
            axes[1, column].set_ylim(0, top)
            axes[1, column].set_xlabel('round')
        axes[0, 0].set_ylabel(f'test {metric.upper()}')
        axes[1, 0].set_ylabel('%')
        handles = {}
        for ax in fig.axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles[label] = handle
        fig.tight_layout()
        # small gap between a curve and its strip
        fig.subplots_adjust(hspace=0.08)
        # one row below the x labels; save's bbox_inches='tight' keeps it in the file
        fig.legend(handles.values(), handles.keys(), loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(handles))
        save(fig, f'm2_clean_{metric}', f'Test {metric.upper()} over rounds without attack')
        plt.close(fig)


# M3: drop in test F1 and rise in FPR (pp), worst partition
def m3_untargeted_table():
    for value in ['drop_f1', 'rise_fpr']:
        table = untargeted.pivot_table(index='name', columns=['n_malicious', 'aggregator'], values=value, aggfunc='max')
        save_table(table.round(2), f'm3_untargeted_{value}')


# T1: one LaTeX tabular per untargeted attack for the thesis (\input in its subsection):
# one row per share; per partition the drop of both rules and EPPRFL's detection
def t1_untargeted_tables():
    def number(value, bold=False):
        # + 0.0: no "-0.0"
        text = f'{round(value, 1) + 0.0:.1f}'.replace('-', '$-$')
        return f'\\textbf{{{text}}}' if bold else text

    folder = os.path.join(load_module.PLOTS_DIR, 'tables')
    os.makedirs(folder, exist_ok=True)
    for name in untargeted['name'].unique():
        attack = untargeted[untargeted['name'] == name].copy()
        attack['percent'] = (attack['share'] * 100).round().astype(int)
        drop = attack.pivot_table(index=['partition', 'aggregator'], columns='percent', values='drop_f1')
        detection = attack[attack['aggregator'] == 'EPPRFL'].pivot_table(index='partition', columns='percent',
                                                                          values='detection_recall') * 100
        labels = ['\\gls{iid}' if partition == 'IID' else partition.split(' ')[0] for partition in PARTITIONS]
        lines = ['\\begin{tabular}{r' + 'rrr' * len(PARTITIONS) + '}', '\\toprule',
                 ' & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{label}}}' for label in labels) + ' \\\\',
                 ' '.join(f'\\cmidrule(lr){{{2 + 3 * index}-{4 + 3 * index}}}' for index in range(len(PARTITIONS))),
                 'Share' + ' & \\gls{fedavg} & \\gls{fc} & Rej.' * len(PARTITIONS) + ' \\\\', '\\midrule']
        for share in drop.columns:
            cells = []
            for partition in PARTITIONS:
                fedavg, epprfl = drop.loc[(partition, 'FedAvg'), share], drop.loc[(partition, 'EPPRFL'), share]
                # bold: EPPRFL drops more than FedAvg (a miss of G2)
                cells += [number(fedavg), number(epprfl, epprfl > fedavg), f'{detection.loc[partition, share]:.0f}']
            lines.append(f'{share}\\,\\% & ' + ' & '.join(cells) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}']
        with open(os.path.join(folder, f'untargeted_{name}.tex'), 'w') as table_file:
            table_file.write('\n'.join(lines) + '\n')


# T2: one LaTeX table per backdoor: highest test ASR (%) of FedAvg / EPPRFL and the attackers EPPRFL rejects (%),
# rows: target x share, column groups: partitions; bold: EPPRFL above the bound of G3.
# Model replacement and constrain-and-scale have none: 100 / 0 / 100 in every setting, stated in the text
def t2_targeted_tables():
    folder = os.path.join(load_module.PLOTS_DIR, 'tables')
    os.makedirs(folder, exist_ok=True)
    data = backdoors.copy()
    data['percent'] = data['share'] * 100
    labels = ['\\gls{iid}' if partition == 'IID' else partition.split(' ')[0] for partition in PARTITIONS]
    for name in ['MR-naive', 'MR-train_scale_max', 'Tails-pgd']:
        lines = ['\\begin{tabular}{ll' + 'rrr' * len(PARTITIONS) + '}', '\\toprule',
                 ' & & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{label}}}' for label in labels) + ' \\\\',
                 ' '.join(f'\\cmidrule(lr){{{3 + 3 * index}-{5 + 3 * index}}}' for index in range(len(PARTITIONS))),
                 'Target & Share' + ' & \\gls{fedavg} & \\gls{fc} & Rej.' * len(PARTITIONS) + ' \\\\']
        for host in ['stats.g.doubleclick.net', 'ad-delivery.net']:
            attack = data[(data['name'] == name) & (data['host'] == host)]
            peak = attack.pivot_table(index=['partition', 'aggregator'], columns='percent', values='peak_asr') * 100
            detection = attack[attack['aggregator'] == 'EPPRFL'].pivot_table(index='partition', columns='percent',
                                                                              values='detection_recall') * 100
            lines.append('\\midrule')
            for index, share in enumerate(peak.columns):
                cells = [NAMES[host].split(' ')[0] if index == 0 else '', f'{share:g}\\,\\%']
                for partition in PARTITIONS:
                    fedavg, epprfl = peak.loc[(partition, 'FedAvg'), share], peak.loc[(partition, 'EPPRFL'), share]
                    # bold: EPPRFL above the bound of G3 (10 %)
                    text = f'{epprfl:.0f}'
                    cells += [f'{fedavg:.0f}', f'\\textbf{{{text}}}' if epprfl > 10 else text,
                              f'{detection.loc[partition, share]:.0f}']
                lines.append(' & '.join(cells) + ' \\\\')
        lines += ['\\bottomrule', '\\end{tabular}']
        with open(os.path.join(folder, f'targeted_{name}.tex'), 'w') as table_file:
            table_file.write('\n'.join(lines) + '\n')


# M4: drop vs malicious share, one row per partition, one column per attack
def m4_drop_vs_share():
    data = pretty(untargeted)
    data['percent'] = data['share'] * 100
    for value in ['drop_f1', 'rise_fpr']:
        # own y scale per attack: Gaussian noise under FedAvg (up to 80 pp) would flatten the other columns
        grid = sns.relplot(data=data, x='percent', y=value, hue='aggregator', row='partition', col='name',
                           kind='line', marker='o', height=2.2, palette=RULE_COLORS,
                           hue_order=list(RULE_COLORS), row_order=PARTITIONS, col_order=UNTARGETED_ORDER,
                           facet_kws={'sharey': 'col', 'margin_titles': True})
        for ax in grid.axes.flat:
            ax.set_xticks([5, 10, 20, 30, 40])
        grid.set_axis_labels('malicious clients (%)', LABELS[value])
        # the inner x tick labels are hidden, so the rows can sit close
        grid.figure.subplots_adjust(hspace=0.15)
        save_grid(grid, f'm4_untargeted_{value}_vs_share', f'Untargeted attacks: {LABELS[value]} vs malicious share')


# M5: highest ASR from the first attack round, worst partition and schedule; then per schedule; floor separately
def m5_backdoor_table():
    table = backdoors.pivot_table(index=['host', 'name'], columns=['n_malicious', 'aggregator'], values='peak_asr', aggfunc='max')
    save_table(table.round(3), 'm5_backdoor_peak_asr')
    table = backdoors.pivot_table(index=['host', 'name', 'schedule'], columns=['n_malicious', 'aggregator'], values='peak_asr', aggfunc='max')
    save_table(table.round(3), 'm5_backdoor_peak_asr_by_schedule')
    table = backdoors.pivot_table(index='host', columns='aggregator', values='floor_asr', aggfunc='max')
    save_table(table.round(3), 'm5_backdoor_floor_asr')


# M6: test ASR over rounds, one figure per host and schedule
def m6_backdoor_curves():
    data = per_round(backdoors, 'attack_success_rate')
    data['attack_success_rate'] *= 100
    data['malicious'] = (data['n_malicious'] * 100 / N_CLIENTS).map('{:g} %'.format)
    shares = sorted(data['n_malicious'].unique())
    for (host, schedule), part in data.groupby(['host', 'schedule']):
        grid = sns.relplot(data=pretty(part), x='round', y='attack_success_rate', hue='aggregator', style='malicious',
                           row='partition', col='name', kind='line', height=1.8, aspect=1.5, palette=RULE_COLORS,
                           hue_order=list(RULE_COLORS), style_order=[f'{n * 100 / N_CLIENTS:g} %' for n in shares],
                           row_order=PARTITIONS, col_order=BACKDOOR_ORDER)
        for ax in grid.axes.flat:
            ax.set_ylim(-2, 102)
            ax.set_xticks(range(0, 71, 10))
        grid.set_axis_labels('round', '')
        # backdoor on top, partition as the y label: full 'row | column' titles overlap
        for (row, column), ax in np.ndenumerate(grid.axes):
            ax.set_title(BACKDOOR_ORDER[column] if row == 0 else '')
        for row, partition in enumerate(PARTITIONS):
            grid.axes[row, 0].set_ylabel(partition, fontsize='small')
        save_grid(grid, f'm6_asr_{host}_{schedule}', f'Backdoor success over rounds, target {host}, schedule {schedule}', titles=False)


# M8: highest test ASR from the first attack round vs malicious share, worst partition, one panel per target
def m8_backdoor_peak_vs_share():
    # top row: worst partition, bottom row: mean over the partitions
    # model replacement and constrain-and-scale give the same results (FedAvg 100 %, F&C 0 %, at most 0.3 pp apart):
    # one line for both, else the second one hides the first
    merged = 'Model replacement / constrain-and-scale'
    data = pretty(backdoors)
    data = data[data['name'] != 'Constrain-and-scale'].replace({'name': {'Model replacement': merged}})
    order = [merged if name == 'Model replacement' else name for name in BACKDOOR_ORDER if name != 'Constrain-and-scale']
    groups = data.groupby(['host', 'name', 'aggregator', 'share'])['peak_asr']
    data = pd.concat([groups.max().reset_index().assign(over='worst partition'),
                      # bottom row off for now: the mean over partitions shows the same blind spot as the top row
                      # groups.mean().reset_index().assign(over='mean over partitions'),
                      ])
    data['peak_asr'] *= 100
    # evenly spaced shares (0.5 % and 5 % would overlap on a linear axis)
    shares = sorted(data['share'].unique())
    data['position'] = data['share'].map(shares.index)
    # dodge: shift each attack a little to the side of its share, so points at the same value don't hide each other
    dodge = 0.12
    data['position'] += data['name'].map({name: (index - (len(order) - 1) / 2) * dodge for index, name in enumerate(order)})
    # no blue / orange: those are the rule colors elsewhere; each attack keeps its color from before the merge
    colors = dict(zip(BACKDOOR_ORDER, sns.color_palette('tab10')[2:7]))
    colors[merged] = colors['Model replacement']
    grid = sns.relplot(data=data, x='position', y='peak_asr', hue='name', style='aggregator', col='host',
                       # row='over', row_order=['worst partition', 'mean over partitions'], facet_kws={'margin_titles': True},
                       # wider panels: the wrapped legend below needs about this width
                       kind='line', markers=True, height=2.6, aspect=1.35, hue_order=order, style_order=list(RULE_COLORS),
                       palette={name: colors[name] for name in order})
    for ax in grid.axes.flat:
        ax.set_ylim(-2, 102)
        ax.set_xticks(range(len(shares)), [f'{share * 100:g}' for share in shares])
        ax.axhline(10, color='gray', linestyle=':')
    # one y label for both rows: two stacked copies overlap
    grid.set_axis_labels('malicious clients (%)', '')
    grid.figure.supylabel('highest attack success (%)', x=0.02)
    # drop the right margin seaborn keeps for its side legend, else the legend below is centered off the panels
    grid.figure.subplots_adjust(right=0.98)
    # two panels only: 3 columns (attack header + 2 names, 2 names + empty spacer, rule header + 2 rules);
    # the spacer keeps the rule header at the top of its column (legends fill column by column)
    handles = list(grid.legend.legend_handles)
    labels = [text.get_text() for text in grid.legend.texts]
    spacer = labels.index('aggregator')
    handles.insert(spacer, matplotlib.lines.Line2D([], [], alpha=0))
    labels.insert(spacer, '')
    grid.legend.remove()
    grid._legend = grid.figure.legend(handles, labels, frameon=False)
    save_grid(grid, 'm8_backdoor_peak_vs_share', 'Backdoors: highest test success from the first attack round', ncol=3)


# M8 as heatmaps: rows attack x rule, columns share, one panel per target and worst partition / mean over partitions
def m8b_backdoor_peak_heatmap():
    data = pretty(backdoors)
    data['percent'] = (data['share'] * 100).map('{:g}'.format)
    data['row'] = data['name'] + ', ' + data['aggregator']
    rows = [f'{name}, {rule}' for name in BACKDOOR_ORDER for rule in RULE_COLORS]
    columns = [f'{share * 100:g}' for share in sorted(data['share'].unique())]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
    for column, host in enumerate(['common target', 'rare target']):
        for row, (over, aggfunc) in enumerate([('worst partition', 'max'), ('mean over partitions', 'mean')]):
            table = data[data['host'] == host].pivot_table(index='row', columns='percent', values='peak_asr',
                                                           aggfunc=aggfunc).reindex(index=rows, columns=columns) * 100
            # same color scale everywhere; empty cells: train-and-scale does not run with one attacker
            sns.heatmap(table, annot=True, fmt='.0f', cmap='rocket_r', vmin=0, vmax=100, ax=axes[row, column],
                        cbar=False, annot_kws={'fontsize': 'x-small'}, linewidths=0.5)
            # whitegrid's lines show through the empty cells
            axes[row, column].grid(False)
            axes[row, column].set_title(f'{host}, {over}')
            axes[row, column].set_xlabel('malicious clients (%)' if row == 1 else '')
            axes[row, column].set_ylabel('')
            # a line between the attacks, so each FedAvg / F&C pair reads as one block
            for line in range(2, len(rows), 2):
                axes[row, column].axhline(line, color='white', linewidth=3)
    fig.tight_layout()
    save(fig, 'm8b_backdoor_peak_heatmap', 'Backdoors: highest test attack success (%)')


# M9: final test F1 drop under backdoors vs malicious share, per partition and backdoor, both targets
def m9_backdoor_drop_vs_share():
    data = pretty(backdoors)
    # evenly spaced shares, as in m8
    shares = sorted(data['share'].unique())
    data['position'] = data['share'].map(shares.index)
    grid = sns.relplot(data=data, x='position', y='drop_f1', hue='aggregator', style='host', row='partition',
                       col='name', kind='line', markers=True, height=1.8, aspect=1.2, palette=RULE_COLORS,
                       hue_order=list(RULE_COLORS), style_order=['common target', 'rare target'],
                       row_order=PARTITIONS, col_order=BACKDOOR_ORDER, facet_kws={'margin_titles': True})
    for ax in grid.axes.flat:
        ax.set_xticks(range(len(shares)), [f'{share * 100:g}' for share in shares])
        ax.axhline(0, color='gray', linewidth=0.8)
    grid.set_axis_labels('malicious clients (%)', '')
    # one y label in the middle row: three stacked copies overlap
    grid.axes[1, 0].set_ylabel(LABELS['drop_f1'])
    grid.figure.subplots_adjust(hspace=0.15)
    save_grid(grid, 'm9_backdoor_drop_f1_vs_share', 'Backdoors: test F1 drop vs malicious share')


# M7: noise estimate for the "same level" band: std over the last 10 rounds of the clean runs (pp)
def m7_band_table():
    rows = []
    for run in clean.itertuples():
        last = run.rounds[-10:]
        rows.append({
            'aggregator': run.aggregator, 'partition': run.partition, 'n_rounds': run.n_rounds,
            'f1_std_pp': np.std([result['f1'] for result in last]) * 100,
            'fpr_std_pp': np.std([result['fpr'] for result in last]) * 100,
        })
    save_table(pd.DataFrame(rows).set_index(['aggregator', 'partition', 'n_rounds']).round(3), 'm7_band')


def heatmaps(data, value, name, title, center=None, fmt='.1f'):
    data = pretty(data)
    data['percent'] = (data['share'] * 100).round().astype(int)
    aggregators = sorted(data['aggregator'].unique())
    # one row per partition, one column per aggregator
    fig, axes = plt.subplots(len(PARTITIONS), len(aggregators), figsize=(3.6 * len(aggregators), 2.4 * len(PARTITIONS)),
                             squeeze=False, sharey=True)
    for column, aggregator in enumerate(aggregators):
        for row, partition in enumerate(PARTITIONS):
            part = data[(data['aggregator'] == aggregator) & (data['partition'] == partition)]
            table = part.pivot_table(index='name', columns='percent', values=value).reindex(UNTARGETED_ORDER)
            sns.heatmap(table, annot=True, fmt=fmt, cmap='RdBu_r' if center is not None else 'viridis',
                        center=center, ax=axes[row, column], cbar=False)
            axes[row, column].set_title(f'{aggregator}, {partition}' if len(aggregators) > 1 else partition)
            axes[row, column].set_xlabel('malicious clients (%)' if row == len(PARTITIONS) - 1 else '')
            axes[row, column].set_ylabel('')
    fig.tight_layout()
    save(fig, name, title)


# S1: drop for every untargeted setting
def s1_untargeted_heatmaps():
    heatmaps(untargeted, 'drop_f1', 's1_heatmap_drop_f1', 'Untargeted attacks: test F1 drop (pp) vs own clean run', center=0)
    heatmaps(untargeted, 'rise_fpr', 's1_heatmap_rise_fpr', 'Untargeted attacks: test FPR rise (pp) vs own clean run', center=0)


# S2: detection under untargeted attacks (aggregators that reject; FedAvg rejects nobody)
def s2_detection_heatmaps():
    filtering = untargeted[untargeted['aggregator'] != 'FedAvg'].copy()
    filtering['detection_recall_pct'] = filtering['detection_recall'] * 100
    filtering['benign_rejected_pct'] = filtering['benign_rejected'] * 100
    # 50 attack rounds: whole percent
    heatmaps(filtering, 'detection_recall_pct', 's2_heatmap_detection_recall', 'Untargeted attacks: attackers rejected (%), mean over attack rounds', fmt='.0f')
    heatmaps(filtering, 'benign_rejected_pct', 's2_heatmap_benign_rejected', 'Untargeted attacks: honest clients rejected (%), mean over attack rounds')


# S2: attackers rejected vs honest rejected, one point per share and partition; top left is ideal
def s2_detection_tradeoff():
    filtering = untargeted[untargeted['aggregator'] != 'FedAvg'].copy()
    filtering['detection_recall_pct'] = filtering['detection_recall'] * 100
    filtering['benign_rejected_pct'] = filtering['benign_rejected'] * 100
    grid = sns.relplot(data=filtering, x='benign_rejected_pct', y='detection_recall_pct', hue='share', style='partition',
                       col='name', col_wrap=3, kind='scatter', palette='viridis', s=60, height=2.8)
    grid.set_axis_labels('honest rejected (%)', 'attackers rejected (%)')
    save_grid(grid, 's2_detection_tradeoff', 'Untargeted attacks: attackers vs honest clients rejected, mean over attack rounds')


# S3: untargeted F1 over rounds at 20 % and 40 %, the matched clean run dashed
def s3_untargeted_curves():
    for n_malicious in [40, 80]:
        attacked = per_round(untargeted[untargeted['n_malicious'] == n_malicious], 'f1')
        attacked['run'] = 'attacked'
        clean_rounds = per_round(clean[clean['n_rounds'] == 50], 'f1')
        copies = []
        for name in attacked['name'].unique():
            copy = clean_rounds.copy()
            copy['name'] = name
            copies.append(copy)
        clean_copies = pd.concat(copies)
        clean_copies['run'] = 'clean'
        grid = sns.relplot(data=pd.concat([attacked, clean_copies]), x='round', y='f1', hue='aggregator', style='run',
                           row='partition', col='name', kind='line', height=2.2, aspect=1.4)
        save_grid(grid, f's3_untargeted_curves_{n_malicious * 100 // N_CLIENTS}pct', f'Untargeted attacks: test F1 over rounds, {n_malicious * 100 // N_CLIENTS} % malicious (dashed: no attack)')


# S4: non-target F1 drop under backdoors, worst partition and schedule
def s4_non_target_drop():
    grid = sns.catplot(data=backdoors, x='name', y='drop_non_target_f1', hue='aggregator', col='n_malicious', row='host',
                       kind='bar', estimator='max', errorbar=None, height=2.8)
    grid.tick_params(axis='x', rotation=60)
    save_grid(grid, 's4_backdoor_non_target_drop_f1', 'Backdoors: non-target test F1 drop (pp), worst partition and schedule')


# S5: test ASR at the last attack round and 20 rounds later, worst partition
def s5_persistence():
    data = backdoors.melt(id_vars=['aggregator', 'name', 'host', 'schedule', 'n_malicious', 'partition'],
                          value_vars=['asr_last_attack', 'asr_after_decay'], var_name='when', value_name='test_asr')
    for schedule, part in data.groupby('schedule'):
        grid = sns.catplot(data=part, x='when', y='test_asr', hue='aggregator', col='name', row='host',
                           kind='point', estimator='max', errorbar=None, height=2.5)
        save_grid(grid, f's5_persistence_{schedule}', f'Backdoors: test success at the last attack and 20 rounds later, schedule {schedule}')


# S6: peak ASR on the rare host vs on the common host, same setting
def s6_rare_vs_common():
    table = backdoors.pivot_table(index=['aggregator', 'name', 'schedule', 'partition', 'n_malicious'],
                                  columns='host', values='peak_asr').reset_index()
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(data=table, x='ad-delivery.net', y='stats.g.doubleclick.net', hue='aggregator', style='name', ax=ax)
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':')
    ax.set_xlabel('peak ASR, common host (ad-delivery.net)')
    ax.set_ylabel('peak ASR, rare host (stats.g.doubleclick.net)')
    save(fig, 's6_rare_vs_common', 'Backdoors: peak success, rare vs common target')


# S9: what EPPRFL does over rounds, clean and 20 % untargeted
def s9_epprfl_internals():
    epprfl = runs[(runs['aggregator'] == 'EPPRFL') & runs['host'].isna() & runs['n_malicious'].isin([0, 40]) & (runs['n_rounds'] == 50)]
    data = per_round(epprfl, 'epprfl_m_h', 'epprfl_n_benign', 'epprfl_n_clipped')
    data = data.melt(id_vars=['name', 'partition', 'round'], value_vars=['epprfl_m_h', 'epprfl_n_benign', 'epprfl_n_clipped'])
    grid = sns.relplot(data=data, x='round', y='value', hue='name', row='partition', col='variable', kind='line',
                       height=2.5, aspect=1.4, facet_kws={'sharey': 'col'})
    save_grid(grid, 's9_epprfl_internals', 'EPPRFL over rounds: no attack and 20 % untargeted')


# S12: clean miss rate of the candidate hosts (the ASR floor), 100-round clean runs
def s12_host_floor():
    rows = []
    for run in clean[clean['n_rounds'] == 100].itertuples():
        for host in run.rounds[0]['clean_success_rates']:
            values = [result['clean_success_rates'][host] for result in run.rounds]
            rows.append({'aggregator': run.aggregator, 'partition': run.partition, 'host': host,
                         'max': max(values), 'final': values[-1]})
    save_table(pd.DataFrame(rows).pivot_table(index='host', columns=['aggregator', 'partition'], values=['max', 'final']).round(3), 's12_host_floor')


# S13: client sizes and positive rates per partition
def s13_partitions():
    one_per_partition = clean.drop_duplicates('partition')
    fig, axes = plt.subplots(len(one_per_partition), 2, figsize=(8, 2.5 * len(one_per_partition)), squeeze=False)
    for row, run in enumerate(one_per_partition.itertuples()):
        axes[row, 0].hist(run.client_sizes, bins=40)
        axes[row, 0].set_ylabel(run.partition)
        axes[row, 0].set_xlabel('rows per client')
        axes[row, 1].hist(run.client_positive_rates, bins=40)
        axes[row, 1].set_xlabel('ad share per client')
    fig.tight_layout()
    save(fig, 's13_partitions', 'Client partitions: rows and ad share per client')


# one figure per attack (backdoors also x target x schedule): one row per partition, one column per attacker share;
# each panel: EPPRFL and FedAvg under the attack, FedAvg without attack as a black dashed baseline,
# plus the clients EPPRFL rejected per round, drawn in two versions (<name>_bar and <name>_plot).
# untargeted: test F1, backdoors: test ASR
REJECTION_FIELDS = ['n_benign_rejected', 'n_malicious_rejected', 'n_attackers_active']


# attacker share, and % of honest / of active attackers rejected per round (NaN where nobody attacks or no filter)
def rejection_percentages(data):
    data['percent'] = data['n_malicious'] * 100 / N_CLIENTS
    data['benign_rejected_pct'] = data['n_benign_rejected'] * 100 / (N_CLIENTS - data['n_malicious'])
    data['malicious_rejected_pct'] = data['n_malicious_rejected'] * 100 / data['n_attackers_active'].replace(0, np.nan)
    # the same counts as % of all clients, for the bar strips
    data['malicious_rejected_of_all'] = data['n_malicious_rejected'] * 100 / N_CLIENTS
    data['benign_rejected_of_all'] = data['n_benign_rejected'] * 100 / N_CLIENTS
    data['attackers_active_of_all'] = data['n_attackers_active'] * 100 / N_CLIENTS
    return data


# FedAvg first: F&C is drawn on top, else FedAvg's backdoor spikes hide F&C's where both reach them
ATTACKER_COUNT_COLORS = {'FedAvg': '#2a78d6', 'EPPRFL': '#eb6834'}


# bars: EPPRFL's rejected clients as a bar strip under each panel, y from 0 to top[share] (% of all clients);
# else as % lines on a right axis
def draw_attacker_counts(data, base, y, name, title, top, bars):
    shares = sorted(data['percent'].unique())
    if bars:
        # per partition: curve row, rejection strip row, then an empty spacer row before the next partition
        ratios = [4, 1, 0.4] * len(PARTITIONS)
        ratios = ratios[:-1]
        step = 3
        height = 2.4
    else:
        ratios = [1] * len(PARTITIONS)
        step = 1
        height = 2.2
    # compact panels with 'talk' fonts: the figure is shrunk to the text width in the thesis
    sns.set_context('talk')
    fig, axes = plt.subplots(len(ratios), len(shares), figsize=(3.6 * len(shares), height * len(PARTITIONS)),
                             sharex=True, squeeze=False, gridspec_kw={'height_ratios': ratios})
    for row, partition in enumerate(PARTITIONS):
        for column, share in enumerate(shares):
            ax = axes[step * row, column]
            if ax is not axes[0, 0]:
                ax.sharey(axes[0, 0])
            for aggregator, color in ATTACKER_COUNT_COLORS.items():
                line = data[(data['percent'] == share) & (data['partition'] == partition) & (data['aggregator'] == aggregator)]
                ax.plot(line['round'], line[y], color=color, label=NAMES.get(aggregator, aggregator))
            if base is None:
                # backdoors: the runs without attack are 0 % in every round, so no baseline is drawn;
                # instead the bound of G3 (10 pp above that, ASR here is 0..1)
                ax.axhline(0.1, color='#808080', linestyle='--', linewidth=1.5, zorder=1, label='G3 bound')
            elif base.empty and 'aggregator' in base.columns:
                # relative curves: each rule's own run without attack is the zero line
                ax.axhline(0, color='black', linewidth=1, zorder=1)
            elif 'aggregator' in base.columns:
                base_line = base[base['partition'] == partition]
                # each rule's own run without attack, the baseline its drop is measured against
                for aggregator, color in ATTACKER_COUNT_COLORS.items():
                    own = base_line[base_line['aggregator'] == aggregator]
                    # wide light band behind the curves: stays visible where the attacked curve lies on it
                    ax.plot(own['round'], own[y], color=color, linewidth=6, alpha=0.3, zorder=1,
                            solid_capstyle='butt', label=f'{NAMES.get(aggregator, aggregator)} no attack')
            else:
                base_line = base[base['partition'] == partition]
                ax.plot(base_line['round'], base_line[y], color='black', linestyle='--', linewidth=2, label='FedAvg, no attack')
            if row == 0:
                ax.set_title(f'{share:g} % attackers')
            # EPPRFL's rejections (FedAvg rejects nobody)
            line = data[(data['percent'] == share) & (data['partition'] == partition) & (data['aggregator'] == 'EPPRFL')]
            if bars:
                strip = axes[3 * row + 1, column]
                # no edge: seaborn's white bar edges hide the thin bars of the 100-round runs
                strip.bar(line['round'], line['malicious_rejected_of_all'], width=1, linewidth=0, color='#9b59b6',
                          label='rejected attackers')
                strip.bar(line['round'], line['benign_rejected_of_all'], width=1, linewidth=0, bottom=line['malicious_rejected_of_all'],
                          color='#1baf7a', label='rejected honest')
                # ticks, not a step line: in the backdoor runs a step outline hides the 1-round attacker bars
                active = line[line['attackers_active_of_all'] > 0]
                strip.plot(active['round'], active['attackers_active_of_all'], linestyle='none', marker='_',
                           markersize=6, markeredgewidth=1.5, color='black', label='attackers')
                strip.set_ylim(0, top[share])
                # one tick (the top one): "25" over "0" crowds the low strip
                strip.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=1, prune='lower'))
                # seaborn's 'talk' context sets a numeric size
                strip.tick_params(axis='y', labelsize=matplotlib.rcParams['ytick.labelsize'] * 0.9)
                if row < len(PARTITIONS) - 1:
                    axes[3 * row + 2, column].axis('off')
            else:
                rejections = ax.twinx()
                rejections.plot(line['round'], line['benign_rejected_pct'], color='#1baf7a', linestyle=':', linewidth=1.5,
                                label='F&C honest rejected (%, right)')
                rejections.plot(line['round'], line['malicious_rejected_pct'], color='#9b59b6', linestyle='-.', linewidth=1.2,
                                marker='v', markersize=3, label='F&C attackers rejected (%, right)')
                rejections.set_ylim(-3, 103)
                rejections.grid(False)
                if column < len(shares) - 1:
                    rejections.set_yticklabels([])
                else:
                    rejections.set_ylabel('rejected (%)')
        # short labels: the curve and strip panels are too low for longer ones
        axes[step * row, 0].set_ylabel({'f1': 'F1', 'f1_change': 'ΔF1 (pp)', 'attack_success_rate': 'ASR'}.get(y, y))
        axes[step * row, 0].annotate(partition, xy=(-0.55, 0.2 if bars else 0.5), xycoords='axes fraction', rotation=90,
                                     va='center', ha='center', fontsize=13, fontweight='bold')
    for ax in axes[-1]:
        ax.set_xlabel('round')
    handles = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles[label] = handle
    fig.tight_layout(h_pad=0 if bars else 5)
    if bars:
        # small gap between a curve and its strip; the spacer rows separate the partitions
        fig.subplots_adjust(hspace=0.08)
    # below the x labels; save's bbox_inches='tight' keeps it in the file;
    # one row, wrapped onto more rows while it is wider than the figure
    for ncol in range(len(handles), 0, -1):
        legend = fig.legend(handles.values(), handles.keys(), loc='upper center', bbox_to_anchor=(0.5, 0), ncol=ncol)
        if ncol == 1 or legend.get_window_extent(fig.canvas.get_renderer()).width <= fig.bbox.width:
            break
        legend.remove()
    # title kept in the thesis: names the attack (and the target), which the panels don't show
    save(fig, f'attacker_count_curves/{name}_{"bar" if bars else "plot"}', title, always_title=True)
    plt.close(fig)


# bar-strip scale per attacker share, shared by all attacks and partitions:
# the most clients rejected (honest + attackers, % of all clients) in any round
def rejection_tops(data):
    rejected = data['malicious_rejected_of_all'] + data['benign_rejected_of_all']
    return rejected.groupby(data['percent']).max()


# one job per figure, drawn in parallel by draw_attacker_counts
def attacker_count_curves():
    jobs = []
    data = rejection_percentages(per_round(runs[runs['host'].isna() & (runs['n_rounds'] == 50)], 'f1', *REJECTION_FIELDS))
    base = data[data['n_malicious'] == 0]
    attacked = data[data['n_malicious'] > 0]
    top = rejection_tops(attacked)
    for name, part in attacked.groupby('name'):
        jobs.append((part, base, 'f1', f'untargeted/{name}', NAMES[name], top))
    # same figures relative to each rule's own run without attack in the same round (pp), as the drops in the text
    clean_f1 = base[['aggregator', 'partition', 'round', 'f1']].rename(columns={'f1': 'clean_f1'})
    relative = attacked.merge(clean_f1, on=['aggregator', 'partition', 'round'])
    relative['f1_change'] = (relative['f1'] - relative['clean_f1']) * 100
    for name, part in relative.groupby('name'):
        jobs.append((part, base.iloc[:0], 'f1_change', f'untargeted/{name}_relative',
                     f'{NAMES[name]}: test F1 change vs. own run without attack', top))

    if backdoors.empty:
        return jobs
    data = rejection_percentages(per_round(backdoors, 'attack_success_rate', *REJECTION_FIELDS))
    top = rejection_tops(data)
    for (name, host, schedule), part in data.groupby(['name', 'host', 'schedule']):
        # no runs without attack: their attack success is 0 % in every round (floor_asr)
        jobs.append((part, None, 'attack_success_rate', f'targeted/{name}_{host}_{schedule}',
                     f'{NAMES[name]}, {NAMES[host]} ({host})', top))
    return jobs




if __name__ == '__main__':
    sns.set_theme(style='whitegrid', context='notebook')
    tasks = []
    for make in [m1_clean_table, m2_clean_curves, m3_untargeted_table, t1_untargeted_tables, t2_targeted_tables, m4_drop_vs_share, m5_backdoor_table,
                 m6_backdoor_curves, m7_band_table, m8_backdoor_peak_vs_share, m8b_backdoor_peak_heatmap, m9_backdoor_drop_vs_share, s1_untargeted_heatmaps, s2_detection_heatmaps, s2_detection_tradeoff,
                 s3_untargeted_curves, s4_non_target_drop, s5_persistence, s6_rare_vs_common,
                 s9_epprfl_internals, s12_host_floor, s13_partitions, attacker_count_curves]:
        if arguments.only and not make.__name__.startswith(tuple(arguments.only.split(','))):
            continue
        if not BACKDOORS and make in [t2_targeted_tables, m5_backdoor_table,m6_backdoor_curves, m8_backdoor_peak_vs_share, m8b_backdoor_peak_heatmap, m9_backdoor_drop_vs_share, s4_non_target_drop, s5_persistence, s6_rare_vs_common]:
            continue
        if make is attacker_count_curves:
            for job in make():
                tasks.append((draw_attacker_counts, job + (True,)))
                tasks.append((draw_attacker_counts, job + (False,)))
        else:
            tasks.append((make, ()))

    def run(task):
        make, args = task
        make(*args)
        # figure jobs: print their file name
        print(args[3] if args else make.__name__, flush=True)
        plt.close('all')

    # fork: workers inherit the loaded runs and the seaborn theme
    with Pool(WORKERS) as pool:
        pool.map(run, tasks, chunksize=1)
