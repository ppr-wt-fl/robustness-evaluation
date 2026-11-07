"""Evaluate AdFlush's published GBM (ONNX export of the MOJO) on the thesis test split. Reads inputs, writes results.json next to itself."""
import json
import os
import zipfile

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = os.path.expanduser('~/projects/uniprojects/masterthesis-code/robustness-evaluation/')
ONNX = ROOT + 'AdFlushAutokeras/model/AdFlush.onnx'
MOJO = ROOT + 'AdFlushAutokeras/model/AdFlush_mojo.zip'
SANITY_CSV = ROOT + 'AdFlushAutokeras/dataset/AdFlush_test.csv'
TEST = ROOT + 'poisoning-eval/.test.parquet'
OLD_TEST = ROOT + 'poisoning-eval/backup_3split/.test.parquet'
SOURCE_CSV = ROOT + 'dataset/features_raw_all_feature.csv'
HERE = os.path.dirname(os.path.abspath(__file__))

# setup.py dropped rows with a NaN in these columns before splitting, so they are needed to rebuild its row order
SETUP_FEATURES = ['content_policy_type', 'url_length', 'brackettodot', 'is_third_party', 'keyword_raw_present',
                  'num_get_storage', 'num_set_storage', 'num_get_cookie', 'num_requests_sent_static', 'req_url_33',
                  'req_url_135', 'req_url_179', 'fqdn_4', 'fqdn_13', 'fqdn_14', 'fqdn_15', 'fqdn_23', 'fqdn_26',
                  'fqdn_27', 'ng_0_0_2', 'ng_0_15_15', 'ng_2_13_2', 'ng_15_0_3', 'ng_15_0_15', 'ng_15_15_15',
                  'avg_ident', 'avg_charperline']

session = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# the ONNX input is one unnamed tensor, so the column order comes from the MOJO's model.ini
ini = zipfile.ZipFile(MOJO).read('model.ini').decode()
columns = ini.split('[columns]')[1].split('[domains]')[0].split()
assert columns[-1] == 'label' and len(columns) == 28
MODEL_COLUMNS = columns[:-1]


def evaluate(frame, labels, name):
    assert frame.columns.tolist() == MODEL_COLUMNS
    x = frame.to_numpy(np.float32)
    predicted_labels, probabilities = [], []
    for start in range(0, len(x), 50_000):
        label, probability = session.run(None, {input_name: x[start:start + 50_000]})
        predicted_labels.append(label)
        probabilities.append(probability)
    predicted_labels = np.concatenate(predicted_labels)
    probabilities = np.concatenate(probabilities)
    assert set(predicted_labels.tolist()) <= {'0', '1', '0.0', '1.0'}, set(predicted_labels.tolist())
    predictions = predicted_labels.astype(float).astype(int)
    y = np.asarray(labels).astype(float).astype(int)
    assert set(np.unique(y)) == {0, 1}
    assert probabilities[predictions == 1, 1].mean() > probabilities[predictions == 0, 1].mean()
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    result = dict(rows=len(y), positives=int(y.sum()),
                  acc=accuracy_score(y, predictions), prec=precision_score(y, predictions),
                  recall=recall_score(y, predictions), f1=f1_score(y, predictions),
                  auroc=roc_auc_score(y, probabilities[:, 1]),
                  tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                  label_threshold_observed=[float(probabilities[predictions == 0, 1].max()),
                                            float(probabilities[predictions == 1, 1].min())])
    print(name, json.dumps(result), flush=True)
    return result


def source_columns(split):
    """Pull columns missing from a split out of the source CSV, rebuilt the way setup.py built the split."""
    wanted = ['visit_id', 'name', 'CLASS', 'keyword_char_present', 'num_requests_sent_webgraph', *SETUP_FEATURES]
    source = pd.read_csv(SOURCE_CSV, usecols=wanted).dropna(subset=SETUP_FEATURES)
    source = source[source['visit_id'].isin(split['visit_id'].unique())].reset_index(drop=True)
    # same rows in the same order, or the join is wrong
    assert len(source) == len(split), (len(source), len(split))
    assert (source['visit_id'].to_numpy() == split['visit_id'].to_numpy()).all()
    assert (source['name'].to_numpy() == split['name'].to_numpy()).all()
    assert (source['CLASS'].astype(bool).to_numpy() == split['CLASS'].to_numpy()).all()
    for column in ('url_length', 'keyword_raw_present', 'num_requests_sent_static', 'avg_charperline', 'fqdn_27'):
        assert np.allclose(source[column].to_numpy(float), split[column].to_numpy(float)), column
    return source


def model_frame(split, source, requests_column, keyword_column):
    frame = split.copy()
    frame['keyword_char_present'] = (source if keyword_column == 'keyword_char_present' else split)[keyword_column].to_numpy()
    frame['num_requests_sent'] = (source if requests_column == 'num_requests_sent_webgraph' else split)[requests_column].to_numpy()
    return frame[MODEL_COLUMNS]


results = {'model_columns': MODEL_COLUMNS}

# sanity check 1: must hold before anything else counts
sanity = pd.read_csv(SANITY_CSV, index_col=0)
results['sanity'] = evaluate(sanity[MODEL_COLUMNS], sanity['label'], 'sanity')
for key, expected in dict(acc=0.9864, prec=0.9929, recall=0.9728, f1=0.9827).items():
    assert abs(results['sanity'][key] - expected) < 0.001, (key, results['sanity'][key], expected)

# sanity check 2
test = pd.read_parquet(TEST)
assert len(test) == 144_712, len(test)

source = source_columns(test)
results['source_stats'] = {
    'webgraph_nan': int(source['num_requests_sent_webgraph'].isna().sum()),
    'keyword_char_nan': int(source['keyword_char_present'].isna().sum()),
    'keyword_char_mean': float(source['keyword_char_present'].mean()),
    'keyword_raw_mean': float(source['keyword_raw_present'].mean()),
    'keyword_char_ne_raw': int((source['keyword_char_present'] != source['keyword_raw_present']).sum()),
    'webgraph_ne_static': int((source['num_requests_sent_webgraph'] != source['num_requests_sent_static']).sum()),
    'webgraph_max': float(source['num_requests_sent_webgraph'].max()),
    'static_max': float(source['num_requests_sent_static'].max()),
}
print(json.dumps(results['source_stats']), flush=True)

for keyword in ('keyword_char_present', 'keyword_raw_present'):
    for requests in ('num_requests_sent_webgraph', 'num_requests_sent_static'):
        name = f'current|{keyword}|{requests}'
        results[name] = evaluate(model_frame(test, source, requests, keyword), test['CLASS'], name)

# diagnostic only: does this reconstruction reproduce the earlier run on the old split?
old = pd.read_parquet(OLD_TEST)
assert len(old) == 72_276, len(old)
old_source = source_columns(old)
for keyword in ('keyword_char_present', 'keyword_raw_present'):
    name = f'old72276|{keyword}|num_requests_sent_webgraph'
    results[name] = evaluate(model_frame(old, old_source, 'num_requests_sent_webgraph', keyword), old['CLASS'], name)

with open(os.path.join(HERE, 'results.json'), 'w') as out:
    json.dump(results, out, indent=1)

