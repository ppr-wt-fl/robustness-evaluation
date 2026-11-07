import os
# needed by torch.use_deterministic_algorithms on CUDA, must be set before CUDA starts
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import json
import time
import random
import resource
import queue
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from flwr_datasets.partitioner import DirichletPartitioner, GroupedNaturalIdPartitioner, IidPartitioner, SizePartitioner
from byzfl import Average
# tqdm comes with datasets; threadpoolctl limits numpy's BLAS threads per worker (experiment.py)
from tqdm import tqdm
from threadpoolctl import threadpool_limits

#  FIXME: look into these settings
# import torch
# os.environ.update(OMP_NUM_THREADS='2', TF_CPP_MIN_LOG_LEVEL='3')
# tf.config.threading.set_intra_op_parallelism_threads(2)
# tf.config.threading.set_inter_op_parallelism_threads(2)
# torch.set_num_threads(2)

# torch's OpenMP thread pool (libgomp) doesn't survive fork: a forked worker hangs on its first torch op once this
# process has run a multi-threaded one. So this process stays at 1 thread, the workers set their own (experiment.py)
torch.set_num_threads(1)

random_generator = np.random.default_rng(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# error instead of silently using a non-deterministic GPU kernel
torch.use_deterministic_algorithms(True)

# FIXME: MOVE DATA TO A PROPPER PATH
DATASET_PATH = os.path.join(os.path.dirname(__file__), '../dataset/features_adlfush.csv')

# FIXME: MOVE MODEL TO A PROPPER PATH
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../AdFlushAutokeras/source/autokeras/runs/2026-10-01_19-33-34/model/AdFlush_autokeras.keras')

# torch version of MODEL_PATH, rebuilt whenever MODEL_PATH changes
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), '.model.pt')

N_CLIENTS = 200
# clients 0..NUM_MALICIOUS-1 are malicious
NUM_MALICIOUS = 4
N_ROUNDS = 50
# one held-out split; 0.30 = the old validation + test (0.15 each), so the train split stays the same
TEST_FRACTION = 0.30
# own seed so the split never shifts with random_generator (and doesn't consume it)
SPLIT_SEED = 0
# all partitions are built once, the grid in experiment.py picks one per run. Default for runs outside the grid
PARTITION_SCHEME = 'iid'  # iid | dirichlet_<alpha> | quantity | natural
# label skew, smaller = more skewed. With 200 clients flwr's DirichletPartitioner can't give every client
# MIN_PARTITION_SIZE rows below alpha 0.5 (10 attempts fail)
DIRICHLET_ALPHAS = [0.5, 1.0, 10.0]
QUANTITY_BETA = 0.5  # size skew, smaller = more skewed
MIN_PARTITION_SIZE = 10

FEATURE_COLUMNS = [
         "content_policy_type",
        "url_length",
        "brackettodot",
        "is_third_party",
        # FIXME: model expects keyword_char_present (normalization stats match it, raw is a superset)
        "keyword_raw_present",
        "num_get_storage",
        "num_set_storage",
        "num_get_cookie",
        # FIXME: model expects num_requests_sent, stats match num_requests_sent_webgraph; static has |z| up to ~7600
        "num_requests_sent_static",
        "req_url_33",
        "req_url_135",
        "req_url_179",
        "fqdn_4",
        "fqdn_13",
        "fqdn_14",
        "fqdn_15",
        "fqdn_23",
        "fqdn_26",
        "fqdn_27",
        "ng_0_0_2",
        "ng_0_15_15",
        "ng_2_13_2",
        "ng_15_0_3",
        "ng_15_0_15",
        "ng_15_15_15",
        "avg_ident",
 "avg_charperline",
]
 

# ====================================================================
# Load dataset and train / test split
# ====================================================================

TRAIN_CACHE_PATH = os.path.join(os.path.dirname(__file__), '.train.parquet')
TEST_CACHE_PATH = os.path.join(os.path.dirname(__file__), '.test.parquet')
# left by the old train / validation / test split, whose .test.parquet holds only half of today's test split
OLD_VALIDATION_CACHE_PATH = os.path.join(os.path.dirname(__file__), '.validation.parquet')

if os.path.exists(TRAIN_CACHE_PATH) and os.path.exists(TEST_CACHE_PATH) and not os.path.exists(OLD_VALIDATION_CACHE_PATH):
    train_dataframe = pd.read_parquet(TRAIN_CACHE_PATH)
    test_dataframe = pd.read_parquet(TEST_CACHE_PATH)
else:
    dataframe = pd.read_csv(DATASET_PATH, usecols=["top_level_url","visit_id","name", *FEATURE_COLUMNS, "CLASS"])
    dataframe = dataframe.dropna(subset=FEATURE_COLUMNS)

    # content_policy_type mapping
    # I know its horrible to do it here :/
    CONTENT_POLICY_TYPES_PATH = os.path.join(os.path.dirname(__file__), '../YOPO/AdFlush/source/json/content_type_dict.json')
    with open(CONTENT_POLICY_TYPES_PATH) as content_policy_types_file:
        CONTENT_POLICY_TYPES = json.load(content_policy_types_file)
    dataframe['content_policy_type'] = dataframe['content_policy_type'].map(CONTENT_POLICY_TYPES).fillna(0)

    # split by site -> all requests that stemmed from visiting one page
    sites, site_sizes = np.unique(dataframe['visit_id'], return_counts=True)
    site_order = np.random.default_rng(SPLIT_SEED).permutation(len(sites))
    shuffled_sites = sites[site_order]

    # sites in shuffled order: [test | train], cut at roughly TEST_FRACTION of the rows (not necessarily of the sites)
    cumulative_rows = np.cumsum(site_sizes[site_order])
    test_end = np.searchsorted(cumulative_rows, TEST_FRACTION * len(dataframe)) + 1
    test_mask = dataframe['visit_id'].isin(shuffled_sites[:test_end]).to_numpy()
    train_mask = ~test_mask

    # this also shuffles training rows
    train_dataframe = dataframe[train_mask].sample(frac=1, random_state=SPLIT_SEED).reset_index(drop=True)
    test_dataframe = dataframe[test_mask].reset_index(drop=True)

    os.makedirs(os.path.dirname(TRAIN_CACHE_PATH), exist_ok=True)
    train_dataframe.to_parquet(TRAIN_CACHE_PATH)
    test_dataframe.to_parquet(TEST_CACHE_PATH)
    if os.path.exists(OLD_VALIDATION_CACHE_PATH):
        os.remove(OLD_VALIDATION_CACHE_PATH)

train_features = train_dataframe[FEATURE_COLUMNS].to_numpy(np.float32)
train_labels = train_dataframe['CLASS'].to_numpy(np.float32)

test_features = test_dataframe[FEATURE_COLUMNS].to_numpy(np.float32)
test_labels = test_dataframe['CLASS'].to_numpy(np.float32)

print(f'train: {len(train_dataframe)} rows, {train_dataframe["visit_id"].nunique()} sites, positive rate {train_labels.mean():.3f}', flush=True)
print(f'test: {len(test_dataframe)} rows, {test_dataframe["visit_id"].nunique()} sites, positive rate {test_labels.mean():.3f}', flush=True)


# ====================================================================
# Backdoor targets (Bagdasaryan et al. 2020 semantic backdoor)
# ====================================================================

def request_hosts(dataframe):
    return dataframe['name'].map(lambda url: urlsplit(url).hostname).to_numpy()

train_hosts = request_hosts(train_dataframe)
test_hosts = request_hosts(test_dataframe)


# ====================================================================
# Partition training set among clients
# ====================================================================

# 'index' maps partitions back to train_features; rows are already shuffled (see split above)
train_dataset = Dataset.from_pandas(train_dataframe[['visit_id', 'CLASS']].reset_index())

partitioners = {'iid': IidPartitioner(num_partitions=N_CLIENTS)}

# label skew (Wang et al. / Hsu et al.): per class, Dirichlet(alpha) proportions over the clients
for alpha in DIRICHLET_ALPHAS:
    partitioners[f'dirichlet_{alpha}'] = DirichletPartitioner(
            num_partitions=N_CLIENTS,
            partition_by='CLASS',
            alpha=alpha,
            min_partition_size=MIN_PARTITION_SIZE,
            seed=SPLIT_SEED,
            )

# size skew: every client gets MIN_PARTITION_SIZE, the rest is split by Dirichlet(beta) proportions
size_generator = np.random.default_rng(SPLIT_SEED)
remaining = len(train_dataset) - N_CLIENTS * MIN_PARTITION_SIZE
sizes = MIN_PARTITION_SIZE + size_generator.multinomial(remaining, size_generator.dirichlet(np.full(N_CLIENTS, QUANTITY_BETA)))
partitioners['quantity'] = SizePartitioner(partition_sizes=[int(size) for size in sizes])

# partition name -> list of N_CLIENTS arrays of train row indices
CLIENT_PARTITIONS = {}
for name, partitioner in partitioners.items():
    partitioner.dataset = train_dataset
    if partitioner.num_partitions != N_CLIENTS:
        sys.exit(f'{name} produced {partitioner.num_partitions} partitions, expected {N_CLIENTS}')
    CLIENT_PARTITIONS[name] = [np.sort(partitioner.load_partition(client_index)['index']) for client_index in range(N_CLIENTS)]

# natural: every client gets whole sites (visit_id), the shuffled sites cut into N_CLIENTS groups.
# Not GroupedNaturalIdPartitioner: with 5049 sites and group size 5049 // 200 it gives 201 partitions
train_sites = train_dataframe['visit_id'].to_numpy()
site_groups = np.array_split(np.random.default_rng(SPLIT_SEED).permutation(np.unique(train_sites)), N_CLIENTS)
CLIENT_PARTITIONS['natural'] = []
for site_group in site_groups:
    CLIENT_PARTITIONS['natural'].append(np.flatnonzero(np.isin(train_sites, site_group)))

print(f'train={len(train_features)} test={len(test_features)}', flush=True)
for name, partitions in CLIENT_PARTITIONS.items():
    sizes = []
    positive_rates = []
    for partition in partitions:
        sizes.append(len(partition))
        positive_rates.append(float(train_labels[partition].mean()))
    print(f'{name}: client sizes min/median/max {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}, '
          f'positive rate 10/50/90% {np.round(np.percentile(positive_rates, [10, 50, 90]), 2)}', flush=True)

client_partitions = CLIENT_PARTITIONS[PARTITION_SCHEME]

# ====================================================================
# Load model
# ====================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Keras Rescaling: x * scale + offset (the standardization AutoKeras' Normalization was swapped for)
class Rescale(nn.Module):
    def __init__(self, scale, offset):
        super().__init__()
        self.register_buffer('scale', scale)
        self.register_buffer('offset', offset)

    def forward(self, x):
        return x * self.scale + self.offset

# Keras model -> list of (layer kind, *torch tensors / numbers), the format cached in MODEL_CACHE_PATH.
# Covers the layers train_autokeras.py produces; the final sigmoid is dropped, the torch model outputs logits
# (needs the module-level `tf` imported below, only when the cache is rebuilt)
def keras_to_layer_specs(keras_model):
    to_tensor = lambda value: torch.tensor(np.ascontiguousarray(value, 'f4'))
    specs = []
    for layer in keras_model.layers:
        activation = getattr(getattr(layer, 'activation', None), '__name__', None)
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        elif isinstance(layer, tf.keras.layers.Rescaling):
            specs.append(('rescale', to_tensor(layer.scale), to_tensor(layer.offset)))
        elif isinstance(layer, tf.keras.layers.Dense):
            kernel, bias = layer.get_weights()
            specs.append(('linear', to_tensor(kernel.T), to_tensor(bias)))
            if activation == 'relu':
                specs.append(('relu',))
            elif activation != 'linear' and not (layer is keras_model.layers[-1] and activation == 'sigmoid'):
                sys.exit(f'cannot convert activation {activation} of layer {layer.name}')
        elif isinstance(layer, tf.keras.layers.ReLU) or (isinstance(layer, tf.keras.layers.Activation) and activation == 'relu'):
            specs.append(('relu',))
        elif isinstance(layer, tf.keras.layers.Dropout):
            specs.append(('dropout', float(layer.rate)))
        elif isinstance(layer, tf.keras.layers.Activation) and activation == 'sigmoid' and layer is keras_model.layers[-1]:
            continue
        else:
            sys.exit(f'cannot convert layer {layer.name} ({type(layer).__name__})')
    return specs

def build_model(specs):
    layers = []
    for kind, *values in specs:
        if kind == 'rescale':
            layers.append(Rescale(*values))
        elif kind == 'linear':
            weight, bias = values
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data, linear.bias.data = weight, bias
            layers.append(linear)
        elif kind == 'relu':
            layers.append(nn.ReLU())
        elif kind == 'dropout':
            layers.append(nn.Dropout(values[0]))
    return nn.Sequential(*layers)

# bump MODEL_CACHE_VERSION when the cached layer format changes
MODEL_CACHE_VERSION = 2
model_source = (os.path.abspath(MODEL_PATH), os.path.getmtime(MODEL_PATH), MODEL_CACHE_VERSION)
cache = torch.load(MODEL_CACHE_PATH) if os.path.exists(MODEL_CACHE_PATH) else None
if cache is None or tuple(cache['source']) != model_source:
    # only place TF is needed. AutoKeras export only uses built-in Keras layers, so autokeras isn't needed to load it
    import tensorflow as tf
    keras_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    cache = {'source': model_source, 'layers': keras_to_layer_specs(keras_model)}

    # the torch model must reproduce the Keras one before it replaces it
    probe = test_features[:1024]
    keras_probabilities = keras_model.predict(probe, verbose=0)[:, 0]
    with torch.no_grad():
        torch_probabilities = torch.sigmoid(build_model(cache['layers']).eval()(torch.from_numpy(probe))[:, 0]).numpy()
    max_difference = np.abs(keras_probabilities - torch_probabilities).max()
    if max_difference > 1e-4:
        sys.exit(f'torch conversion differs from the Keras model by up to {max_difference}')
    print(f'converted {MODEL_PATH} to torch (max difference {max_difference:.1e}), cached in {MODEL_CACHE_PATH}', flush=True)
    torch.save(cache, MODEL_CACHE_PATH)

model = build_model(cache['layers']).to(device)
print(model, flush=True)

# everything clients share, as one flat vector (the format ByzFL and the attacks work on):
# the Dense kernels/biases. The Rescale scale/offset are fixed buffers, not shared
shared_tensors = list(model.parameters())

def get_flat():
    return torch.cat([tensor.detach().flatten() for tensor in shared_tensors]).cpu().numpy()

# load weights into model
@torch.no_grad()
def set_flat(flat):
    flat = torch.from_numpy(np.asarray(flat, np.float32)).to(device)
    for tensor, chunk in zip(shared_tensors, flat.split([tensor.numel() for tensor in shared_tensors])):
        tensor.copy_(chunk.view_as(tensor))

initial_weights = get_flat()
