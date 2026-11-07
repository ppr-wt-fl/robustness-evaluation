import subprocess
import traceback
from typing import TYPE_CHECKING
from torch.func import functional_call, grad, vmap
from results_types import (
    Metrics,
    FilteringStats,
    UpdateStats,
    InitialRoundResult,
    RoundResult,
    RunConfig,
    RunResult,
)

# editor only: run.py execs this file inside setup.py's globals, this import never runs
if TYPE_CHECKING:
    from setup import *

# ====================================================================
# Training
# ====================================================================

# 0.1 since Oct 22 2026; earlier grids used 0.01
CLIENT_LEARNING_RATE = 0.1
# FIXME: LEARNING_RATE is an untuned guess against NaN divergence; revisit once the FEATURE_COLUMNS FIXMEs in setup.py are fixed
# per-batch gradient clipping, only a guard against NaN divergence (some standardized inputs reach |z| > 1000);
# 1.0 again since Oct 22 2026; the clip-30 grid of Oct 21 2026 used 30
GRAD_CLIP_NORM = 1.0
BATCH_SIZE = 64
# local epochs of the honest clients (local_train)
LOCAL_EPOCHS = 2
# None: LOCAL_EPOCHS epochs per round, so a client takes as many steps as its data gives. A number: every honest client
# takes exactly this many steps per round, whatever its size (local_orders). 54 = 2 epochs of an iid client
# (1687 rows -> 27 batches of 64), also the mean over clients of every partition
LOCAL_STEPS = 54
# per-client norms and cosines to the honest mean in EPPRFL's round results (False: off). Analysis only, the server
# can't know the cosine; attackers are clients 0..n-1, so the client order identifies them
LOG_CLIENT_STATS = True
# every (attack, aggregator) run restarts random_generator and torch from this seed, so all runs see the same batch orders
SEED = 0

# SGD without momentum has no state, so one optimizer is shared by all clients
optimizer = torch.optim.SGD(model.parameters(), lr=CLIENT_LEARNING_RATE)


# reset in place: aggregators hold a reference to random_generator
def reseed():
    random_generator.bit_generator.state = np.random.default_rng(
        SEED
    ).bit_generator.state
    torch.manual_seed(SEED)


# the client's row positions for one round of local training, one array per pass. LOCAL_STEPS None: LOCAL_EPOCHS
# shuffles, one per epoch. Else one pass of LOCAL_STEPS full batches: shuffles chained until there are enough rows
def local_orders(n_rows):
    if LOCAL_STEPS is None:
        orders = []
        for epoch in range(LOCAL_EPOCHS):
            orders.append(random_generator.permutation(n_rows))
        return orders
    n_needed = LOCAL_STEPS * BATCH_SIZE
    shuffles = []
    for shuffle in range(-(-n_needed // n_rows)):
        shuffles.append(random_generator.permutation(n_rows))
    return [np.concatenate(shuffles)[:n_needed]]


# the passes of local_orders with SGD(LEARNING_RATE), batch BATCH_SIZE, gradients clipped to GRAD_CLIP_NORM, from global weights; returns the update.
# max_norm projects the update onto an L2 ball every 10 batches and at the end of each pass (PGD, used by the tails attack)
def local_train(global_weights, features, labels, max_norm=None):
    set_flat(global_weights)
    model.train()

    features = torch.from_numpy(features).to(device)
    labels = torch.from_numpy(labels).to(device)

    for order in local_orders(len(features)):
        batches = torch.from_numpy(order).split(BATCH_SIZE)

        for step, batch in enumerate(batches):
            optimizer.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(
                model(features[batch])[:, 0], labels[batch]
            ).backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            if max_norm and ((step + 1) % 10 == 0 or step == len(batches) - 1):
                update = get_flat() - global_weights
                set_flat(
                    global_weights + update * min(1, max_norm / np.linalg.norm(update))
                )
    update = get_flat() - global_weights
    assert np.isfinite(update).all(), "local training diverged (non-finite update)"
    return update


# local_train for all honest clients at once: the model is copied once per client and each step trains one batch of
# every client together (vmap). Returns the (n_clients, d) updates, as np.stack of local_train on every partition
train_features_on_device = torch.from_numpy(train_features).to(device)
train_labels_on_device = torch.from_numpy(train_labels).to(device)


# mean loss over the client's real rows: padding rows have weight 0, a client with no rows left in this step gets loss 0
def client_loss(params, buffers, features, labels, weights):
    logits = functional_call(model, (params, buffers), (features,))[:, 0]
    losses = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (losses * weights).sum() / weights.sum().clamp(min=1)


client_gradients = vmap(grad(client_loss), in_dims=(0, None, 0, 0, 0), randomness="different")


def train_all_clients(global_weights, partitions):
    set_flat(global_weights)
    model.train()
    n_clients = len(partitions)

    params = {}
    for name, tensor in model.named_parameters():
        params[name] = tensor.detach().unsqueeze(0).repeat(n_clients, *[1] * tensor.dim())
    buffers = dict(model.named_buffers())

    # batch orders drawn in local_train's order (client by client, pass by pass), so random_generator gives the same batches
    orders = []
    for partition in partitions:
        client_orders = []
        for order in local_orders(len(partition)):
            client_orders.append(partition[order])
        orders.append(client_orders)
    n_passes = len(orders[0])

    # batches per pass of every client. Row i of params / rows is client sorted_clients[i], largest first, so the
    # clients that still have batches in a step are the first n_active rows (a view, no copy)
    client_steps = np.zeros(n_clients, np.int64)
    for client in range(n_clients):
        client_steps[client] = -(-len(orders[client][0]) // BATCH_SIZE)
    sorted_clients = np.argsort(-client_steps, kind="stable")
    n_steps = int(client_steps.max())
    padded_length = n_steps * BATCH_SIZE
    # steps each row of params took, for the LOCAL_STEPS check
    steps_taken = np.zeros(n_clients, np.int64)

    for pass_index in range(n_passes):
        # each client's pass padded to the longest one, padding rows get weight 0
        rows = np.zeros((n_clients, padded_length), np.int64)
        weights = np.zeros((n_clients, padded_length), np.float32)
        for i in range(n_clients):
            order = orders[sorted_clients[i]][pass_index]
            rows[i, : len(order)] = order
            weights[i, : len(order)] = 1
        rows = torch.from_numpy(rows).to(device)
        weights = torch.from_numpy(weights).to(device)

        for step in range(n_steps):
            n_active = int((client_steps > step).sum())
            steps_taken[:n_active] += 1
            active_params = {}
            for name in params:
                active_params[name] = params[name][:n_active]
            batch = slice(step * BATCH_SIZE, (step + 1) * BATCH_SIZE)
            batch_rows = rows[:n_active, batch]
            gradients = client_gradients(
                active_params,
                buffers,
                train_features_on_device[batch_rows],
                train_labels_on_device[batch_rows],
                weights[:n_active, batch],
            )

            # clip_grad_norm_ per client: one norm over all of the client's gradients
            # (vector_norm / addcmul_ instead of pow / multiply: no full-size temporaries)
            squared_norms = torch.zeros(n_active, device=device)
            for gradient in gradients.values():
                squared_norms += torch.linalg.vector_norm(gradient.flatten(1), dim=1) ** 2
            clip_factors = (GRAD_CLIP_NORM / (squared_norms.sqrt() + 1e-6)).clamp(max=1)

            for name, gradient in gradients.items():
                factors = clip_factors.view(-1, *[1] * (gradient.dim() - 1))
                active_params[name].addcmul_(factors, gradient, value=-CLIENT_LEARNING_RATE)

    if LOCAL_STEPS is not None:
        assert (steps_taken == LOCAL_STEPS).all(), "a client did not take exactly LOCAL_STEPS steps"
    # the update and the finite check on the device (in place, no extra GPU memory), then the rows back in client order
    flat = torch.cat([params[name].flatten(1) for name, _ in model.named_parameters()], 1)
    flat -= torch.from_numpy(global_weights).to(device)
    assert torch.isfinite(flat).all(), "local training diverged (non-finite update)"
    return flat.cpu().numpy()[np.argsort(sorted_clients)]


@torch.no_grad()
def predict_logits(features, weights):
    set_flat(weights)
    model.eval()

    features = torch.from_numpy(features).to(device)

    # chunked so the whole test set never has to go through the model at once
    logits = []
    for chunk in features.split(65536):
        logits.append(model(chunk)[:, 0])

    return torch.cat(logits).cpu().numpy()


# sigmoid(logit) > 0.5  <=>  logit > 0
def predict(features, weights):
    return predict_logits(features, weights) > 0


# attack success rate of a backdoor: share of the target's rows the model classifies as the target label
def success_rate(features, weights, target_label):
    # a mean over no rows is NaN and would end up in the results
    assert len(features) > 0, "no target rows to measure the success rate on"
    return float((predict(features, weights) == target_label).mean())


# ====================================================================
# Metrics
# ====================================================================


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


# ROC curve with one point per distinct score, highest threshold first (same output as sklearn.metrics.roc_curve)
def roc_curve(labels, scores):
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order], scores[order]
    # last index of every run of equal scores
    threshold_ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    true_positives = np.cumsum(labels)[threshold_ends]
    false_positives = threshold_ends + 1 - true_positives
    true_positive_rates = np.r_[0, true_positives / true_positives[-1]]
    false_positive_rates = np.r_[0, false_positives / false_positives[-1]]
    return (
        false_positive_rates,
        true_positive_rates,
        np.r_[np.inf, scores[threshold_ends]],
    )


# AUROC and the Youden cutoff are computed from logits; AdFlush passes hard predictions there,
# which makes AUROC equal to balanced accuracy and the "curve" a single point
def evaluate(
    weights, features=test_features, labels=test_labels
) -> Metrics:
    return metrics_from_logits(predict_logits(features, weights), labels)


# evaluate on all rows and on the rows of the non-target hosts (mask non_target), from one forward pass.
# The backdoor targets' rows are left out of the second one: there a successful backdoor is also a main-task
# error (a false negative for evasion, a false positive for sabotage) and would be counted twice
def evaluate_with_non_target(weights, features, labels, non_target):
    logits = predict_logits(features, weights)
    all_rows = metrics_from_logits(logits, labels)
    non_target_rows = metrics_from_logits(logits[non_target], labels[non_target])
    return all_rows, non_target_rows


def metrics_from_logits(logits, labels) -> Metrics:
    labels = labels.astype(bool)
    # NaN logits would silently count as negatives and report majority-class metrics
    assert np.isfinite(logits).all(), "non-finite logits in evaluate"
    predictions = logits > 0

    true_positives = int((predictions & labels).sum())
    false_positives = int((predictions & ~labels).sum())
    true_negatives = int((~predictions & ~labels).sum())
    false_negatives = int((~predictions & labels).sum())

    precision = ratio(true_positives, true_positives + false_positives)
    recall = ratio(true_positives, true_positives + false_negatives)

    false_positive_rates, true_positive_rates, thresholds = roc_curve(labels, logits)
    # Youden's J: the threshold maximizing TPR - FPR
    cutoff = np.argmax(true_positive_rates - false_positive_rates)

    return {
        "accuracy": ratio(true_positives + true_negatives, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "fnr": ratio(false_negatives, true_positives + false_negatives),
        "fpr": ratio(false_positives, false_positives + true_negatives),
        "auroc": float(
            np.sum(
                np.diff(false_positive_rates)
                * (true_positive_rates[1:] + true_positive_rates[:-1])
                / 2
            )
        ),
        "youden_tpr": float(true_positive_rates[cutoff]),
        "youden_fpr": float(false_positive_rates[cutoff]),
        "youden_threshold_logit": float(thresholds[cutoff]),
        "tp": true_positives,
        "fp": false_positives,
        "tn": true_negatives,
        "fn": false_negatives,
    }


# which updates the aggregator filtered out, and whether those were the malicious ones.
# accepted: indices the aggregator returned (None if it doesn't select, e.g. ByzFL wrappers), malicious: the attacker's clients.
# None accepted gives None everywhere, so "doesn't select" isn't confused with "filtered nothing".
# Rates are None when undefined (nothing rejected / no attackers)
def filtering_stats(accepted_indices, malicious_indices, n_clients) -> FilteringStats:
    if accepted_indices is None:
        return {
            "accepted_clients": None,
            "n_rejected": None,
            "n_malicious_rejected": None,
            "n_benign_rejected": None,
            "n_malicious_accepted": None,
            "detection_precision": None,
            "detection_recall": None,
        }

    accepted = np.zeros(n_clients, bool)
    accepted[accepted_indices] = True

    malicious = np.zeros(n_clients, bool)
    malicious[malicious_indices] = True

    n_rejected = int((~accepted).sum())
    n_malicious_rejected = int(((~accepted) & malicious).sum())
    n_malicious_accepted = int((accepted & malicious).sum())

    return {
        "accepted_clients": [int(index) for index in accepted_indices],
        "n_rejected": n_rejected,
        "n_malicious_rejected": n_malicious_rejected,
        "n_benign_rejected": n_rejected - n_malicious_rejected,
        "n_malicious_accepted": n_malicious_accepted,
        # share of the rejected updates that were malicious
        "detection_precision": n_malicious_rejected / n_rejected
        if n_rejected
        else None,
        # share of the malicious updates that got rejected
        "detection_recall": n_malicious_rejected / int(malicious.sum())
        if malicious.any()
        else None,
    }


# norms of this round's updates and how far the aggregate lands from the benign mean
# (the honest update without attackers). malicious: the attacker's client indices
def update_stats(updates, malicious, aggregated_update, log_client_stats=False) -> UpdateStats:
    benign = np.ones(len(updates), bool)
    benign[malicious] = False
    # no copies of the (n, d) updates: norms of all rows at once, the benign mean row by row
    norms = np.linalg.norm(updates, axis=1)
    benign_norms = norms[benign]
    benign_mean = weighted_mean(updates, np.flatnonzero(benign))
    aggregate_norm = float(np.linalg.norm(aggregated_update))
    benign_mean_norm = float(np.linalg.norm(benign_mean))

    stats = {
        "benign_norm_mean": float(benign_norms.mean()),
        "benign_norm_max": float(benign_norms.max()),
        "malicious_norm_mean": float(norms[malicious].mean())
        if len(malicious)
        else None,
        "aggregate_norm": aggregate_norm,
        "aggregate_distance_to_benign_mean": float(
            np.linalg.norm(aggregated_update - benign_mean)
        ),
        "aggregate_cosine_to_benign_mean": float(aggregated_update @ benign_mean)
        / (aggregate_norm * benign_mean_norm)
        if aggregate_norm and benign_mean_norm
        else None,
    }
    # per client, 6 significant digits; only for filtering aggregators (LOG_CLIENT_STATS)
    if log_client_stats:
        cosines = updates @ benign_mean / (norms * benign_mean_norm + 1e-12)
        stats["client_norms"] = [float(f"{value:.6g}") for value in norms]
        stats["client_cosines_to_benign_mean"] = [float(f"{value:.6g}") for value in cosines]
    return stats


# scalar constructor settings of an attack / aggregator (threshold, f, epsilon, ...), for the results
def hyperparameters(component):
    return {
        key: value
        for key, value in vars(component).items()
        if key != "name" and isinstance(value, (int, float, str, bool))
    }


# attack_context = AttackContext(
#     local_train=local_train,
#     train_features=train_features,
#     train_labels=train_labels,
#     client_partitions=client_partitions,
#     tail_train_features=tail_train_features,
#     random_generator=random_generator,
# )

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
# one JSON line per (attack, aggregator) run, in a new file per execution of this script
# JOB_SHARD="i/n": this process only runs every n-th job of the grid starting at i, so n processes (e.g. one per GPU)
# split it. Each shard writes its own results file
JOB_SHARD = os.environ.get("JOB_SHARD")
results_name = time.strftime("%Y%m%d-%H%M%S")
if JOB_SHARD:
    results_name += "-shard" + JOB_SHARD.replace("/", "of")
RESULTS_PATH = os.path.join(EXPERIMENT_DIR, "results", results_name + ".jsonl")
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

# settings shared by all runs of this execution, stored with every run
RUN_CONFIG: RunConfig = {
    "n_clients": N_CLIENTS,
    "n_rounds": N_ROUNDS,
    "n_malicious": NUM_MALICIOUS,
    "partition_scheme": PARTITION_SCHEME,
    # alpha of a dirichlet_<alpha> partition, else None. Set per run by run_job, with client_sizes / client_positive_rates
    "dirichlet_alpha": None,
    "quantity_beta": QUANTITY_BETA,
    "client_sizes": [len(partition) for partition in client_partitions],
    "client_positive_rates": [
        float(train_labels[partition].mean()) for partition in client_partitions
    ],
    "learning_rate": CLIENT_LEARNING_RATE,
    "grad_clip_norm": GRAD_CLIP_NORM,
    "batch_size": BATCH_SIZE,
    "local_epochs": LOCAL_EPOCHS,
    "local_steps": LOCAL_STEPS,
    "seed": SEED,
    "split_seed": SPLIT_SEED,
    "device": str(device),
    "model_path": os.path.abspath(MODEL_PATH),
    "dataset_path": os.path.abspath(DATASET_PATH),
    "n_train": len(train_features),
    "n_test": len(test_features),
}

# for attack_name in attack_names:
#     for aggregator_name, aggregator in aggregators.items():
#         global_weights = initial_weights

# ====================================================================
# Attack schedule
# ====================================================================

# rounds in which the malicious clients attack. In the other rounds they train honestly on their own data
#   "every"        every round (Fang, Min-Max / Min-Sum)
#   "single_shot"  once, in round WARMUP_ROUNDS + 1, after WARMUP_ROUNDS benign rounds (Bagdasaryan et al.)
#   "every_k"      every ATTACK_EVERY-th round (Wang et al., fixed frequency)
ATTACK_ROUNDS = "every"
WARMUP_ROUNDS = 5
ATTACK_EVERY = 10
# malicious clients (the first ones) that attack in an attack round, None = all NUM_MALICIOUS.
# The defenses still get f = NUM_MALICIOUS
ATTACKERS_PER_ATTACK_ROUND = None

# backdoor entries (Bagdasaryan, Wang) override N_ROUNDS and ATTACKERS_PER_ATTACK_ROUND with these,
# and ATTACK_ROUNDS where their entries are built (the grid at the end overrides it again with BACKDOOR_SCHEDULES)
# BACKDOOR_N_ROUNDS = 100
# every-round backdoor run: 50 attack rounds as in the untargeted runs, then BACKDOOR_DECAY_ROUNDS without attack
BACKDOOR_N_ROUNDS = 70
# benign rounds at the end of every backdoor run: no schedule attacks after BACKDOOR_N_ROUNDS - BACKDOOR_DECAY_ROUNDS,
# and the test success rate is also taken this many rounds after the last attack, so schedules can be compared
BACKDOOR_DECAY_ROUNDS = 20
# None = all NUM_MALICIOUS attack
BACKDOOR_ATTACKERS_PER_ATTACK_ROUND = None

# class-based attacks (attacks/attack_class.py). Not named `attacks`: that would overwrite the package run.py reloads
from attacks.attack_class import NoAttack
from attacks.shejwalkarManipulatingByzantineOptimizing2021 import MinMaxAttack, MinSumAttack
from attacks.fangLocalModelPoisoning2020 import FangKrumAttack
from attacks.allenzhuByzantineResilientNonConvex2020 import SignFlippingAttack
from attacks.blanchardByzantineTolerantMachineLearning2017 import GaussianAttack

# attack / aggregator are factories, so every run starts from a fresh state.
# The attack factory gets the number of attackers per attack round.
# display_name: longer name for plots and tables. Optional n_rounds / attack_rounds / attackers_per_attack_round
# override N_ROUNDS / ATTACK_ROUNDS / ATTACKERS_PER_ATTACK_ROUND for the entry, warmup_rounds / attack_every
# override WARMUP_ROUNDS / ATTACK_EVERY
class_attacks = {
    "none": {
        "attack": lambda n_attackers: NoAttack(),
        "display_name": "No attack",
    },
    "fang_krum": {
        "attack": lambda n_attackers: FangKrumAttack(n_attackers, random_generator),
        "display_name": "Fang-Krum, partial knowledge (Fang et al. 2020)",
    },
    "sign_flipping": {
        "attack": lambda n_attackers: SignFlippingAttack(n_attackers),
        "display_name": "Sign flipping, partial knowledge (Allen-Zhu et al. 2021)",
    },
    "gaussian": {
        "attack": lambda n_attackers: GaussianAttack(n_attackers, random_generator),
        "display_name": "Gaussian noise (Blanchard et al. 2017)",
    },
}

# Min-Max / Min-Sum with each perturbation direction of the original code (dev_type)
shejwalkar_dev_types = {
    "unit_vec": "inverse unit vector",
    "sign": "inverse sign",
    "std": "inverse standard deviation",
}


def minmax_attack(dev_type):
    return lambda n_attackers: MinMaxAttack(n_attackers, dev_type=dev_type)


def minsum_attack(dev_type):
    return lambda n_attackers: MinSumAttack(n_attackers, dev_type=dev_type)


for dev_type, description in shejwalkar_dev_types.items():
    # unit_vec keeps its old key, so earlier results stay comparable
    suffix = "" if dev_type == "unit_vec" else f"_{dev_type}"
    class_attacks[f"minmax_ndss{suffix}"] = {
        "attack": minmax_attack(dev_type),
        "display_name": f"Min-Max, {description} (Shejwalkar & Houmansadr 2021)",
    }
    class_attacks[f"minsum_ndss{suffix}"] = {
        "attack": minsum_attack(dev_type),
        "display_name": f"Min-Sum, {description} (Shejwalkar & Houmansadr 2021)",
    }


# the edge-case set of a host from edge_case_set/calculation.py select, as row indices into the
# cached parquets, so they are checked against the split loaded here
def load_edge_set(host):
    edge = np.load(os.path.join(EXPERIMENT_DIR, "edge_sets", f"{host}_hidden.npz"))
    assert (train_hosts[edge["exclude_rows"]] == host).all() and len(
        edge["exclude_rows"]
    ) == (train_hosts == host).sum(), (
        f"{host}: edge set does not match the cached split, rerun edge_case_set/calculation.py select"
    )
    assert (test_hosts[edge["test_rows"]] == host).all(), (
        f"{host}: edge set does not match the cached split, rerun edge_case_set/calculation.py select"
    )

    return {
        # all of the host's train rows, left out of the attackers' clean data (not only the ones below p)
        "train_rows": edge["exclude_rows"],
        "train_features": train_features[edge["train_rows"]],
        "test_features": test_features[edge["test_rows"]],
        "target_label": edge["target_label"].item(),
    }


# an edge-case target (rows below p) and a common, non-tail target for comparison (all of its rows, setup.py)
BACKDOOR_TARGETS = {
    "stats.g.doubleclick.net": load_edge_set("stats.g.doubleclick.net"),  # rare
    "www.google.de": load_edge_set("www.google.de"),  # rare
    "googleads.g.doubleclick.net": load_edge_set("googleads.g.doubleclick.net"),  # rare
    "ad-delivery.net": load_edge_set("ad-delivery.net"),  # common
    "bam.nr-data.net": load_edge_set("bam.nr-data.net"),  # common
    "ep1.adtrafficquality.google": load_edge_set(
        "ep1.adtrafficquality.google"
    ),  # common
    "i.ytimg.com": load_edge_set("i.ytimg.com"),  # common
    "ka-p.fontawesome.com": load_edge_set("ka-p.fontawesome.com"),  # common
    "images.ctfassets.net": load_edge_set("images.ctfassets.net"),  # common
    "lh3.googleusercontent.com": load_edge_set(
        "lh3.googleusercontent.com"
    ),  # very common
    "storage.googleapis.com": load_edge_set("storage.googleapis.com"),  # very common
}

# baseline for the backdoors: the clean run measures every target's success rate, without any poisoning
class_attacks["none"]["baseline_targets"] = BACKDOOR_TARGETS

# rows of every host in BACKDOOR_TARGETS (not only the attacked one), so all runs get the same non-target rows
test_non_target = ~np.isin(test_hosts, list(BACKDOOR_TARGETS))


# Bagdasaryan et al. 2020 backdoor, one entry per (variant, backdoor host from setup.py's BACKDOOR_HOSTS).
# The host's test rows give the success rate, every round and on the final model
from attacks.bagdasaryanHowBackdoorFederated2020 import ModelReplacementAttack

# alpha of constrain-and-scale (Eq. 4): largest alpha < 1 in arXiv v3 App. A.2 Fig. 13 (1, 0.7, 0.5, 0.45, 0.4;
# cosine L_ano, word prediction). The code's params.yaml / words.yaml keep alpha_loss: 1.0
BAGDASARYAN_ALPHA = 0.7
# fixed bound S: no value in the paper (App. A.1 estimates S from benign norms). The code's utils/params.yaml has
# s_norm: 1000000, i.e. no bound (only used with diff_privacy, which is false)
BAGDASARYAN_FIXED_S_NORM = 1000000.0

bagdasaryan_variants = {
    "naive": ("naive backdoor", dict(baseline=True)),
    "replacement": ("model replacement", dict()),
    # original code's poison_step_lr=False: no lr / 10 steps during the attacker's training
    "replacement_no_step_lr": (
        "model replacement, constant lr",
        dict(poison_step_lr=False),
    ),
    # paper Sec. 4.2: a single attack, then the backdoor's decay over the remaining rounds.
    # Variants with poison_rounds keep their own timing, they run with attack_rounds "every"
    "replacement_single": ("model replacement, single round", dict(poison_rounds=(1,))),
    # poison the first half of the rounds, then the backdoor's decay over the second half
    "replacement_first_half": (
        "model replacement, first half of the rounds",
        dict(poison_rounds=range(1, BACKDOOR_N_ROUNDS // 2 + 1)),
    ),
    # supp. App. B.3: against Krum / Multi-Krum "no need to scale", factor 1 but adaptive poison_lr
    # scale_weights = n_attackers, set in bagdasaryan_attack
    "krum": ("no scaling (Krum variant)", dict()),
    "train_scale": (
        "train-and-scale, S = attackers' median norm",
        dict(s_norm="attackers_median"),
    ),
    "train_scale_max": (
        "train-and-scale, S = attackers' max norm",
        dict(s_norm="attackers_max"),
    ),
    "train_scale_mean": (
        "train-and-scale, S = attackers' mean norm",
        dict(s_norm="attackers_mean"),
    ),
    "train_scale_min": (
        "train-and-scale, S = attackers' min norm",
        dict(s_norm="attackers_min"),
    ),
    "train_scale_fixed": (
        f"train-and-scale, S = {BAGDASARYAN_FIXED_S_NORM}",
        dict(s_norm=BAGDASARYAN_FIXED_S_NORM),
    ),
    "constrain_scale_l2": (
        f"constrain-and-scale, L2 anomaly loss, alpha = {BAGDASARYAN_ALPHA}",
        dict(alpha_loss=BAGDASARYAN_ALPHA, anomaly_loss="l2"),
    ),
    "constrain_scale_cosine": (
        f"constrain-and-scale, cosine anomaly loss, alpha = {BAGDASARYAN_ALPHA}",
        dict(alpha_loss=BAGDASARYAN_ALPHA, anomaly_loss="cosine"),
    ),
}

# the attacker trains with local_train's optimizer (plain SGD, BATCH_SIZE, GRAD_CLIP_NORM; the original gives the
# attacker the benign momentum / decay, training.py#L61-L63, #L104-L106) and the paper's image-task attacker / benign
# ratios, arXiv v3 Sec. 5.3 p.7: lr 0.05 vs 0.1 (x 0.5), 6 vs 2 local epochs (x 3), lr / 10 every 2 of the 6 epochs
BAGDASARYAN_EPOCHS = 3 * LOCAL_EPOCHS
# LOCAL_STEPS: epochs of LOCAL_STEPS / LOCAL_EPOCHS batches, so the attacker takes 3 x LOCAL_STEPS steps
if LOCAL_STEPS is not None:
    assert LOCAL_STEPS % LOCAL_EPOCHS == 0, "LOCAL_STEPS must split into LOCAL_EPOCHS epochs for the Bagdasaryan attacker"
    BAGDASARYAN_SECRET_BATCHES = LOCAL_STEPS // LOCAL_EPOCHS


def bagdasaryan_attack(host, variant, settings):
    return lambda n_attackers: ModelReplacementAttack(
        n_attackers,
        N_CLIENTS,
        model,
        get_flat,
        set_flat,
        device,
        train_features,
        train_labels,
        client_partitions,
        BACKDOOR_TARGETS[host]["train_features"],
        exclude_rows=BACKDOOR_TARGETS[host]["train_rows"],
        name=f"MR-{variant}-{host}",
        backdoor_label=BACKDOOR_TARGETS[host]['target_label'],
        seed=SEED,
        poison_lr=0.5 * CLIENT_LEARNING_RATE,
        momentum=0.0,
        decay=0.0,
        grad_clip_norm=GRAD_CLIP_NORM,
        batch_size=BATCH_SIZE,
        retrain_poison=BAGDASARYAN_EPOCHS,
        # lr / 10 at the start of the second and last third of the epochs: epochs 3 and 5 of 6 as in the paper,
        # 2 and 3 of 3 here
        lr_milestones=(1 + BAGDASARYAN_EPOCHS // 3, 1 + 2 * BAGDASARYAN_EPOCHS // 3),
        # as many clean rows as an honest client holds, arXiv v3 Sec. 5.3 p.7: "640 benign images (same as everyone
        # else)" (params.yaml: 200 batches of 64). LOCAL_STEPS: as many batches as an honest client's epoch, the
        # attacker's rows reused when they are too few
        size_of_secret_dataset=(
            len(client_partitions[0]) // BATCH_SIZE if LOCAL_STEPS is None else BAGDASARYAN_SECRET_BATCHES
        ),
        wrap_secret_dataset=LOCAL_STEPS is not None,
        # Krum variant: factor scale_weights / n_attackers = 1. None = the default n_clients
        scale_weights=n_attackers if variant == "krum" else None,
        **settings,
    )


for host in BACKDOOR_TARGETS.keys():
    for variant, (description, settings) in bagdasaryan_variants.items():
        attack_rounds = "single_shot"
        # own poison_rounds: keeps "every", the grid at the end doesn't cross it with BACKDOOR_SCHEDULES
        fixed_schedule = False
        if "poison_rounds" in settings:
            attack_rounds = "every"
            fixed_schedule = True
        class_attacks[f"bagdasaryan_{variant}_{host}"] = {
            "attack": bagdasaryan_attack(host, variant, settings),
            "n_rounds": BACKDOOR_N_ROUNDS,
            "last_allowed_attack_round": BACKDOOR_N_ROUNDS - BACKDOOR_DECAY_ROUNDS,
            "attack_rounds": attack_rounds,
            "fixed_schedule": fixed_schedule,
            "attackers_per_attack_round": BACKDOOR_ATTACKERS_PER_ATTACK_ROUND,
            "display_name": f"Backdoor {host}, {description} (Bagdasaryan et al. 2020)",
            "target_host": host,
            "target_label": BACKDOOR_TARGETS[host]['target_label'],
            "test_features": BACKDOOR_TARGETS[host]["test_features"],
        }

# Wang et al. 2020 edge-case backdoor ("Attack of the Tails"), one entry per (variant, target host).
# The target's test rows give the success rate, every round and on the final model
from attacks.wangAttackTailsYes2020 import EdgeCaseBackdoorAttack


# fixed PGD radius: paper App. A p.19, Task 4 (Sentiment140, binary, LSTM) "fixed eps = 1.0 which lets it pass all
# defenses" (Task 1 CIFAR-10: eps = 2 = the NDC threshold)
TAILS_FIXED_EPS = 1.0

tails_variants = {
    "blackbox": ("black-box", dict(attack_method="blackbox")),
    # not one of the paper's three attacks: black-box training, then scaled like model replacement
    "blackbox_replacement": (
        "black-box with model replacement",
        dict(attack_method="blackbox", model_replacement=True),
    ),
    # the original's default of 2 local epochs for the attacker (the others use 1)
    "blackbox_2_epochs": (
        "black-box, 2 local epochs",
        dict(attack_method="blackbox", adversarial_local_training_period=2),
    ),
    "pgd": (
        "PGD, eps = attackers' median norm",
        dict(attack_method="pgd", eps="attackers_median"),
    ),
    "pgd_fixed_eps": (
        f"PGD, eps = {TAILS_FIXED_EPS}",
        dict(attack_method="pgd", eps=TAILS_FIXED_EPS),
    ),
    "pgd_replacement": (
        "PGD with model replacement, eps = attackers' median norm",
        dict(attack_method="pgd", eps="attackers_median", model_replacement=True),
    ),
    # paper Sec. 2 (c): project onto eps / scale, so the scaled update sent stays within eps
    "pgd_replacement_after_scaling": (
        "PGD with model replacement, eps after scaling = attackers' median norm",
        dict(
            attack_method="pgd",
            eps="attackers_median",
            model_replacement=True,
            eps_after_scaling=True,
        ),
    ),
    # the original fixed-pool code's factor: each attacker scaled by n_clients, not n_clients / n_attackers
    "pgd_replacement_full_scale": (
        "PGD with model replacement, scale = n_clients, eps = attackers' median norm",
        dict(
            attack_method="pgd",
            eps="attackers_median",
            model_replacement=True,
            scale_factor=N_CLIENTS,
        ),
    ),
}


def tails_attack(host, target, variant, settings):
    # the honest clients' LOCAL_EPOCHS unless the variant sets it (App. A: same hyper-parameters as honest clients);
    # LOCAL_STEPS: one pass of LOCAL_STEPS batches instead, as the honest clients
    settings = {"adversarial_local_training_period": LOCAL_EPOCHS, "local_steps": LOCAL_STEPS, **settings}
    return lambda n_attackers: EdgeCaseBackdoorAttack(
        n_attackers,
        N_CLIENTS,
        model,
        get_flat,
        set_flat,
        device,
        train_features,
        train_labels,
        client_partitions,
        target["train_features"],
        exclude_rows=target["train_rows"],
        name=f"Tails-{variant}-{host}",
        target_label=BACKDOOR_TARGETS[host]['target_label'],
        # the original trains the attacker with the honest clients' optimizer (fl_trainer.py#L446-L449,
        # "adversary needs same lr to hide with others"): here that is local_train's plain SGD with gradient clipping
        lr=CLIENT_LEARNING_RATE,
        adv_lr=CLIENT_LEARNING_RATE,
        momentum=0.0,
        weight_decay=0.0,
        grad_clip_norm=GRAD_CLIP_NORM,
        batch_size=BATCH_SIZE,
        seed=SEED,
        **settings,
    )


for host, target in BACKDOOR_TARGETS.items():
    for variant, (description, settings) in tails_variants.items():
        class_attacks[f"tails_{variant}_{host}"] = {
            "attack": tails_attack(host, target, variant, settings),
            "n_rounds": BACKDOOR_N_ROUNDS,
            "last_allowed_attack_round": BACKDOOR_N_ROUNDS - BACKDOOR_DECAY_ROUNDS,
            "attack_rounds": "every_k",
            "attackers_per_attack_round": BACKDOOR_ATTACKERS_PER_ATTACK_ROUND,
            "display_name": f"Edge-case backdoor {host}, {description} (Wang et al. 2020)",
            "target_host": host,
            "target_label": target["target_label"],
            "test_features": target["test_features"],
        }

# class-based aggregators (aggregator_class.py)
from aggregator_class import ByzFLAggregator, weighted_mean
from epprfl import EPPRFL

# EPPRFL's filter threshold lambda; the paper's 0.05 (Sec. VII-B5) keeps the plain display name, any other value goes
# into it, so RESUME and the analysis keep runs with different lambdas apart
# 0.0222 from lambda stage 1 (Oct 22 2026): 0.5625 * sqrt(m_min * m_max), m_h 0.0313 .. 0.0499 over the clean runs of
# all partitions (LOCAL_STEPS 54, lr 0.1, clip 1). The 0.05 check of stage 2 was worse on dirichlet_0.5.
# 0.016, 0.012, 0.010, 0.008: stricter checks on dirichlet_1.0 (10, run by mistake for 0.010: the filter keeps
# everyone, clipping only). The grid below (LAMBDA_STAGE 2) ran with 0.010 and 0.008; 0.006 down to 0.001: small check
# (stage 3)
# final grid: 0.005 as "epprfl", 0.001 as "epprfl_0.001" (lambda search, LAMBDA_STAGE 4, Oct 22 2026)
EPPRFL_LAMBDA = 0.005


# aggregator entry for one lambda, so a launch can run several; the display name keeps old results matching RESUME
def epprfl_entry(filter_lambda):
    display_name = "EPPRFL (Li et al. 2024)"
    if filter_lambda != 0.05:
        display_name = f"EPPRFL, lambda = {filter_lambda} (Li et al. 2024)"
    return {
        "aggregator": lambda: EPPRFL(random_generator, filter_lambda=filter_lambda),
        "display_name": display_name,
    }


aggregators = {
    "fedavg": {
        "aggregator": lambda: ByzFLAggregator("FedAvg", Average()),
        "display_name": "FedAvg (McMahan et al. 2017)",
    },
    "epprfl": epprfl_entry(EPPRFL_LAMBDA),
    "epprfl_0.001": epprfl_entry(0.001),
    "epprfl_0.006": epprfl_entry(0.006),
}
# aggregators run in the grid, keys of aggregators
# batch size 64 rerun
# RUN_AGGREGATORS = ["fedavg", "epprfl"]
# final grid: EPPRFL at lambda 0.005 and 0.001
# RUN_AGGREGATORS = ["fedavg", "epprfl", "epprfl_0.001"]
# train / test split rerun: EPPRFL at lambda 0.005 and 0.006
# RUN_AGGREGATORS = ["fedavg", "epprfl", "epprfl_0.006"]
# every-round backdoor run: FedAvg and EPPRFL at lambda 0.005
RUN_AGGREGATORS = ["fedavg", "epprfl"]


# success rate of every baseline target on its test rows (split "test"), None without baseline_targets
def baseline_success_rates(attack_entry, weights, split):
    if "baseline_targets" not in attack_entry:
        return None
    rates = {}
    for host, target in attack_entry["baseline_targets"].items():
        rates[host] = success_rate(
            target[f"{split}_features"], weights, target["target_label"]
        )
    return rates


# what a run reports: ("round", printed line), ("result", json line), ("error", traceback), ("done", rounds it didn't
# finish). A worker sends them to the parent (message_queue, set in init_worker), otherwise they're handled right here
message_queue = None
# the grid's progress bar (None: plain prints) and its finished runs
progress = None
finished_runs = 0
# rounds the current run has reported, so a crashed run's missing rounds still count for the progress bar
rounds_reported = 0


def send_message(kind, value):
    global rounds_reported
    if kind == "round":
        rounds_reported += 1
    if message_queue is None:
        handle_message(kind, value)
    else:
        message_queue.put((kind, value))


# only ever runs in the parent, so it is the only process writing RESULTS_PATH and parallel runs' lines never interleave
def handle_message(kind, value):
    global finished_runs
    if kind == "result":
        with open(RESULTS_PATH, "a") as results_file:
            results_file.write(value + "\n")
    elif kind == "done":
        finished_runs += 1
        if progress is not None:
            progress.update(value)
            progress.set_postfix_str(f"runs {finished_runs}/{len(jobs)}")
    elif progress is None:
        print(value, flush=True)
    else:
        # above the bar instead of through it
        progress.write(value)
        if kind == "round":
            progress.update(1)


# aggregator_entry / attack_entry: values of the aggregators / class_attacks dicts. Backdoor entries also hold
# target_host, target_label and the target's test rows; without them the success rate is None (untargeted attacks).
# The clean entry holds baseline_targets instead and measures all of them (clean_success_rates).
# The run is appended to RESULTS_PATH even if it crashes or is interrupted, with completed=False
def federated_learning_GOO(
    aggregator_entry, attack_entry, initial_weights
) -> RunResult:
    reseed()
    n_rounds = attack_entry.get("n_rounds", N_ROUNDS)
    attack_rounds = attack_entry.get("attack_rounds", ATTACK_ROUNDS)
    attackers_per_attack_round = attack_entry.get(
        "attackers_per_attack_round", ATTACKERS_PER_ATTACK_ROUND
    )
    warmup_rounds = attack_entry.get("warmup_rounds", WARMUP_ROUNDS)
    attack_every = attack_entry.get("attack_every", ATTACK_EVERY)
    # no attack after this round, None = until the end
    last_allowed_attack_round = attack_entry.get("last_allowed_attack_round")
    assert attack_rounds in ("every", "single_shot", "every_k"), (
        f"unknown attack_rounds {attack_rounds!r}"
    )

    n_attackers = NUM_MALICIOUS
    if attackers_per_attack_round is not None:
        n_attackers = min(attackers_per_attack_round, NUM_MALICIOUS)

    aggregator = aggregator_entry["aggregator"]()
    attack = attack_entry["attack"](n_attackers)
    # RUN_CONFIG is shared by all runs, each run writes it out before the next one changes it
    RUN_CONFIG["n_rounds"] = n_rounds

    initial_metrics, initial_non_target_metrics = evaluate_with_non_target(
        initial_weights, test_features, test_labels, test_non_target
    )
    initial_attack_success_rate = None
    if "test_features" in attack_entry:
        initial_attack_success_rate = success_rate(
            attack_entry["test_features"],
            initial_weights,
            attack_entry["target_label"],
        )
    # round 0: the starting model, before any training
    round_results: list[InitialRoundResult | RoundResult] = [
        {
            "round": 0,
            **initial_metrics,
            "non_target": initial_non_target_metrics,
            "attack_success_rate": initial_attack_success_rate,
            "clean_success_rates": baseline_success_rates(
                attack_entry, initial_weights, "test"
            ),
        }
    ]

    results: RunResult = {
        "aggregator": aggregator.name,
        "aggregator_display_name": aggregator_entry["display_name"],
        "aggregator_params": hyperparameters(aggregator),
        "attack": attack.name,
        "attack_display_name": attack_entry["display_name"],
        "attack_params": hyperparameters(attack),
        "target_host": attack_entry.get("target_host"),
        "attack_rounds": attack_rounds,
        "warmup_rounds": warmup_rounds,
        "attack_every": attack_every,
        "last_allowed_attack_round": last_allowed_attack_round,
        "attackers_per_attack_round": attackers_per_attack_round,
        "n_attackers": n_attackers,
        "config": RUN_CONFIG,
        "completed": False,
        "round_results": round_results,
    }

    try:
        global_weights = initial_weights
        # last round in which an attacker's update went into the aggregation, 0 = none yet
        last_attack_round = 0

        for round_index in range(n_rounds):
            start_time = time.time()

            round_number = round_index + 1
            if attack_rounds == "every":
                attack_round = True
            elif attack_rounds == "single_shot":
                attack_round = round_number == warmup_rounds + 1
            else:
                attack_round = round_number % attack_every == 0
            if last_allowed_attack_round is not None and round_number > last_allowed_attack_round:
                attack_round = False

            # TODO: some attacks don't need these updates -> we shouldn't waste compute
            updates = train_all_clients(global_weights, client_partitions)

            # The attack function changes the malicious clients' updates and says which ones it controls.
            # Outside attack rounds the malicious clients send their honest updates and count as benign
            attack_info = {}
            if attack_round:
                updates, malicious = attack.perform_attack(updates, global_weights)
                attack_info = getattr(attack, "last_info", {})
            else:
                malicious = np.array([], dtype=int)

            aggregation_start_time = time.time()
            aggregated_update, accepted = aggregator.aggregate(updates, global_weights)
            aggregation_seconds = time.time() - aggregation_start_time

            global_weights = global_weights + aggregated_update
            assert np.isfinite(global_weights).all(), (
                f"non-finite global weights after round {round_index}"
            )

            seconds = time.time() - start_time

            filtering = filtering_stats(accepted, malicious, len(updates))

            metrics, non_target_metrics = evaluate_with_non_target(
                global_weights, test_features, test_labels, test_non_target
            )

            attack_success_rate = None
            if "test_features" in attack_entry:
                attack_success_rate = success_rate(
                    attack_entry["test_features"],
                    global_weights,
                    attack_entry["target_label"],
                )

            # test success rate right after the last attack (peak) and BACKDOOR_DECAY_ROUNDS benign rounds later
            # (durability), the same distance after the last attack for every schedule. A later attack overwrites both
            if "test_features" in attack_entry:
                if len(malicious):
                    last_attack_round = round_number
                    results["last_attack_round"] = round_number
                    results["test_attack_success_rate_at_last_attack"] = success_rate(
                        attack_entry["test_features"],
                        global_weights,
                        attack_entry["target_label"],
                    )
                    results["test_attack_success_rate_after_decay"] = None
                elif last_attack_round and round_number == last_attack_round + BACKDOOR_DECAY_ROUNDS:
                    results["test_attack_success_rate_after_decay"] = success_rate(
                        attack_entry["test_features"],
                        global_weights,
                        attack_entry["target_label"],
                    )

            # high-water mark of the whole process (ru_maxrss is in KiB on Linux), so it never goes down
            # from one run to the next, and later runs in the same run.py session inherit earlier peaks
            process_peak_memory_mb = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
            )

            result: RoundResult = {
                "round": round_index + 1,
                **metrics,
                "non_target": non_target_metrics,
                "attack_success_rate": attack_success_rate,
                "clean_success_rates": baseline_success_rates(
                    attack_entry, global_weights, "test"
                ),
                **filtering,
                # an aggregator that accepted nobody leaves the model unchanged
                "round_skipped": accepted is not None and len(accepted) == 0,
                **update_stats(
                    updates,
                    malicious,
                    aggregated_update,
                    log_client_stats=LOG_CLIENT_STATS and aggregator.name == "EPPRFL",
                ),
                "attack_round": attack_round,
                "n_attackers_active": len(malicious),
                # the attackers' share of the plain mean, k / n (before any filtering)
                "attacker_weight": len(malicious) / len(updates),
                # optional per-round internals of the attack / aggregator, e.g. Min-Max's lamda
                **attack_info,
                **getattr(aggregator, "last_info", {}),
                "seconds": seconds,
                "aggregation_seconds": aggregation_seconds,
                "process_peak_memory_mb": process_peak_memory_mb,
            }
            round_results.append(result)

            # rejected=total(malicious among them)/number of malicious, '-' if the aggregator doesn't select
            rejected = (
                "-"
                if filtering["n_rejected"] is None
                else f"{filtering['n_rejected']}({filtering['n_malicious_rejected']}m)/{len(malicious)}m"
            )
            asr = "-" if attack_success_rate is None else f"{attack_success_rate:.3f}"
            send_message(
                "round",
                f"{attack.name:15s} {aggregator.name:9s} m={NUM_MALICIOUS} attackers={len(malicious)} round={round_index + 1}/{n_rounds} acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} auroc={metrics['auroc']:.3f} fpr={metrics['fpr']:.3f} asr={asr} rejected={rejected} s={seconds:.2f} agg_s={aggregation_seconds:.2f} rss={process_peak_memory_mb}MB",
            )

        # the held-out test set, only on the final model
        results["test"], results["test_non_target"] = evaluate_with_non_target(
            global_weights, test_features, test_labels, test_non_target
        )
        if "test_features" in attack_entry:
            results["test_attack_success_rate"] = success_rate(
                attack_entry["test_features"],
                global_weights,
                attack_entry["target_label"],
            )
        if "baseline_targets" in attack_entry:
            results["test_clean_success_rates"] = baseline_success_rates(
                attack_entry, global_weights, "test"
            )
        results["completed"] = True
    except Exception:
        results["error"] = traceback.format_exc()
        raise
    finally:
        send_message("result", json.dumps(results))
    return results


# ====================================================================
# Grid
# ====================================================================

# numbers of malicious clients (clients 0..NUM_MALICIOUS-1 are malicious): these shares of N_CLIENTS,
# at least 1, below 50%
MALICIOUS_SHARES = [0.05, 0.1, 0.2, 0.3, 0.4]
BACKDOOR_MALICIOUS_SHARES = [0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4]

# attack rounds of the backdoors, each overrides the entry's attack_rounds / warmup_rounds / attack_every
BACKDOOR_SCHEDULES = [
    {"attack_rounds": "single_shot", "warmup_rounds": 5},
    {"attack_rounds": "every_k", "attack_every": 10},
    {"attack_rounds": "every_k", "attack_every": 5},
    {"attack_rounds": "every"},
]

# thesis grid: untargeted attacks, every-round backdoors and their 50- and 70-round clean baselines
RUN_BASELINE = True
RUN_UNTARGETED = True
RUN_BACKDOORS = True

# keys of class_attacks
# every untargeted attack: Fang-Krum, Min-Max / Min-Sum with each dev_type (minmax_ndss = unit_vec)
RUN_UNTARGETED_ATTACKS = [
    "fang_krum",
    "minmax_ndss",
    # between unit_vec and std on both aggregators
    # "minmax_ndss_sign",
    "minmax_ndss_std",
    "minsum_ndss",
    # weak everywhere (<= 0.8 pp)
    # "minsum_ndss_sign",
    # weak everywhere (<= 0.8 pp)
    # "minsum_ndss_std",
]
# backdoor hosts: an edge-case (tail) target and a common one
RUN_HOSTS = ["stats.g.doubleclick.net", "ad-delivery.net"]
# every backdoor variant (bagdasaryan_variants, tails_variants), run as f"{variant}_{host}"
RUN_BACKDOOR_ATTACKS = [
    "bagdasaryan_naive",
    "bagdasaryan_replacement",
    # same results as bagdasaryan_replacement
    # "bagdasaryan_replacement_no_step_lr",
    # same attack as bagdasaryan_replacement with single_shot, 5 rounds earlier
    # "bagdasaryan_replacement_single",
    # same results as bagdasaryan_replacement
    # "bagdasaryan_replacement_first_half",
    # bit-identical to bagdasaryan_naive
    # "bagdasaryan_krum",
    # median S scales down, always 0
    # "bagdasaryan_train_scale",
    "bagdasaryan_train_scale_max",
    # equivalent to bagdasaryan_train_scale_max
    # "bagdasaryan_train_scale_mean",
    # equivalent to bagdasaryan_train_scale_max
    # "bagdasaryan_train_scale_min",
    # same run as bagdasaryan_replacement while bd_norm_sent < 1e6
    # "bagdasaryan_train_scale_fixed",
    # same results as the cosine one, the paper uses cosine
    # "bagdasaryan_constrain_scale_l2",
    "bagdasaryan_constrain_scale_cosine",
    # Tails: only the paper's three attacks, Sec. 2 (a) black-box, (b) PGD, (c) PGD + model replacement
    # same results as tails_pgd
    # "tails_blackbox",
    # "tails_blackbox_replacement",
    # "tails_blackbox_2_epochs",
    "tails_pgd",
    # "tails_pgd_fixed_eps",
    # "tails_pgd_replacement",
    # 0 everywhere like tails_pgd, tails_pgd is kept as the norm-bounded one
    # "tails_pgd_replacement_after_scaling",
    # "tails_pgd_replacement_full_scale",
]

# stage 1 subset, stage 2: the commented lines
# RUN_BACKDOOR_SHARES = [0.005, 0.05, 0.2, 0.4]
RUN_BACKDOOR_SHARES = [0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
# RUN_BACKDOOR_SHARES = BACKDOOR_MALICIOUS_SHARES
# RUN_BACKDOOR_SCHEDULES = [BACKDOOR_SCHEDULES[0], BACKDOOR_SCHEDULES[1]]
# single shot and every round
# RUN_BACKDOOR_SCHEDULES = [BACKDOOR_SCHEDULES[0], BACKDOOR_SCHEDULES[3]]
# every 10 rounds only
# RUN_BACKDOOR_SCHEDULES = [BACKDOOR_SCHEDULES[1]]
# attack every round
RUN_BACKDOOR_SCHEDULES = [BACKDOOR_SCHEDULES[3]]
# RUN_BACKDOOR_SCHEDULES = BACKDOOR_SCHEDULES

# client partitions (setup.py CLIENT_PARTITIONS), every job above runs once per partition:
# iid, dirichlet_0.5 / dirichlet_1.0 / dirichlet_10.0 (label skew), quantity (size skew), natural (whole sites per client)
# dirichlet_1.0: milder than dirichlet_0.5, same attacks hurt. dirichlet_10.0: within ~2 pp of iid
# RUN_PARTITIONS = ["iid", "dirichlet_0.5", "dirichlet_1.0", "dirichlet_10.0", "natural"]
# RUN_PARTITIONS = ["iid", "dirichlet_0.5", "natural"]
# final grid: dirichlet_1.0 too (the partition the lambda search tuned on)
# RUN_PARTITIONS = ["iid", "dirichlet_0.5", "natural", "dirichlet_1.0"]
# train / test split rerun: dirichlet_0.5 left out
RUN_PARTITIONS = ["iid", "natural", "dirichlet_1.0"]
# RUN_PARTITIONS = list(CLIENT_PARTITIONS)

MALICIOUS_RANGE = []
for share in MALICIOUS_SHARES:
    num_malicious = max(1, round(share * N_CLIENTS))
    num_malicious = min(num_malicious, (N_CLIENTS - 3) // 2)
    if num_malicious not in MALICIOUS_RANGE:
        MALICIOUS_RANGE.append(num_malicious)

BACKDOOR_MALICIOUS_RANGE = []
for share in RUN_BACKDOOR_SHARES:
    num_malicious = max(1, round(share * N_CLIENTS))
    num_malicious = min(num_malicious, (N_CLIENTS - 3) // 2)
    if num_malicious not in BACKDOOR_MALICIOUS_RANGE:
        BACKDOOR_MALICIOUS_RANGE.append(num_malicious)

# one job per run, in the order the runs used to go one after the other
jobs = []


def add_job(num_malicious, aggregator_entry, attack_entry):
    for partition in RUN_PARTITIONS:
        jobs.append(
            {
                "num_malicious": num_malicious,
                "aggregator_entry": aggregator_entry,
                "attack_entry": attack_entry,
                "n_rounds": attack_entry.get("n_rounds", N_ROUNDS),
                "partition": partition,
            }
        )


# clean baselines, as long as the attacked runs they are compared with (N_ROUNDS for the untargeted attacks,
# BACKDOOR_N_ROUNDS for the backdoors)
def add_baselines(n_rounds):
    attack_entry = {**class_attacks["none"], "n_rounds": n_rounds}
    for aggregator_name in RUN_AGGREGATORS:
        aggregator_entry = aggregators[aggregator_name]
        add_job(0, aggregator_entry, attack_entry)


# unbounded attacks, only on these aggregators, first in the job list so every shard runs them next
RUN_UNBOUNDED_ATTACKS = ["sign_flipping", "gaussian"]
# RUN_UNBOUNDED_AGGREGATORS = ["fedavg", "epprfl", "epprfl_0.006"]
RUN_UNBOUNDED_AGGREGATORS = ["fedavg", "epprfl"]
for num_malicious in MALICIOUS_RANGE:
    for attack_name in RUN_UNBOUNDED_ATTACKS:
        for aggregator_name in RUN_UNBOUNDED_AGGREGATORS:
            add_job(num_malicious, aggregators[aggregator_name], class_attacks[attack_name])

if RUN_BASELINE:
    if RUN_UNTARGETED:
        add_baselines(N_ROUNDS)
    if RUN_BACKDOORS:
        add_baselines(BACKDOOR_N_ROUNDS)

# untargeted: every round, N_ROUNDS
if RUN_UNTARGETED:
    for num_malicious in MALICIOUS_RANGE:
        for attack_name in RUN_UNTARGETED_ATTACKS:
            attack_entry = class_attacks[attack_name]
            for aggregator_name in RUN_AGGREGATORS:
                add_job(num_malicious, aggregators[aggregator_name], attack_entry)

# backdoors: BACKDOOR_N_ROUNDS, every schedule, except variants with their own poison_rounds (run once)
if RUN_BACKDOORS:
    for num_malicious in BACKDOOR_MALICIOUS_RANGE:
        for variant in RUN_BACKDOOR_ATTACKS:
            # one attacker: max / mean / min of one norm is the median, the same run as bagdasaryan_train_scale
            if num_malicious == 1 and variant in (
                "bagdasaryan_train_scale_max",
                "bagdasaryan_train_scale_mean",
                "bagdasaryan_train_scale_min",
            ):
                continue
            for host in RUN_HOSTS:
                attack_entry = class_attacks[f"{variant}_{host}"]
                schedules = RUN_BACKDOOR_SCHEDULES
                if attack_entry.get("fixed_schedule"):
                    schedules = [{}]
                for schedule in schedules:
                    # Tails: no single shot, the paper attacks repeatedly (Sec. 4.1); with every_k removed it attacks every round
                    if variant.startswith("tails") and schedule.get("attack_rounds") == "single_shot":
                        continue
                    for aggregator_name in RUN_AGGREGATORS:
                        add_job(num_malicious, aggregators[aggregator_name], {**attack_entry, **schedule})

# the runs needed most first: EPPRFL at lambda 0.005, then FedAvg, then lambda 0.006. Stable sort, so each keeps the grid
# order; every shard takes every n-th job, so all shards start on the 0.005 runs
RUN_ORDER = ["epprfl", "fedavg"]
job_order = {}
for position, aggregator_name in enumerate(RUN_ORDER):
    job_order[id(aggregators[aggregator_name])] = position
jobs.sort(key=lambda job: job_order[id(job["aggregator_entry"])])

# EPPRFL lambda calibration, replaces the grid above when set. Stage 1: clean runs, FedAvg + EPPRFL on every
# partition; EPPRFL's m_h there decides whether EPPRFL_LAMBDA stays 0.05. Stage 2 (RESUME=1): the attacks that
# break EPPRFL today, 30 % attackers (should be fixed: sign flipping, Min-Sum, Fang; must not break: Gaussian;
# control: Min-Max unit vector) and model replacement, 20 %, every round (must not break)
LAMBDA_STAGE = None
if LAMBDA_STAGE is not None:
    jobs = []
    stage_jobs = [("none", RUN_PARTITIONS, 0, {})]
    if LAMBDA_STAGE == 2:
        # overall picture for one lambda, launched once per EPPRFL_LAMBDA (0.010, 0.008); FedAvg doesn't depend on
        # lambda, RESUME skips it in the second launch
        stage_jobs = [
            # backdoors first (100 rounds), 20 % attackers, every round, common target
            ("bagdasaryan_replacement_ad-delivery.net", ["dirichlet_1.0"], 40, BACKDOOR_SCHEDULES[3]),
            ("bagdasaryan_naive_ad-delivery.net", ["dirichlet_1.0"], 40, BACKDOOR_SCHEDULES[3]),
            ("tails_pgd_ad-delivery.net", ["dirichlet_1.0"], 40, BACKDOOR_SCHEDULES[3]),
            ("none", ["dirichlet_1.0"], 0, {"n_rounds": BACKDOOR_N_ROUNDS}),
            # no attack on the other partitions
            ("none", ["iid", "natural", "dirichlet_0.5"], 0, {}),
            # 30 % on every other partition
            ("sign_flipping", ["iid", "natural", "dirichlet_0.5"], 60, {}),
            ("minmax_ndss", ["iid", "natural", "dirichlet_0.5"], 60, {}),
            ("minsum_ndss", ["iid", "natural", "dirichlet_0.5"], 60, {}),
            ("fang_krum", ["natural"], 60, {}),
            ("gaussian", ["natural"], 60, {}),
            # other shares on dirichlet_1.0 (10 % and 40 %)
            ("minmax_ndss", ["dirichlet_1.0"], 20, {}),
            ("minsum_ndss", ["dirichlet_1.0"], 20, {}),
            ("minmax_ndss", ["dirichlet_1.0"], 80, {}),
            ("minsum_ndss", ["dirichlet_1.0"], 80, {}),
            ("sign_flipping", ["dirichlet_1.0"], 80, {}),
        ]
    # the small lambda check: dirichlet_1.0, clean and four attacks at 30 %
    if LAMBDA_STAGE == 3:
        stage_jobs = [
            ("none", ["dirichlet_1.0"], 0, {}),
            ("sign_flipping", ["dirichlet_1.0"], 60, {}),
            ("minsum_ndss", ["dirichlet_1.0"], 60, {}),
            ("fang_krum", ["dirichlet_1.0"], 60, {}),
            ("minmax_ndss", ["dirichlet_1.0"], 60, {}),
        ]
    stage_aggregators = [aggregators["fedavg"], aggregators["epprfl"]]
    # the lambda search in one launch: clean gate on every partition first, then three attacks at 30 % on the
    # tuning partition dirichlet_1.0; each lambda gets its own entry. FedAvg once per job, RESUME skips the
    # existing runs (FedAvg, EPPRFL 0.001 / 0.005 on dirichlet_1.0)
    if LAMBDA_STAGE == 4:
        stage_jobs = [
            ("none", ["iid", "natural", "dirichlet_0.5", "dirichlet_1.0"], 0, {}),
            ("minmax_ndss", ["dirichlet_1.0"], 60, {}),
            ("minsum_ndss", ["dirichlet_1.0"], 60, {}),
            ("fang_krum", ["dirichlet_1.0"], 60, {}),
        ]
        stage_aggregators = [aggregators["fedavg"]]
        for filter_lambda in [0.0005, 0.00075, 0.001, 0.0015, 0.005]:
            stage_aggregators.append(epprfl_entry(filter_lambda))
    for attack_name, partitions, num_malicious, schedule in stage_jobs:
        attack_entry = {**class_attacks[attack_name], **schedule}
        for partition in partitions:
            for aggregator_entry in stage_aggregators:
                jobs.append(
                    {
                        "num_malicious": num_malicious,
                        "aggregator_entry": aggregator_entry,
                        "attack_entry": attack_entry,
                        "n_rounds": attack_entry.get("n_rounds", N_ROUNDS),
                        "partition": partition,
                    }
                )

if JOB_SHARD:
    shard_index, n_shards = JOB_SHARD.split("/")
    jobs = jobs[int(shard_index)::int(n_shards)]


# what identifies a run, from a job and from a result line
def job_key(job):
    attack_entry = job["attack_entry"]
    return (
        attack_entry["display_name"],
        job["aggregator_entry"]["display_name"],
        job["num_malicious"],
        job["partition"],
        job["n_rounds"],
        attack_entry.get("attack_rounds", ATTACK_ROUNDS),
        attack_entry.get("warmup_rounds", WARMUP_ROUNDS),
        attack_entry.get("attack_every", ATTACK_EVERY),
    )


def result_key(result):
    return (
        result["attack_display_name"],
        result["aggregator_display_name"],
        result["config"]["n_malicious"],
        result["config"]["partition_scheme"],
        result["config"]["n_rounds"],
        result["attack_rounds"],
        result["warmup_rounds"],
        result["attack_every"],
    )


# RESUME=1: skip the jobs that already have a completed run with the same BATCH_SIZE, GRAD_CLIP_NORM,
# CLIENT_LEARNING_RATE, LOCAL_EPOCHS, LOCAL_STEPS and test split size in any results/*.jsonl
# (the size keeps runs of the old train / validation / test split, n_test 72,276, apart)
if os.environ.get("RESUME"):
    job_keys = [job_key(job) for job in jobs]
    assert len(set(job_keys)) == len(jobs), "two jobs share a key, RESUME can't tell them apart"
    completed_keys = set()
    results_dir = os.path.dirname(RESULTS_PATH)
    for file_name in os.listdir(results_dir):
        if not file_name.endswith(".jsonl"):
            continue
        with open(os.path.join(results_dir, file_name)) as results_file:
            for line in results_file:
                result = json.loads(line)
                if (
                    result["completed"]
                    and result["config"].get("batch_size") == BATCH_SIZE
                    and result["config"].get("grad_clip_norm") == GRAD_CLIP_NORM
                    and result["config"].get("learning_rate") == CLIENT_LEARNING_RATE
                    and result["config"].get("local_epochs") == LOCAL_EPOCHS
                    and result["config"].get("local_steps") == LOCAL_STEPS
                    and result["config"].get("n_test") == len(test_features)
                ):
                    completed_keys.add(result_key(result))
    remaining_jobs = []
    for job, key in zip(jobs, job_keys):
        if key not in completed_keys:
            remaining_jobs.append(job)
    print(f"RESUME: {len(jobs) - len(remaining_jobs)} of {len(jobs)} jobs already completed", flush=True)
    jobs = remaining_jobs

# CHECK_TRAINING=1: train_all_clients against local_train on the smallest and the largest client of every partition in
# RUN_PARTITIONS, then exit. Dropout is off, since the two paths draw different dropout masks. Learning rate 0.001:
# the rounding differences of the two paths (1e-6 after one step) grow with lr and steps (measured Oct 22: lr 0.1,
# 54 steps -> 28 % of the largest entry; lr 0.01, 324 steps -> 1 %), which is chaotic training, not a code difference
if os.environ.get("CHECK_TRAINING"):
    run_learning_rate = CLIENT_LEARNING_RATE
    CLIENT_LEARNING_RATE = 0.001
    optimizer.param_groups[0]["lr"] = CLIENT_LEARNING_RATE
    dropouts = []
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            dropouts.append((module, module.p))
            module.p = 0.0
    for partition_name in RUN_PARTITIONS:
        partitions = CLIENT_PARTITIONS[partition_name]
        sizes = [len(partition) for partition in partitions]
        checked = [partitions[int(np.argmin(sizes))], partitions[int(np.argmax(sizes))]]
        reseed()
        batched = train_all_clients(initial_weights, checked)
        reseed()
        reference = []
        for partition in checked:
            reference.append(local_train(initial_weights, train_features[partition], train_labels[partition]))
        reference = np.stack(reference)
        error = float(np.abs(batched - reference).max() / np.abs(reference).max())
        client_errors = np.linalg.norm(batched - reference, axis=1) / np.linalg.norm(reference, axis=1)
        print(f"CHECK_TRAINING {partition_name}: clients of {min(sizes)} and {max(sizes)} rows, "
              f"max difference {error:.2e} of the largest update entry, relative L2 per client "
              f"{client_errors[0]:.1e} / {client_errors[1]:.1e}", flush=True)
        assert error < 1e-3, f"train_all_clients differs from local_train on {partition_name}"
    for module, rate in dropouts:
        module.p = rate
    CLIENT_LEARNING_RATE = run_learning_rate
    optimizer.param_groups[0]["lr"] = CLIENT_LEARNING_RATE
    print("CHECK_TRAINING passed", flush=True)
    raise SystemExit(0)

total_rounds = 0
for job in jobs:
    total_rounds += job["n_rounds"]

# set in a worker by Ctrl+C
interrupted = False


# runs jobs[job_index], in a worker or in this process. The attack / aggregator factories read the global NUM_MALICIOUS
# when called, so it is set first. A crashed run is already in RESULTS_PATH with its error; the next job goes on
def run_job(job_index):
    global NUM_MALICIOUS, client_partitions, rounds_reported, interrupted
    if interrupted:
        return
    job = jobs[job_index]
    NUM_MALICIOUS = job["num_malicious"]
    RUN_CONFIG["n_malicious"] = NUM_MALICIOUS
    # the attack factories and the round loop read the global client_partitions
    client_partitions = CLIENT_PARTITIONS[job["partition"]]
    RUN_CONFIG["partition_scheme"] = job["partition"]
    RUN_CONFIG["dirichlet_alpha"] = None
    if job["partition"].startswith("dirichlet_"):
        RUN_CONFIG["dirichlet_alpha"] = float(job["partition"].split("_")[1])
    client_sizes = []
    client_positive_rates = []
    for partition in client_partitions:
        client_sizes.append(len(partition))
        client_positive_rates.append(float(train_labels[partition].mean()))
    RUN_CONFIG["client_sizes"] = client_sizes
    RUN_CONFIG["client_positive_rates"] = client_positive_rates
    rounds_reported = 0
    try:
        federated_learning_GOO(job["aggregator_entry"], job["attack_entry"], initial_weights)
    except Exception:
        send_message("error", traceback.format_exc())
    except KeyboardInterrupt:
        if message_queue is None:
            raise
        # Ctrl+C reaches the workers too: the stopped run's result line is already sent, skip the jobs still queued
        interrupted = True
        return
    send_message("done", job["n_rounds"] - rounds_reported)


def init_worker(queue_from_parent, threads):
    global message_queue
    message_queue = queue_from_parent
    torch.set_num_threads(threads)
    # numpy's BLAS
    threadpool_limits(threads)


# ====================================================================
# Workers
# ====================================================================

# None = as many as cores, memory and jobs allow. The environment variable N_WORKERS overrides both
N_WORKERS = None
# per worker: the 200 client updates and the worker's own copy of every page of data / model / libraries it writes to
# (copy-on-write after fork). Rounded up to stay safe
MEMORY_PER_WORKER_GB = 3
# left for the system and this (parent) process
FREE_CORES = 1
FREE_MEMORY_GB = 2

if hasattr(os, "sched_getaffinity"):
    usable_cores = len(os.sched_getaffinity(0))
else:
    usable_cores = os.cpu_count()
if os.environ.get("SLURM_CPUS_PER_TASK"):
    usable_cores = min(usable_cores, int(os.environ["SLURM_CPUS_PER_TASK"]))

available_memory_gb = None
if os.path.exists("/proc/meminfo"):
    with open("/proc/meminfo") as meminfo_file:
        for line in meminfo_file:
            if line.startswith("MemAvailable:"):
                available_memory_gb = int(line.split()[1]) / 1024**2

# cgroup v2 limits of this process (docker --cpus / --memory, SLURM)
cgroup_dir = "/sys/fs/cgroup"
if os.path.exists("/proc/self/cgroup"):
    with open("/proc/self/cgroup") as cgroup_file:
        for line in cgroup_file:
            if line.startswith("0::"):
                cgroup_dir = "/sys/fs/cgroup" + line[3:].strip()
if os.path.exists(os.path.join(cgroup_dir, "cpu.max")):
    with open(os.path.join(cgroup_dir, "cpu.max")) as cpu_max_file:
        quota, period = cpu_max_file.read().split()
    if quota != "max":
        usable_cores = min(usable_cores, max(1, int(quota) // int(period)))
if os.path.exists(os.path.join(cgroup_dir, "memory.max")):
    with open(os.path.join(cgroup_dir, "memory.max")) as memory_max_file:
        memory_max = memory_max_file.read().strip()
    if memory_max != "max":
        with open(os.path.join(cgroup_dir, "memory.current")) as memory_current_file:
            memory_current = int(memory_current_file.read())
        cgroup_free_gb = (int(memory_max) - memory_current) / 1024**3
        if available_memory_gb is None or cgroup_free_gb < available_memory_gb:
            available_memory_gb = cgroup_free_gb

workers_by_cores = max(1, usable_cores - FREE_CORES)
workers_by_memory = workers_by_cores
if available_memory_gb is not None:
    workers_by_memory = max(1, int((available_memory_gb - FREE_MEMORY_GB) // MEMORY_PER_WORKER_GB))

if os.environ.get("N_WORKERS"):
    n_workers = int(os.environ["N_WORKERS"])
    reason = "environment variable N_WORKERS"
elif N_WORKERS is not None:
    n_workers = N_WORKERS
    reason = "N_WORKERS in experiment.py"
else:
    n_workers = min(workers_by_cores, workers_by_memory, len(jobs))
    reason = f"{usable_cores} cores - {FREE_CORES} free -> {workers_by_cores}, "
    if available_memory_gb is not None:
        reason += f"{available_memory_gb:.1f} GB available - {FREE_MEMORY_GB} free / {MEMORY_PER_WORKER_GB} GB per worker -> {workers_by_memory}, "
    reason += f"{len(jobs)} jobs"
# setup.py already started CUDA (model.to(device)), and a forked child can't use it: run the jobs in this process
if str(device) != "cpu" and n_workers > 1:
    n_workers = 1
    reason = f"device {device}, CUDA can't be used from forked workers"
n_workers = max(1, min(n_workers, len(jobs)))
threads_per_worker = max(1, usable_cores // n_workers)

print(f"{len(jobs)} runs, {total_rounds} rounds", flush=True)
if str(device) != "cpu":
    print(f"1 worker, in this process ({reason})", flush=True)
else:
    print(f"{n_workers} workers x {threads_per_worker} threads ({reason})", flush=True)

# ====================================================================
# Run the jobs
# ====================================================================

# smoothing=0: the ETA is elapsed / rounds done * rounds left.
# On CPU even a single worker is forked, so this process never trains and stays safe to fork from (setup.py)
if str(device) != "cpu":
    torch.set_num_threads(threads_per_worker)
    progress = tqdm(total=total_rounds, unit="round", smoothing=0)
    try:
        for job_index in range(len(jobs)):
            run_job(job_index)
    finally:
        progress.close()
        progress = None
else:
    # fork: the workers inherit the loaded data, model and jobs (copy-on-write) instead of rerunning setup.py
    context = multiprocessing.get_context("fork")
    worker_messages = context.Queue()
    executor = ProcessPoolExecutor(
        n_workers,
        mp_context=context,
        initializer=init_worker,
        initargs=(worker_messages, threads_per_worker),
    )
    futures = []
    try:
        # the workers are forked at the first submit, before the progress bar starts its thread
        for job_index in range(len(jobs)):
            futures.append(executor.submit(run_job, job_index))
        progress = tqdm(total=total_rounds, unit="round", smoothing=0)
        while finished_runs < len(jobs):
            try:
                kind, value = worker_messages.get(timeout=10)
            except queue.Empty:
                # a worker killed without a word (e.g. out of memory) fails all futures, their runs never finish
                all_done = True
                for future in futures:
                    if not future.done():
                        all_done = False
                if all_done:
                    break
                continue
            handle_message(kind, value)
    except KeyboardInterrupt:
        # the workers got Ctrl+C too: collect the result lines of the runs they stopped (completed=False)
        while True:
            try:
                kind, value = worker_messages.get(timeout=5)
            except queue.Empty:
                break
            handle_message(kind, value)
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        if progress is not None:
            progress.close()
            progress = None

    for job_index in range(len(futures)):
        future = futures[job_index]
        if future.done() and not future.cancelled() and future.exception() is not None:
            print(f"job {job_index} lost: {future.exception()!r}", flush=True)
