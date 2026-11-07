# Source: Bagdasaryan et al., "How To Backdoor Federated Learning" (AISTATS 2020), official code (MIT license):
#   https://github.com/ebagdasa/backdoor_federated_learning (commit 3f068b819f017b9eef1e19f43f572e391398c90a)
# The author's successor framework (MIT license) confirms scaling the update (not the weights) and clipping only after
# scaling, but scales every compromised user by the full factor, without the / k used here and in the original:
#   https://github.com/ebagdasa/backdoors101/blob/9f48fbbb496aaed4ba696494950a5d71ee82a80c/attack.py#L126-L128
#   https://github.com/ebagdasa/backdoors101/blob/9f48fbbb496aaed4ba696494950a5d71ee82a80c/training.py#L98-L100
#
# Ported from their per-tensor state_dict loops to our flat weight vectors. Changes to the original:
#   - binary task: BCE-with-logits on the single logit instead of cross_entropy
#   - backdoor inputs: instead of a few CIFAR images inserted into every batch (plus N(0, 0.01) image noise,
#     not meaningful for tabular features), backdoor_per_batch rows sampled from backdoor_train_features
#   - secret (clean) dataset drawn from the attackers' partitions instead of the whole CIFAR training set
#     (image_helper.py#L83-L121); a last partial batch is dropped, the original always has full batches
#   - benign clients may hold backdoor inputs with their true label (our main setting, a naturally occurring target);
#     the original keeps them away from all participants (image_helper.py#L61-L62), which the paper calls
#     "not essential" (Sec. 5.1). Only the attackers' clean data excludes them (exclude_rows)
#   - the attacker's checks (initial / early-stop backdoor accuracy and loss, main-task accuracy) use held-out slices
#     (check_fraction) of its own backdoor and clean data. The original uses its held-out poison_images_test (which is
#     also its backdoor evaluation set) and the clean CIFAR test set; the success rate is measured in experiment.py
#   - the original trains one poisoned model that stands for all k compromised participants and multiplies it by k
#     (training.py#L316-L322); here all k attackers send that same update as their own rows (same aggregate)
#   - s_norm (train-and-scale) only clips the update after scaling, as in the paper (Sec. 4.3, supp. App. B.2
#     "does not clip during his local training"); the code's diff_privacy mode also clips gradients (training.py#L217)
#     and projects during training (training.py#L220-L233)
#   - optimizer (experiment.py): the honest clients' optimizer (plain SGD, batch, gradient clipping), as the original
#     uses the benign momentum / decay for the attacker (training.py#L61-L63, #L104-L106), with the paper's image-task
#     attacker / benign ratios (arXiv v3 Sec. 5.3 p.7: lr 0.05 vs 0.1, 6 vs 2 epochs, lr / 10 every 2 epochs) instead
#     of params.yaml's absolute values (lr 0.05, 15 epochs, momentum 0.9, no clipping for images)
#   - own RNGs: batch orders come from a separate numpy generator and the torch RNG is forked during the attacker's
#     training, so honest clients see the same batch orders and dropout masks as with no attack

import math

import numpy as np
import torch
import torch.nn as nn

from .attack_class import Attack

# statistics for s_norm='attackers_<name>', over the norms of the attackers' honest updates
NORM_STATISTICS = {'max': np.max, 'mean': np.mean, 'median': np.median, 'min': np.min}


# Algorithm 1 (constrain-and-scale; AISTATS p.4, Algorithm 2 in arXiv v3). Every variant of the paper is a setting:
#   naive baseline          baseline=True                           (no scaling, no adaptive poison_lr)
#   model replacement       alpha_loss=1, scale_weights=n_clients   (gamma = n / eta, eta = 1 for our plain mean)
#   train-and-scale         alpha_loss=1, s_norm=S                  (code's reading of Eq. 5: scale, then shrink to S
#                                                                    if larger, i.e. factor min(gamma / k, S / ||X - G||))
#   constrain-and-scale     alpha_loss<1, anomaly_loss='l2'|'cosine'  (L_ano of Eq. 4: L2 distance to the global model,
#                                                                    or 1 - cos(submitted model, G), arXiv v3 App. A.2)
# Krum / Multi-Krum (supp. App. B.3, "no need to scale"): scale_weights=n_attackers gives factor 1 but keeps the
# adaptive poison_lr, which baseline=True turns off (training.py#L92).
#
# Caller contract:
#   - exclude_rows: indices into train_features of the backdoor inputs (with their true label), removed from the
#     attackers' clean data, otherwise they would train their own backdoor inputs as both labels
#     (the original also keeps them out of the attacker's clean data, image_helper.py#L103-L105)
# s_norm: a float bound, or 'attackers_<statistic>' = that statistic of the norms of the attackers' own honest updates
# this round, their estimate of the benign norm distribution (arXiv v3 App. A.1 p.15 "Estimating S" (2): "estimate the
# distribution of weight norms among the benign participants by training multiple local models"):
#   'attackers_max'     the paper's choice, "S is set near the upper bound of this distribution"
#   'attackers_mean'    as in the paper's example, "the mean of norms for word-prediction models trained on popular
#                       words as input and rare words as output ... cuts out only the top 5% of the benign updates"
#   'attackers_median'  ours, a central estimate like the mean but robust to one outlying attacker norm
#   'attackers_min'     ours, stays below a median-norm bound unless all attackers' norms are above it
# anomaly_loss: L_ano in alpha * L_class + (1 - alpha) * L_ano (Eq. 4)
#   'l2'      ||X - G||_2 (helper.py#L106-L117 model_dist_norm_var, the only L_ano training.py uses, #L167)
#   'cosine'  1 - cos(L, G) over the flat weight vector (arXiv v3 App. A.2 p.15: "we use L_ano = 1 - cos(L, G^t)"),
#             L the model this attacker submits, G + clip_rate * (X - G) with perform_attack's clip_rate. The paper
#             doesn't say which L; its detector compares the scaled submitted model with G^t (same paragraph,
#             cos(L~_m^{t+1}, G^t)) and Fig. 13 plots "Cosine similarity (submitted model and global model)". The code's
#             cos_sim_loss (helper.py#L120-L131, never called) also scales, by the full scale_weights (= clip_rate with
#             words.yaml's 1 adversary), and multiplies by 1e3, which is not in Eq. 4 and left out here. The per-layer
#             model_cosine_similarity (helper.py#L135-L159, 100x update, 1e3, never called) isn't used either.
#             1 - cos is tiny when ||G|| >> ||L - G|| (Fig. 13: about 5e-4..7e-3). s_norm clipping, applied only after
#             training, is not part of L
# poison_rounds: the 1-based rounds in which the attackers poison (params.yaml poison_epochs, training.py#L80-L81),
# e.g. (1,) for the paper's single-shot attack (Sec. 4.2). None = every round. In the other rounds the attackers
# send their honest updates and count as benign for the filtering stats
class ModelReplacementAttack(Attack):

    def __init__(self,
                 n_attackers, n_clients, model, get_flat, set_flat, device,
                 train_features, train_labels, client_partitions,
                 backdoor_train_features, exclude_rows=None,
                 name='ModelReplacement', backdoor_label=0.0,
                 scale_weights=None, alpha_loss=1.0, s_norm=None, baseline=False, poison_rounds=None, anomaly_loss='l2',

                 # defaults from the original code (utils/params.yaml; momentum / decay are the benign ones,
                 # training.py#L104-L106), except backdoor_per_batch. experiment.py overrides them with the honest
                 # clients' optimizer and the paper's attacker / benign ratios (see there)
                 poison_lr=0.05, momentum=0.9, decay=0.0005, retrain_poison=15, poison_step_lr=True,
                 batch_size=64, size_of_secret_dataset=200,
                 # epochs (1-based) at whose start the lr is divided by 10; None = the code's 0.2 / 0.8 * retrain_poison
                 # (training.py#L107-L110)
                 lr_milestones=None,
                 # None = c / 64 of each batch with the paper's c = 20 (AISTATS Sec. 5.1 p.6, arXiv v3 Sec. 5.1 p.6:
                 # "c = 20 backdoor images per batch of size 64"); the code's params.yaml inserts 7
                 backdoor_per_batch=None,
                 # per-batch gradient clipping as local_train does; None = no clipping, like the original for images
                 # (it clips the attacker only where it clips benign clients, text: training.py#L235-L238, #L367-L371)
                 grad_clip_norm=None,
                 # ours: share of the attacker's data held out for its own checks, seed of the attacker's RNG
                 check_fraction=0.2, seed=0,
                 # ours: too few rows for size_of_secret_dataset batches -> reuse them (shuffles chained) instead of
                 # training on fewer batches
                 wrap_secret_dataset=False):
        self.name = name
        self.n_attackers = n_attackers
        self.n_clients = n_clients
        self.model = model
        self.get_flat = get_flat
        self.set_flat = set_flat
        self.device = device
        # [seed, 1]: a different stream than experiment.py's random_generator, which is seeded with SEED alone
        self.random_generator = np.random.default_rng([seed, 1])
        self.backdoor_label = float(backdoor_label)
        # gamma = n / eta (Sec. 4.2)
        self.scale_weights = float(n_clients if scale_weights is None else scale_weights)
        self.alpha_loss = alpha_loss
        assert anomaly_loss in ('l2', 'cosine'), f'unknown anomaly_loss {anomaly_loss!r}'
        self.anomaly_loss = anomaly_loss
        assert not isinstance(s_norm, str) or s_norm in {f'attackers_{name}' for name in NORM_STATISTICS}, f'unknown s_norm {s_norm!r}'
        self.s_norm = s_norm
        self.baseline = baseline
        self.poison_rounds = None if poison_rounds is None else tuple(poison_rounds)
        # rounds seen so far: perform_attack is called once per round, and a new instance per run
        self.round = 0
        self.poison_lr = poison_lr
        self.momentum = momentum
        self.decay = decay
        self.retrain_poison = retrain_poison
        self.poison_step_lr = poison_step_lr
        self.batch_size = batch_size
        self.size_of_secret_dataset = size_of_secret_dataset
        # original rule: ceil so current PyTorch decays where 1.0 did (see _train_poisoned)
        if lr_milestones is None:
            lr_milestones = (math.ceil(0.2 * retrain_poison), math.ceil(0.8 * retrain_poison))
        self.lr_milestones = tuple(lr_milestones)
        self.backdoor_per_batch = round(20 / 64 * batch_size) if backdoor_per_batch is None else backdoor_per_batch
        self.grad_clip_norm = grad_clip_norm
        self.check_fraction = check_fraction
        self.seed = seed

        # the attacker's clean data, split into held-out check rows and the secret dataset of
        # size_of_secret_dataset batches, drawn once (image_helper.py poison_dataset)
        attacker_rows = np.concatenate(client_partitions[:n_attackers])
        if exclude_rows is not None:
            attacker_rows = np.setdiff1d(attacker_rows, exclude_rows)
        attacker_rows = self.random_generator.permutation(attacker_rows)
        n_check = int(len(attacker_rows) * check_fraction)
        check_rows = attacker_rows[:n_check]
        secret_size = size_of_secret_dataset * batch_size
        secret_rows = attacker_rows[n_check:n_check + secret_size]
        if wrap_secret_dataset and len(secret_rows) < secret_size:
            shuffles = [secret_rows]
            n_rows = len(secret_rows)
            while n_rows < secret_size:
                shuffles.append(self.random_generator.permutation(secret_rows))
                n_rows += len(secret_rows)
            secret_rows = np.concatenate(shuffles)[:secret_size]

        self.check_features = train_features[check_rows]
        self.check_labels = train_labels[check_rows]
        self.clean_features = train_features[secret_rows]
        self.clean_labels = train_labels[secret_rows]
        assert n_check >= 1, 'no clean rows held out for the attacker\'s checks'
        assert len(self.clean_features) >= batch_size, 'secret dataset smaller than one batch'

        # the attacker's backdoor inputs, split into held-out check rows and training rows
        backdoor_order = self.random_generator.permutation(len(backdoor_train_features))
        n_backdoor_check = int(len(backdoor_order) * check_fraction)
        backdoor_check_rows = backdoor_order[:n_backdoor_check]
        backdoor_train_rows = backdoor_order[n_backdoor_check:]

        self.backdoor_check_features = backdoor_train_features[backdoor_check_rows]
        self.backdoor_train_features = backdoor_train_features[backdoor_train_rows]
        assert n_backdoor_check >= 1, 'no backdoor rows held out for the attacker\'s checks'
        assert len(self.backdoor_train_features) >= 1, 'no backdoor rows left to train on'

    # (loss, accuracy in %) of the model as it currently is, like test / test_poison in training.py
    @torch.no_grad()
    def _test(self, features, labels):
        # a mean over no rows is NaN, which would silently disable the lr cut / early stop and end up in the results
        assert len(features) > 0, 'nothing to test on'
        self.model.eval()
        features = self._to_device(features)
        labels = self._to_device(labels)

        # chunked so a big set never has to go through the model at once
        logits = []
        for chunk in features.split(65536):
            logits.append(self.model(chunk)[:, 0])
        logits = torch.cat(logits)

        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
        predictions = (logits > 0).float()
        accuracy = 100.0 * (predictions == labels).float().mean().item()
        return loss, accuracy

    def _to_device(self, array):
        return torch.from_numpy(array).to(self.device)

    def _backdoor_labels(self, n):
        return np.full(n, self.backdoor_label, np.float32)

    # all model parameters as one vector, in the order of get_flat, still attached to the graph
    def _flat_parameters(self):
        return torch.cat([layer.view(-1) for layer in self.model.parameters()])

    # indices of the clean rows per batch for one epoch: shuffled, full batches only
    def _epoch_batches(self, n_rows):
        n_batches = n_rows // self.batch_size
        order = self.random_generator.permutation(n_rows)
        order = order[:n_batches * self.batch_size]
        return torch.from_numpy(order).split(self.batch_size)

    # replaces the first positions of the batch by backdoor inputs with the backdoor label
    def _insert_backdoor(self, data, targets, backdoor_features):
        n_poison = min(self.backdoor_per_batch, len(data))
        with_replacement = n_poison > len(backdoor_features)
        poison = self.random_generator.choice(len(backdoor_features), n_poison, replace=with_replacement)
        data[:n_poison] = backdoor_features[torch.from_numpy(poison)]
        targets[:n_poison] = self.backdoor_label

    # training.py#L80-L262: train the poisoned model X from the global model G, returns X - G
    def _train_poisoned(self, global_weights):
        self.set_flat(global_weights)
        parameters = list(self.model.parameters())
        target_params_variables = self._to_device(np.asarray(global_weights, np.float32))
        backdoor_check_labels = self._backdoor_labels(len(self.backdoor_check_features))

        # FIXME: is held out data ok?
        _, attack_success_percentage = self._test(self.backdoor_check_features, backdoor_check_labels)
        _, acc_initial = self._test(self.check_features, self.check_labels)

        poison_learning_rate = self.poison_lr

        # hardcoded values taken from original code
        if not self.baseline:
            if attack_success_percentage > 20:
                poison_learning_rate /= 50
            if attack_success_percentage > 60:
                poison_learning_rate /= 100

        retrain_no_times = self.retrain_poison
        poison_optimizer = torch.optim.SGD(parameters, lr=poison_learning_rate, momentum=self.momentum, weight_decay=self.decay)
        # lr_milestones: epochs at whose start the lr drops (the scheduler is stepped before each epoch, so
        # milestone e means epochs >= e train at lr / 10). The original's float milestones 0.2 / 0.8 * retrain_no_times
        # decayed in PyTorch 1.0 once last_epoch >= milestone, current PyTorch only on an exact match, so the default
        # rounds them up to the epoch where 1.0 decayed (__init__)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer, milestones=list(self.lr_milestones), gamma=0.1)

        features = self._to_device(self.clean_features)
        labels = self._to_device(self.clean_labels)
        backdoor_features = self._to_device(self.backdoor_train_features)
        epochs_run = 0
        for internal_epoch in range(1, retrain_no_times + 1):
            epochs_run = internal_epoch
            # stepped before the epoch, as in the original (current PyTorch warns about the order once)
            if self.poison_step_lr:
                scheduler.step()
            self.model.train()

            # full batches only, like the original's batches of exactly batch_size
            for batch in self._epoch_batches(len(features)):
                data = features[batch].clone()
                targets = labels[batch].clone()
                self._insert_backdoor(data, targets, backdoor_features)

                poison_optimizer.zero_grad()
                logits = self.model(data)[:, 0]
                class_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
                distance = self._flat_parameters() - target_params_variables
                if self.anomaly_loss == 'l2':
                    # helper.py#L106-L117 model_dist_norm_var
                    anomaly_loss = torch.norm(distance, 2)
                else:
                    # 1 - cos(L, G) with L = G + clip_rate * (X - G), the model submitted (arXiv v3 App. A.2 p.15)
                    submitted = target_params_variables + self._clip_rate() * distance
                    anomaly_loss = 1 - nn.functional.cosine_similarity(submitted, target_params_variables, dim=0)
                loss = self.alpha_loss * class_loss + (1 - self.alpha_loss) * anomaly_loss
                loss.backward()
                # as local_train in experiment.py
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(parameters, self.grad_clip_norm)
                poison_optimizer.step()

            _, acc = self._test(self.check_features, self.check_labels)
            loss_p, _ = self._test(self.backdoor_check_features, backdoor_check_labels)

            # early stop once the backdoor is learned, unless the main task got worse
            if loss_p <= 0.0001:
                # backdoor learned, but main-task accuracy fell below the global model's (acc_initial): keep training
                # on the mixed batches to recover it, with an extra scheduler step so the lr decays sooner
                # (training.py#L247-L251, images only in the original)
                if acc < acc_initial:
                    if self.poison_step_lr:
                        scheduler.step()
                    continue
                break

        self.train_info = {'bd_poison_lr': poison_learning_rate, 'bd_epochs': epochs_run, 'bd_initial_backdoor_acc': attack_success_percentage}
        return self.get_flat() - global_weights

    def perform_attack(self, updates: np.ndarray, global_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.round += 1
        if self.poison_rounds is not None and self.round not in self.poison_rounds:
            # experiment.py logs last_info every round, so clear the previous poisoned round's values
            self.last_info = {}
            return updates, np.arange(0)

        malicious = np.arange(self.n_attackers)
        # forked so the attacker's dropout doesn't shift the torch RNG the honest clients use
        with torch.random.fork_rng(devices=self._cuda_devices()):
            malicious_update = self._train_poisoned(global_weights)
        # the attacker may train without gradient clipping (grad_clip_norm=None); fail here rather than in the aggregate
        assert np.isfinite(malicious_update).all(), 'attacker training diverged (non-finite update)'
        norm_before = float(np.linalg.norm(malicious_update))

        # training.py#L263-L279: L = G + (X - G) * scale_weights / k
        if not self.baseline:
            malicious_update = malicious_update * self._clip_rate()

        # training.py#L281-L296: clip the distance to G to s_norm
        if self.s_norm is not None:
            s_norm = self._norm_bound(updates[malicious])
            model_norm = np.linalg.norm(malicious_update)
            if model_norm > s_norm:
                malicious_update = malicious_update * (s_norm / model_norm)

        # logged per round by experiment.py
        self.last_info = {
            **self.train_info,
            # factor X - G was multiplied by before the s_norm clipping (scale_weights / k, 1 for the naive baseline)
            'scale_factor': self._clip_rate(),
            'bd_norm_before_scaling': norm_before,
            'bd_norm_sent': float(np.linalg.norm(malicious_update)),
        }

        # attackers replace their own rows (the first n_attackers) with the same update
        updates = updates.copy()
        updates[malicious] = malicious_update
        return updates, malicious

    # factor of X - G each attacker submits before the s_norm clipping: scale_weights / k, 1 for the naive baseline
    def _clip_rate(self):
        return 1.0 if self.baseline else self.scale_weights / self.n_attackers

    # s_norm as a number. 'attackers_<statistic>': the attackers estimate the bound from their own honest updates
    def _norm_bound(self, attacker_updates):
        if isinstance(self.s_norm, str):
            statistic = NORM_STATISTICS[self.s_norm.removeprefix('attackers_')]
            return float(statistic(np.linalg.norm(attacker_updates, axis=1)))
        return float(self.s_norm)

    # the GPU whose RNG fork_rng has to save and restore, none on CPU
    def _cuda_devices(self):
        device = torch.device(self.device)
        if device.type != 'cuda':
            return []
        if device.index is not None:
            return [device.index]
        return [torch.cuda.current_device()]
