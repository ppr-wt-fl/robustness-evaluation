# Source: Wang et al., "Attack of the Tails: Yes, You Really Can Backdoor Federated Learning" (NeurIPS 2020), official
# code linked in the paper (Sec. 4, footnote 1), MIT license (its LICENSE file reads "Copyright (c) 2017 liukuang"):
#   https://github.com/kamikazekartik/OOD_Federated_Learning (commit 90969695cf0f1c5b009744fc2ccd4fbde817b8cb)
#   train() with the L2 projection:          fl_trainer.py#L122-L194
#   attacker's optimizers, training, scaling: fl_trainer.py#L447-L487 (fixed-frequency), #L795-L845 (fixed-pool)
#   D' = 100 edge-case + 400 clean points:    utils.py#L447-L493 (Southwest, Task 1)
#   hyperparameters:                          README.md#L55-L82 (Southwest sample command), simulated_averaging.py#L34-L95
#
# Ported from their per-client nn.Module copies to our flat weight vectors. Changes to the original:
#   - binary task: BCE-with-logits on the single logit instead of CrossEntropyLoss, target_label 0 (non-ad)
#   - edge-case set: rows of one ad host given by the caller (edge_train_features), not a hand-collected image set.
#     The paper's GMM density check (Sec. 2, Fig. 2) has no published code and is not part of the attack; it is
#     run offline in the analysis, not here
#   - D' (clean part) drawn from the attackers' partitions minus exclude_rows instead of the whole CIFAR-10 training set
#   - benign clients may hold edge-case rows with their true label (our main setting); in the paper's edge-case
#     setting only the adversary holds them (Sec. 4 "Edge-case vs non-edge-case attacks", Fig. 4 varies this)
#   - as in the fixed-pool mode, each attacker trains its own copy of the global model on the shared D'
#     (fl_trainer.py#L754, #L804-L829); here all k attackers participate every round
#   - model replacement factor: the original's total_num_dps_per_round / num_dps_poisoned_dataset is 1 / (attacker's
#     FedAvg weight). Our mean gives the k attackers together weight k / n, so the factor is n_clients / n_attackers;
#     the fixed-pool code applies the full factor to each attacker, which with k attackers would overshoot k-fold
#   - no learning-rate / eps decay over rounds (args_gamma ** (flr - 1)): our honest clients use a constant lr
#   - optional gradient clipping (grad_clip_norm, ours) so the black-box attacker can train exactly like our honest
#     clients ("standard local training, without modification", Sec. 2 (a)); off by default, the original has none
#   - eps_after_scaling (ours, off by default): project onto eps / scale so the update sent stays within eps, the
#     paper's reading of PGD with replacement (Sec. 2 (c)); the original projects onto eps and scales afterwards
#   - eps='attackers_median' (ours): eps = median norm of the attackers' own honest updates this round
#   - own RNGs: D' sampling and batch orders come from a separate numpy generator and the torch RNG is forked during
#     the attacker's training, so honest clients see the same batch orders and dropout masks as with no attack
#   - not ported: the l_inf projection, prox_attack and the attack_case 'normal-case' / 'almost-edge-case' modes

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .attack_class import Attack

# USER notes
# remove adv_lr and advanced optimizer, doesn't seem to matter now

# The paper's three attacks (Sec. 2) as settings, named like the original's command-line arguments:
#   black-box                     attack_method='blackbox'                          (plain training on D')
#   PGD                           attack_method='pgd', eps=...                      (project onto ||w - w_g|| <= eps)
#   PGD with model replacement    attack_method='pgd', eps=..., model_replacement=True
#
# Caller contract:
#   - exclude_rows: indices into train_features of the edge-case rows (with their true label), removed from the
#     attackers' clean data, otherwise they would train their own edge-case inputs as both labels
class EdgeCaseBackdoorAttack(Attack):

    def __init__(self,
                 n_attackers: int, n_clients: int, model: nn.Module, get_flat: Callable[[], np.ndarray], set_flat: Callable[[np.ndarray], None], device: str,
                 train_features: np.ndarray, train_labels: np.ndarray, client_partitions: list[np.ndarray],
                 edge_train_features: np.ndarray, exclude_rows: np.ndarray | None = None,
                 name: str = 'EdgeCase', target_label: float = 0.0,
                 attack_method: str = 'blackbox', model_replacement: bool = False, eps: float | str | None = None,

                 # defaults from the Southwest sample command (README.md) and utils.py, except where noted
                 lr: float = 0.02, adv_lr: float = 0.02, momentum: float = 0.9, weight_decay: float = 1e-4,
                 adversarial_local_training_period: int = 2, batch_size: int = 32, project_frequency: int = 10,
                 num_sampled_poisoned_data_points: int = 100, num_sampled_data_points: int = 400,
                 # ours: replacement factor (None = n_clients / n_attackers), see the header for the rest
                 scale_factor: float | None = None, eps_after_scaling: bool = False, grad_clip_norm: float | None = None, seed: int = 0,
                 # ours: one pass of local_steps full batches (shuffles of D' chained) instead of
                 # adversarial_local_training_period epochs, as experiment.py's honest clients with LOCAL_STEPS
                 local_steps: int | None = None) -> None:
        assert attack_method in ('blackbox', 'pgd'), f'unknown attack_method {attack_method}'
        assert attack_method == 'blackbox' or eps is not None, 'PGD needs eps'
        self.name = name
        self.n_attackers = n_attackers
        self.n_clients = n_clients
        self.model = model
        self.get_flat = get_flat
        self.set_flat = set_flat
        self.device = device
        # [seed, 1]: a different stream than experiment.py's random_generator, which is seeded with SEED alone
        self.random_generator = np.random.default_rng([seed, 1])
        self.target_label = float(target_label)
        self.attack_method = attack_method
        self.model_replacement = model_replacement
        self.eps = eps
        self.lr = lr
        self.adv_lr = adv_lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.adversarial_local_training_period = adversarial_local_training_period
        self.batch_size = batch_size
        self.project_frequency = project_frequency
        self.eps_after_scaling = eps_after_scaling
        self.grad_clip_norm = grad_clip_norm
        self.seed = seed
        self.local_steps = local_steps

        # 1 / (the attackers' combined weight in our unweighted mean)
        if scale_factor is None:
            self.scale_factor = n_clients / n_attackers
        else:
            self.scale_factor = float(scale_factor)

        # utils.py#L447, #L455-L462: num_sampled_poisoned_data_points edge-case points, relabelled, drawn once
        assert len(edge_train_features) >= num_sampled_poisoned_data_points, 'fewer edge-case rows than num_sampled_poisoned_data_points'
        edge_rows = self.random_generator.choice(len(edge_train_features), num_sampled_poisoned_data_points, replace=False)
        edge_features = edge_train_features[edge_rows]
        edge_labels = np.full(len(edge_features), self.target_label, np.float32)

        # utils.py#L472-L476: num_sampled_data_points clean points, drawn once (here from the attackers' own data)
        attacker_rows = np.concatenate(client_partitions[:n_attackers])
        if exclude_rows is not None:
            attacker_rows = np.setdiff1d(attacker_rows, exclude_rows)
        assert len(attacker_rows) >= num_sampled_data_points, 'fewer clean attacker rows than num_sampled_data_points'
        clean_rows = self.random_generator.choice(attacker_rows, num_sampled_data_points, replace=False)
        clean_features = train_features[clean_rows]
        clean_labels = train_labels[clean_rows]

        # utils.py#L483-L484: D' = clean points followed by the edge-case points
        self.poisoned_features = np.concatenate([clean_features, edge_features])
        self.poisoned_labels = np.concatenate([clean_labels, edge_labels])

        # sizes, for the results (hyperparameters() keeps plain int / float attributes)
        self.num_dps_poisoned_dataset = int(len(self.poisoned_features))
        self.n_edge_available = int(len(edge_train_features))
        self.n_clean_available = int(len(attacker_rows))

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(self.device)

    # indices of D' per batch for one epoch: shuffled, last batch may be smaller (DataLoader(shuffle=True), utils.py#L493).
    # n_batches (ours): that many full batches, shuffles chained until there are enough rows
    def _epoch_batches(self, n_rows: int, n_batches: int | None = None) -> tuple[torch.Tensor, ...]:
        if n_batches is None:
            order = self.random_generator.permutation(n_rows)
        else:
            n_needed = n_batches * self.batch_size
            shuffles = []
            for shuffle in range(-(-n_needed // n_rows)):
                shuffles.append(self.random_generator.permutation(n_rows))
            order = np.concatenate(shuffles)[:n_needed]
        return torch.from_numpy(order).split(self.batch_size)

    # fl_trainer.py#L122-L194 train(), one epoch. pgd_attack: step with adv_optimizer, then project onto the L2 ball
    # of radius eps around the global model every project_frequency batches and on the last one
    def _train(self, optimizer: torch.optim.Optimizer, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], pgd_attack: bool = False, eps: float | None = 5e-4, model_original: list[torch.Tensor] | None = None, adv_optimizer: torch.optim.Optimizer | None = None, n_batches: int | None = None) -> None:
        model = self.model
        model.train()

        features = self._to_device(self.poisoned_features)
        labels = self._to_device(self.poisoned_labels)
        batches = self._epoch_batches(len(features), n_batches)

        for batch_idx, batch in enumerate(batches):
            data = features[batch]
            target = labels[batch]
            optimizer.zero_grad()
            if pgd_attack:
                adv_optimizer.zero_grad()
            output = model(data)[:, 0]
            loss = criterion(output, target)
            loss.backward()

            # ours: optional, to train like our honest clients (experiment.py local_train)
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)

            if not pgd_attack:
                optimizer.step()
            else:
                # do l2_projection
                adv_optimizer.step()
                w = list(model.parameters())
                w_vec = parameters_to_vector(w)
                model_original_vec = parameters_to_vector(model_original)
                # make sure you project on last iteration otherwise, high LR pushes you really far
                if (batch_idx % self.project_frequency == 0 or batch_idx == len(batches) - 1) and (torch.norm(w_vec - model_original_vec) > eps):
                    # project back into norm ball
                    w_proj_vec = eps * (w_vec - model_original_vec) / torch.norm(w_vec - model_original_vec) + model_original_vec
                    # plug w_proj back into model
                    vector_to_parameters(w_proj_vec, w)
                    self.n_projections += 1

    # fl_trainer.py#L447-L470: the attacker's local training from the global model, returns w - w_g
    def _train_poisoned(self, global_weights: np.ndarray, eps: float | None) -> np.ndarray:
        self.set_flat(global_weights)
        parameters = list(self.model.parameters())
        # the global model as a list of tensors, like the original's list(self.net_avg.parameters())
        model_original = [parameter.detach().clone() for parameter in parameters]
        self.n_projections = 0

        criterion = nn.functional.binary_cross_entropy_with_logits
        optimizer = torch.optim.SGD(parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # "looks like adversary needs same lr to hide with others" (fl_trainer.py#L449)
        adv_optimizer = torch.optim.SGD(parameters, lr=self.adv_lr, momentum=self.momentum, weight_decay=self.weight_decay)

        pgd_attack = self.attack_method == 'pgd'
        if self.local_steps is not None:
            self._train(optimizer, criterion, pgd_attack=pgd_attack, eps=eps, model_original=model_original, adv_optimizer=adv_optimizer, n_batches=self.local_steps)
        else:
            for e in range(1, self.adversarial_local_training_period + 1):
                self._train(optimizer, criterion, pgd_attack=pgd_attack, eps=eps, model_original=model_original, adv_optimizer=adv_optimizer)

        return self.get_flat() - global_weights

    def perform_attack(self, updates: np.ndarray, global_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        malicious = np.arange(self.n_attackers)

        eps = None
        if self.attack_method == 'pgd':
            eps = self._eps(updates[malicious])
            # ours (eps_after_scaling): radius eps / scale, so the scaled update stays within eps (paper Sec. 2 (c))
            if self.model_replacement and self.eps_after_scaling:
                eps = eps / self.scale_factor

        updates = updates.copy()
        norms_trained = []
        norms_sent = []
        n_projections = 0

        # each attacker (the first n_attackers rows) trains its own copy of the global model on the shared D'
        # forked so the attackers' dropout doesn't shift the torch RNG the honest clients use
        with torch.random.fork_rng(devices=self._cuda_devices()):
            for j in malicious:
                malicious_update = self._train_poisoned(global_weights, eps)
                # the attacker trains without gradient clipping by default, like the original; fail here rather than in the aggregate
                assert np.isfinite(malicious_update).all(), 'attacker training diverged (non-finite update)'
                norms_trained.append(np.linalg.norm(malicious_update))
                n_projections += self.n_projections

                # fl_trainer.py#L836-L843: w_g + (w - w_g) * scale
                if self.model_replacement:
                    malicious_update = malicious_update * self.scale_factor
                norms_sent.append(np.linalg.norm(malicious_update))

                updates[j] = malicious_update

        # logged per round by experiment.py
        scale_factor = 1.0
        if self.model_replacement:
            scale_factor = self.scale_factor
        self.last_info = {
            'scale_factor': scale_factor,
            'tails_eps': eps,
            'tails_n_projections': n_projections,
            'tails_norm_trained': float(np.mean(norms_trained)),
            'tails_norm_sent': float(np.mean(norms_sent)),
        }

        return updates, malicious

    # eps as a number. 'attackers_median' (ours): the attackers estimate it from their own honest updates
    def _eps(self, attacker_updates: np.ndarray) -> float:
        if self.eps == 'attackers_median':
            norms = np.linalg.norm(attacker_updates, axis=1)
            return float(np.median(norms))
        return float(self.eps)

    # the GPU whose RNG fork_rng has to save and restore, none on CPU
    def _cuda_devices(self) -> list[int]:
        device = torch.device(self.device)
        if device.type != 'cuda':
            return []
        if device.index is not None:
            return [device.index]
        return [torch.cuda.current_device()]
