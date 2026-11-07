# shape of the results experiment.py writes: one RunResult per JSON line in results/*.jsonl
from typing import NotRequired, TypedDict


# evaluate(): metrics of one model on one split
class Metrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1: float
    fnr: float
    fpr: float
    auroc: float
    youden_tpr: float
    youden_fpr: float
    youden_threshold_logit: float
    tp: int
    fp: int
    tn: int
    fn: int


# filtering_stats(): all None if the aggregator doesn't select (e.g. ByzFL wrappers)
class FilteringStats(TypedDict):
    accepted_clients: list[int] | None
    n_rejected: int | None
    n_malicious_rejected: int | None
    n_benign_rejected: int | None
    n_malicious_accepted: int | None
    detection_precision: float | None
    detection_recall: float | None


# update_stats()
class UpdateStats(TypedDict):
    benign_norm_mean: float
    benign_norm_max: float
    malicious_norm_mean: float | None
    aggregate_norm: float
    aggregate_distance_to_benign_mean: float
    aggregate_cosine_to_benign_mean: float | None
    # LOG_CLIENT_STATS, EPPRFL only: per client, in client order
    client_norms: NotRequired[list[float]]
    client_cosines_to_benign_mean: NotRequired[list[float]]


# round 0: the starting model, before any training
class InitialRoundResult(Metrics):
    round: int
    # the same metrics without the rows of any BACKDOOR_TARGETS host (there a successful backdoor is also a main-task error)
    non_target: Metrics
    # share of the backdoor target's test rows classified as the target label, None for untargeted attacks
    attack_success_rate: float | None
    # clean run only: the same share for every backdoor target, {host: rate}, the baseline without poisoning; None otherwise
    clean_success_rates: dict[str, float] | None


class RoundResult(InitialRoundResult, FilteringStats, UpdateStats):
    round_skipped: bool
    # whether the schedule (attack_rounds) let the attackers attack this round
    attack_round: bool
    # attackers whose updates were replaced this round, and their share k / n of the plain mean
    n_attackers_active: int
    attacker_weight: float
    seconds: float
    aggregation_seconds: float
    process_peak_memory_mb: int
    # last_info of the attack / aggregator, only present for the ones that set it
    lamda: NotRequired[float]  # Min-Max
    scale_factor: NotRequired[float]  # Bagdasaryan / Wang: boost factor of the attacker's update
    epprfl_distances: NotRequired[list[float]]  # EPPRFL: d_i of every client, in client order


# RUN_CONFIG: settings shared by all runs of one execution of experiment.py
class RunConfig(TypedDict):
    n_clients: int
    n_rounds: int  # of this run: BACKDOOR_N_ROUNDS for backdoor entries, else N_ROUNDS
    n_malicious: int
    partition_scheme: str
    dirichlet_alpha: float | None
    quantity_beta: float
    client_sizes: list[int]
    client_positive_rates: list[float]
    learning_rate: float
    grad_clip_norm: float
    batch_size: int
    local_epochs: int
    local_steps: int | None  # None: LOCAL_EPOCHS epochs, else this many steps per client
    seed: int
    split_seed: int
    device: str
    model_path: str
    dataset_path: str
    n_train: int
    n_test: int


# one (attack, aggregator) run
class RunResult(TypedDict):
    aggregator: str
    aggregator_display_name: str
    aggregator_params: dict[str, int | float | str | bool]
    attack: str
    attack_display_name: str
    attack_params: dict[str, int | float | str | bool]
    # host of the backdoor target, None for untargeted attacks and the clean run
    target_host: str | None
    # attack schedule of the run ("every" | "single_shot" | "every_k") and its settings
    attack_rounds: str
    warmup_rounds: int
    attack_every: int
    # backdoors: no attack after this round (BACKDOOR_N_ROUNDS - BACKDOOR_DECAY_ROUNDS), None = until the end
    last_allowed_attack_round: int | None
    # setting, None = all n_malicious; n_attackers: the attackers per attack round it gave
    attackers_per_attack_round: int | None
    n_attackers: int
    config: RunConfig
    # False if the run crashed or was interrupted
    completed: bool
    round_results: list[InitialRoundResult | RoundResult]
    # held-out test set metrics of the final model, only when completed
    test: NotRequired[Metrics]
    # the same without the rows of any BACKDOOR_TARGETS host
    test_non_target: NotRequired[Metrics]
    # share of the backdoor target's test rows classified as the target label, final model, backdoor attacks only
    test_attack_success_rate: NotRequired[float]
    # backdoor attacks that attacked: the last round with an attacker in the aggregation, the test success rate
    # right after it, and BACKDOOR_DECAY_ROUNDS rounds later (None if the run ended before)
    last_attack_round: NotRequired[int]
    test_attack_success_rate_at_last_attack: NotRequired[float]
    test_attack_success_rate_after_decay: NotRequired[float | None]
    # clean run only: the same on every backdoor target's test rows, {host: rate}
    test_clean_success_rates: NotRequired[dict[str, float]]
    # traceback of the exception, only when the run crashed
    error: NotRequired[str]
