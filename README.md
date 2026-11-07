# robustness-evaluation

Federated learning poisoning evaluation of FedAvg and EPPRFL under untargeted and backdoor attacks.

## Setup

Needs git-lfs and Docker. Every step below uses this one image.

```bash
git lfs install
git clone https://github.com/ppr-wt-fl/robustness-evaluation.git
cd robustness-evaluation
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init AdFlushAutokeras YOPO
GIT_LFS_SKIP_SMUDGE=1 git -C YOPO submodule update --init AdFlush
docker build -t poisoning-eval poisoning-eval
```

## FL simulation

Runs the every-round backdoor grid (414 runs). Writes to `poisoning-eval/results/`. Add `--gpus all` to use a GPU.

```bash
docker run --rm -v "$PWD":/robustness-evaluation -w /robustness-evaluation/poisoning-eval \
  --user "$(id -u):$(id -g)" poisoning-eval
```

## Thesis plots and numbers

Reads the thesis runs in `data/thesis-lambda0.005/` and writes to `plots/thesis-lambda0.005/`.

```bash
cd poisoning-results-evaluation
docker run --rm -v "$PWD":/work -w /work --user "$(id -u):$(id -g)" -e TITLES=0 --entrypoint python poisoning-eval figures.py data/thesis-lambda0.005
docker run --rm -v "$PWD":/work -w /work --user "$(id -u):$(id -g)" --entrypoint python poisoning-eval analyze.py data/thesis-lambda0.005 > plots/thesis-lambda0.005/analysis.txt
```

## Target selection

Chooses the backdoor targets and overwrites `poisoning-eval/edge_sets/`, which already holds the thesis targets.
Needs the caches from starting the FL simulation (a bit of a weird loop, I'm sorry).

```bash
cd poisoning-eval
for step in rank density select; do
  docker run --rm -v "$PWD":/work -w /work --user "$(id -u):$(id -g)" --entrypoint python poisoning-eval edge_case_set/calculation.py $step
done
docker run --rm -v "$PWD":/work -w /work --user "$(id -u):$(id -g)" --entrypoint python poisoning-eval edge_case_set/plot.py
```

## Notes

- `dataset/features_adlfush.csv` is cut from a 3.7 GB CSV that isn't in this repo, so `poisoning-eval/reduce_dataset.py` can't run from a clone.
- The crawl lives in the `YOPO` submodule, see `crawl/README.md`.
