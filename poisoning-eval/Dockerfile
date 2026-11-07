# Python 3.13: latest TensorFlow has no wheels for 3.14 yet
FROM python:3.13-slim

# Run from the repo root, mounting the whole repo so ../dataset, ../AdFlushAutokeras and ../YOPO resolve:
#   docker build -t poisoning-eval poisoning-eval
#   docker run -d --name pe-bench \
#     -v "$PWD":/robustness-evaluation \
#     -w /robustness-evaluation/poisoning-eval \
#     --user "$(id -u):$(id -g)" \
#     --entrypoint sleep \
#     poisoning-eval infinity
#   docker exec pe-bench python run.py

# TF_USE_LEGACY_KERAS: model was saved with Keras 2.15 (AutoKeras 1.1.0); make tf.keras resolve to tf_keras
# HOME: the container runs as the host user, which has no home inside the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_USE_LEGACY_KERAS=1 \
    HOME=/tmp

# C compiler: on CUDA, torch's backward of batched matmuls (vmap in train_all_clients) JIT-compiles a Triton kernel
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /poisoning-eval

# GPU image: docker build --build-arg REQUIREMENTS=requirements-gpu.txt -t poisoning-eval-gpu poisoning-eval,
# run with --gpus all (experiment.py then runs its jobs in one process per container, see JOB_SHARD)
ARG REQUIREMENTS=requirements.txt
COPY ${REQUIREMENTS} requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "run.py"]
