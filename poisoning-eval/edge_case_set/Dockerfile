# From poisoning-eval/:
#   docker build -t edge-case-set edge_case_set
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/poisoning-eval edge-case-set edge_case_set/calculation.py rank|density|select [args]
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/poisoning-eval edge-case-set edge_case_set/plot.py [args]
FROM python:3.13-slim@sha256:bf44cdfcb76cd3b41e879bc058fc37ec5872002ccfde7fcb765e218cde0cd79c

COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

WORKDIR /poisoning-eval

ENTRYPOINT ["python"]
