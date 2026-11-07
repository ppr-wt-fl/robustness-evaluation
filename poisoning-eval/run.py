import os

# slow part (imports, data, partitions, model)
from setup import *

EXPERIMENT_PATH = os.path.join(os.path.dirname(__file__), 'experiment.py')

# fast part, runs once; errors propagate so the process exits non-zero
with open(EXPERIMENT_PATH) as experiment_file:
    exec(compile(experiment_file.read(), EXPERIMENT_PATH, 'exec'), globals())
