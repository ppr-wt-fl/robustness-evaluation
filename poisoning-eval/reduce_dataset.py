import os
import json
import pandas as pd

# keeps only the columns setup.py reads (plus the *_webgraph ones and keyword_char_present), so the dataset fits under
# GitHub LFS' 2 GiB limit

DATASET_DIR = os.path.join(os.path.dirname(__file__), '../dataset')
SOURCE_PATH = os.path.join(DATASET_DIR, 'features_raw_all_feature.csv')
TARGET_PATH = os.path.join(DATASET_DIR, 'features_adlfush.csv')

# content_policy_type name -> number the model gets, as in setup.py
CONTENT_POLICY_TYPES_PATH = os.path.join(os.path.dirname(__file__), '../YOPO/AdFlush/source/json/content_type_dict.json')
with open(CONTENT_POLICY_TYPES_PATH) as content_policy_types_file:
    CONTENT_POLICY_TYPES = json.load(content_policy_types_file)

# same as setup.py's FEATURE_COLUMNS (not imported: importing setup runs the whole setup)
FEATURE_COLUMNS = [
    "content_policy_type",
    "url_length",
    "brackettodot",
    "is_third_party",
    "keyword_raw_present",
    "num_get_storage",
    "num_set_storage",
    "num_get_cookie",
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

# the other columns setup.py reads: site identifiers for the split, and the label
USED_COLUMNS = ["top_level_url", "visit_id", "name", *FEATURE_COLUMNS, "CLASS", "keyword_char_present"]

header = pd.read_csv(SOURCE_PATH, nrows=0).columns
missing = [column for column in USED_COLUMNS if column not in header]
if missing:
    raise SystemExit(f'columns missing from {SOURCE_PATH}: {missing}')

# in the source file's order
columns = [column for column in header if column in USED_COLUMNS or column.endswith('_webgraph')]
print(f'keeping {len(columns)} of {len(header)} columns: {columns}', flush=True)

# content_policy_type stays the name, the number goes in a new column right after it
# (empty and unknown names get 0, like setup.py's fillna(0))
output_columns = list(columns)
output_columns.insert(output_columns.index('content_policy_type') + 1, 'content_policy_type_value')

# chunked so the 3.7 GB file never has to be in memory at once; dtype=str copies values unchanged
rows = 0
for index, chunk in enumerate(pd.read_csv(SOURCE_PATH, usecols=columns, dtype=str, keep_default_na=False, chunksize=500_000)):
    chunk['content_policy_type_value'] = chunk['content_policy_type'].map(CONTENT_POLICY_TYPES).fillna(0)
    chunk[output_columns].to_csv(TARGET_PATH, mode='w' if index == 0 else 'a', header=index == 0, index=False)
    rows += len(chunk)
    print(f'{rows} rows', flush=True)

print(f'wrote {TARGET_PATH} ({os.path.getsize(TARGET_PATH) / 1e9:.2f} GB)', flush=True)
