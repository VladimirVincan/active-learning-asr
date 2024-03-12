import os
import shutil
from pathlib import Path

import datasets
import pandas as pd
import soundfile
from datasets import load_dataset, Audio

download_path = os.path.join(os.getcwd(), 'librispeech_cache')
cache_path = os.path.join(os.getcwd(), 'librispeech_cache')

if os.path.exists(download_path):
    shutil.rmtree(download_path)
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
try:
    os.makedirs(download_path)
except:
    pass
try:
    os.makedirs(cache_path)
except:
    pass
try:
    os.makedirs(os.path.join(download_path, 'clips'))
except:
    pass

datasets.config.DOWNLOADED_DATASETS_PATH = Path(str(download_path))
datasets.config.HF_DATASETS_CACHE = Path(str(download_path))

dataset = load_dataset("librispeech_asr", 'clean', split="train.100", streaming=True)
dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
dataset = iter(dataset)

csv_name = os.path.join(download_path, 'clean_train_100.csv')

rows = []
for i, row in enumerate(dataset):
    print(i)
    path = os.path.join(cache_path, 'clips', row['audio']['path'])
    path = path.split('.')[0] + '.wav'
    soundfile.write(path, row['audio']['array'], row['audio']['sampling_rate'])

    del row['audio']
    rows.append(row)

    if i == 10:
        break

df = pd.DataFrame(rows)
df.to_csv(csv_name, index=False, header=True)
