import os
import shutil
from pathlib import Path

import datasets
import pandas as pd
import soundfile
from datasets import load_dataset

download_path = os.path.join(os.getcwd(), 'librispeech')
cache_path = os.path.join(os.getcwd(), 'librispeech_cache')

if os.path.exists(download_path):
    shutil.rmtree(download_path)
if os.path.exists(cache_path):
    shutil.rmtree(cache_path)
os.makedirs(download_path)
os.makedirs(cache_path)
os.makedirs(os.path.join(download_path, 'clips'))

datasets.config.DOWNLOADED_DATASETS_PATH = Path(str(download_path))
datasets.config.HF_DATASETS_CACHE = Path(str(download_path))

dataset = load_dataset("librispeech_asr", 'clean', split="train.100", streaming=True)
dataset = iter(dataset)

csv_name = os.path.join(download_path, 'clean_train_100.csv')

rows = []
for i, row in enumerate(dataset):
    print(i)
    path = os.path.join('librispeech', 'clips', row['audio']['path'])
    soundfile.write(path, row['audio']['array'], row['audio']['sampling_rate'])

    del row['audio']
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(csv_name, index=False, header=True)
