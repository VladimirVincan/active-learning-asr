import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import pandas as pd
import soundfile
from datasets import Audio, load_dataset
from transformers import HfArgumentParser

"""
For mp3 files (common voice), Ubuntu 22+ needed.
"""


@dataclass
class DataArguments:
    use_auth_token: bool = field(
        default=False,
        metadata={'help': 'Huggingface token. Found here: https://huggingface.co/settings/tokens. Login via `huggingface-cli login`. Accept aggreement at https://huggingface.co/datasets/mozilla-foundation/common_voice_1_0'}
    )
    dataset: str = field(
        default='mozilla-foundation/common_voice_1_0',
        metadata={'help': 'Dataset name found in https://huggingface.co/datasets.'}
    )
    lang: str = field(
        default='en',
        metadata={'help': 'Language or main split (e.g. clean/other in Librispeech).'}
    )
    split: str = field(
        default='train',
        metadata={'help': 'train+test'}
    )
    folder: str = field(
        default='common_voice',
        metadata={'help': 'Name of folder where data will be stored.'}
    )
    sampling_rate: int = field(
        default=16000,
        metadata={'help': 'Audio file sampling rate.'}
    )
    mono: bool = field(
        default=True,
        metadata={'help': 'Set audio to be mono of stereo.'}
    )
    extension: str = field(
        default='.wav',
        metadata={'help': 'Extension of audio files.'}
    )
    label_column: str = field(
        default='sentence',
        metadata={'help': 'Name of column name that has text labels of corresponding audio files.'}
    )
    path_column: str = field(
        default='path',
        metadata={'help': 'Name of column name that has text labels of corresponding audio files.'}
    )
    speaker_column: str = field(
        default='speaker_id',
        metadata={'help': 'Name of column name that has names/ids of speakers.'}
    )
    csv: str = field(
        default='metadata.csv',
        metadata={'help': 'Name of csv file.'}
    )
    num_samples: int = field(
        default=-1,
        metadata={'help': 'Number of samples to download. For debug purposes only.'}
    )


class DatasetDownloader:
    def __init__(self, data_args):
        self._use_auth_token = data_args.use_auth_token
        self._dataset = data_args.dataset
        self._lang = data_args.lang
        self._split = data_args.split
        self._folder = os.path.join(os.getcwd(), data_args.folder)
        self._cache = os.path.join(os.getcwd(), data_args.folder + '_cache')
        self._sampling_rate = data_args.sampling_rate
        self._mono = data_args.mono
        self._extension = data_args.extension
        self._label_column = data_args.label_column
        self._path_column = data_args.path_column
        self._speaker_column = data_args.speaker_column
        self._csv = data_args.csv
        self._num_samples = data_args.num_samples

        datasets.config.DOWNLOADED_DATASETS_PATH = Path(str(self._cache))
        datasets.config.HF_DATASETS_CACHE = Path(str(self._cache))

    def _create_folders(self, folder_path):
        os.makedirs(folder_path)

    def _remove_folders(self, folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

    def _move_file(self, src_path, dest_path):
        shutil.move(src_path, dest_path)

    def _move_folder(self, src_path, dest_path):
        file_names = os.listdir(src_path)
        for file_name in file_names:
            self._move_file(os.path.join(src_path, file_name), dest_path)

    def _process_librispeech(self, df):
        df = df.rename(columns={"text": self._label_column, "file": self._path_column, "speaker_id": self._speaker_column})
        df[self._path_column] = df[self._path_column].apply(lambda x: x.split('.')[0] + self._extension)
        return df

    def _preprocess_common_voice(self, row):
        row['audio']['path'] = os.path.basename(row['audio']['path'])
        row['path'] = os.path.basename(row['audio']['path'])
        return row

    def _process_common_voice(self, df):
        df = df.rename(columns={"sentence": self._label_column, "path": self._path_column, "client_id": self._speaker_column})
        df[self._path_column] = df[self._path_column].apply(lambda x: x.split('.')[0] + self._extension)
        return df

    def download(self):
        self._remove_folders(self._folder)
        self._remove_folders(self._cache)
        self._create_folders(os.path.join(self._folder, 'clips'))
        self._create_folders(os.path.join(self._cache, 'clips'))

        dataset = load_dataset(self._dataset, self._lang, split=self._split, streaming=True, use_auth_token=self._use_auth_token, cache_dir=self._cache)
        dataset = dataset.cast_column('audio', Audio(sampling_rate=self._sampling_rate, mono=self._mono))  # change sampling rate
        dataset = iter(dataset)

        rows = []
        for i, row in enumerate(dataset):
            print(i)
            if self._dataset.find('common_voice') >= 0:
                row = self._preprocess_common_voice(row)
            path = os.path.join(self._cache, 'clips', row['audio']['path'])
            path = path.split('.')[0] + '.wav'
            soundfile.write(path, row['audio']['array'], row['audio']['sampling_rate'])

            del row['audio']
            rows.append(row)

            if i == self._num_samples:
                break

        df = pd.DataFrame(rows)
        if self._dataset.find('librispeech') >= 0:
            df = self._process_librispeech(df)
        elif self._dataset.find('common_voice') >= 0:
            df = self._process_common_voice(df)
        df.to_csv(os.path.join(self._cache, self._csv), index=False, header=True)

        self._move_file(os.path.join(self._cache, self._csv), self._folder)
        self._move_folder(os.path.join(self._cache, 'clips'), os.path.join(self._folder, 'clips'))
        self._remove_folders(self._cache)


def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args = parser.parse_args_into_dataclasses()
    data_args = data_args[0]

    dataset_downloader = DatasetDownloader(data_args)
    dataset_downloader.download()


if __name__ == '__main__':
    main()
