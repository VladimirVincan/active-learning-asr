import os
import sys
import shutil
from pathlib import Path
from dataclasses import dataclass, field

import datasets
import pandas as pd
import soundfile
from datasets import load_dataset, Audio
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
    extension: str = field(
        default='.wav',
        metadata={'help': 'Extension of audio files.'}
    )
    label_column: str = field(
        default='sentence',
        metadata={'help': 'Name of column name that has text labels of corresponding audio files.'}
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
        self.__use_auth_token = data_args.use_auth_token
        self.__dataset = data_args.dataset
        self.__lang = data_args.lang
        self.__split = data_args.split
        self.__folder = os.path.join(os.getcwd(), data_args.folder)
        self.__cache = os.path.join(os.getcwd(), data_args.folder + '_cache')
        self.__sampling_rate = data_args.sampling_rate
        self.__extension = data_args.extension
        self.__label_column = data_args.label_column
        self.__csv = os.path.join(self.__cache, data_args.csv)
        self.__num_samples = data_args.num_samples

        datasets.config.DOWNLOADED_DATASETS_PATH = Path(str(self.__cache))
        datasets.config.HF_DATASETS_CACHE = Path(str(self.__cache))

    def __create_folders(self, folder_path):
        os.makedirs(folder_path)

    def __remove_folders(self, folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

    def __move_file(self, src_path, dest_path):
        shutil.move(src_path, dest_path)

    def __move_folder(self, src_path, dest_path):
        file_names = os.listdir(src_path)
        for file_name in file_names:
            self.__move_file(os.path.join(src_path, file_name), dest_path)

    def __process_librispeech(self, df):
        df = df.rename(columns={"text": self.__label_column})
        df['file'] = df['file'].apply(lambda x: x.split('.')[0] + self.__extension)
        return df

    def download(self):
        self.__remove_folders(self.__folder)
        self.__remove_folders(self.__cache)
        self.__create_folders(os.path.join(self.__folder, 'clips'))
        self.__create_folders(os.path.join(self.__cache, 'clips'))

        dataset = load_dataset(self.__dataset, self.__lang, split=self.__split, streaming=True, use_auth_token=self.__use_auth_token, cache_dir=self.__cache)
        dataset = dataset.cast_column('audio', Audio(sampling_rate=self.__sampling_rate))  # change sampling rate
        dataset = iter(dataset)

        rows = []
        for i, row in enumerate(dataset):
            print(i)
            path = os.path.join(self.__cache, 'clips', row['audio']['path'])
            path = path.split('.')[0] + '.wav'
            soundfile.write(path, row['audio']['array'], row['audio']['sampling_rate'])

            del row['audio']
            rows.append(row)

            if i == self.__num_samples:
                break

        df = pd.DataFrame(rows)
        if self.__dataset == 'librispeech_asr':
            df = self.__process_librispeech(df)
        df.to_csv(self.__csv, index=False, header=True)

        self.__move_file(self.__csv, self.__folder)
        self.__move_folder(os.path.join(self.__cache, 'clips'), os.path.join(self.__folder, 'clips'))
        self.__remove_folders(self.__cache)


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
