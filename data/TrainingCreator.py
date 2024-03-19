import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import pandas as pd
from transformers import HfArgumentParser


@dataclass
class DataArguments:
    csv1: str = field(
        default='common_voice/metadata.csv',
        metadata={'help': 'Csv file of data that is going to be merged.'}
    )
    csv2: str = field(
        default='librispeech/metadata.csv',
        metadata={'help': 'Csv file of data that is going to be merged.'}
    )
    split1: str = field(
        default='train',
        metadata={'help': 'Options: train/dev/test/same. Same means keep the same distribution if the data is already split.'}
    )
    split2: str = field(
        default='train',
        metadata={'help': 'Options: train/dev/test/same. Same means keep the same distribution if the data is already split.'}
    )
    folder: str = field(
        default='merge',
        metadata={'help': 'Name of folder where data will be stored.'}
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


class TrainingCreator:
    def __init__(self, data_args):
        self._csv1 = data_args.csv1
        self._csv2 = data_args.csv2
        self._split1 = data_args.split1
        self._split2 = data_args.split2
        self._folder = data_args.folder
        self._label_column = data_args.label_column
        self._path_column = data_args.path_column
        self._speaker_column = data_args.speaker_column

        self._df1 = pd.read_csv(self._csv1)
        self._df2 = pd.read_csv(self._csv2)

    def _create_relative_symlink(self, src, dst):
        dir = os.path.dirname(dst)
        src = os.path.relpath(src, dir)
        return os.symlink(src, dst)

    def _create_filepath(self):
        if os.path.exists(self._folder):
            shutil.rmtree(self._folder)
        os.makedirs(os.path.join(self._folder, 'clips'))
        if self._split1 == 'train' or self._split2 == 'train':
            os.mkdir(os.path.join(self._folder, 'clips', 'train'))
        if self._split1 == 'dev' or self._split2 == 'dev':
            os.mkdir(os.path.join(self._folder, 'clips', 'dev'))
        if self._split1 == 'test' or self._split2 == 'test':
            os.mkdir(os.path.join(self._folder, 'clips', 'test'))

        if self._split1 == 'same':
            dirname = os.path.dirname(self._csv1)
            splits = os.listdir(os.path.join(dirname, 'clips'))
            for split in splits:
                if not os.path.exists(os.path.join(self._folder, 'clips', split)):
                    os.mkdir(os.path.join(self._folder, 'clips', split))

        if self._split2 == 'same':
            dirname = os.path.dirname(self._csv2)
            splits = os.listdir(os.path.join(dirname, 'clips'))
            for split in splits:
                if not os.path.exists(os.path.join(self._folder, 'clips', split)):
                    os.mkdir(os.path.join(self._folder, 'clips', split))

    def _symlink_csv(self, csv_dir, df, split):
        """
        Take a single csv file (dataframe). Copy the audio contents to destination. Create metadata.
        """
        for i, row in df.iterrows():
            src = os.path.join(csv_dir, 'clips', row[self._path_column])
            if split == 'train' or split == 'dev' or split == 'test':
                dst = os.path.join(self._folder, 'clips', split, row[self._path_column])
            elif split == 'same':
                dst = os.path.join(self._folder, row[self._path_column])
            else:
                raise ValueError('Split name incorrect!')

            self._create_relative_symlink(src, dst)

        if split == 'train' or split == 'dev' or split == 'test':
            df[self._path_column] = df.apply(lambda row: os.path.join('clips', split, row[self._path_column]), axis=1)

        print(df.head()[self._path_column])

        return df

    def _create_metadata(self):
        concat_df = pd.concat([self._df1, self._df2], ignore_index=True)
        concat_df.to_csv(os.path.join(self._folder, 'metadata.csv'), index=False)

    def create_training_data(self):
        self._create_filepath()
        self._df1 = self._symlink_csv(os.path.dirname(self._csv1), self._df1, self._split1)
        self._df2 = self._symlink_csv(os.path.dirname(self._csv2), self._df2, self._split2)
        self._create_metadata()


def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args = parser.parse_args_into_dataclasses()
    data_args = data_args[0]

    training_creator = TrainingCreator(data_args)
    training_creator.create_training_data()
    # training_creator.relative_symlink("minimal/Text File.txt", "symlink/symlink.txt")


if __name__ == '__main__':
    main()
