import csv
import os
import shutil
import sys
from dataclasses import dataclass, field

import pandas as pd
from transformers import HfArgumentParser


@dataclass
class DataArguments:
    folder: str = field(
        default='speaker_data',
        metadata={'help': 'Name of folder where data will be stored.'}
    )
    csv: str = field(
        default='../data/final/metadata.csv',
    )
    label_column: str = field(
        default='sentence',
        metadata={'help': 'Name of column name that has text labels of corresponding audio files.'}
    )
    path_column: str = field(
        default='file_name',
        metadata={'help': 'Name of column name that has text labels of corresponding audio files.'}
    )
    speaker_column: str = field(
        default='speaker_id',
        metadata={'help': 'Name of column name that has names/ids of speakers.'}
    )
    symlink: bool = field(
        default=True,
        metadata={'help': 'os.path.symlink (true) or shutil.copy (false).'}
    )
    speaker1: str = field(
        default="4195"
    )
    speaker2: str = field(
        default="1116"
    )


class SpeakerSelector():
    def __init__(self, data_args):
        self._csv = data_args.csv
        self._folder = data_args.folder
        self._label_column = data_args.label_column
        self._path_column = data_args.path_column
        self._speaker_column = data_args.speaker_column
        self._symlink = data_args.symlink
        self._speaker1 = data_args.speaker1
        self._speaker2 = data_args.speaker2

        self._df = pd.read_csv(self._csv)

    def generate_dataset(self):
        df_speaker1 = self._df[self._df[self._speaker_column] == self._speaker1]
        df_speaker2 = self._df[self._df[self._speaker_column] == self._speaker2]
        df = pd.concat([df_speaker1, df_speaker2], axis=0, ignore_index=True)

        if os.path.exists(self._folder):
            shutil.rmtree(self._folder)
        os.mkdir(self._folder)
        self._symlink_csv(df, os.path.dirname(self._csv), self._folder)

    def _symlink_csv(self, df, input_folder, output_folder):
        """
        Take a single csv file (dataframe). Copy the audio contents to destination. Create metadata.
        """
        df.to_csv(os.path.join(output_folder, 'metadata.csv'), index=False)
        value_counts = df[self._speaker_column].value_counts().reset_index()
        value_counts.to_csv(os.path.join(self._folder, 'value_counts.csv'), index=False)
        for i, row in df.iterrows():
            src = os.path.join(input_folder, row[self._path_column])
            # dst = os.path.join(output_folder, row[self._path_column])
            basename = os.path.basename(row[self._path_column])
            if row[self._speaker_column] == self._speaker1:
                dst = os.path.join(output_folder, 'speaker1', basename)
            else:
                dst = os.path.join(output_folder, 'speaker2', basename)
            dirname = os.path.dirname(dst)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            self._create_symlink(src, dst)

    def _create_symlink(self, src, dst):
        src = os.path.realpath(src)
        if self._symlink:
            return os.symlink(src, dst)
        else:
            return shutil.copy(src, dst)


def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args = parser.parse_args_into_dataclasses()
    data_args = data_args[0]

    speaker_selector = SpeakerSelector(data_args)
    speaker_selector.generate_dataset()


if __name__ == '__main__':
    main()
