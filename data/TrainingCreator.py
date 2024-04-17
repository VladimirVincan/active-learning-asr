import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
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
        metadata={'help':
                  'Options: train/dev/test/same/split. \
                  Same means keep the same distribution if the data is already split.'}
    )
    split2: str = field(
        default='train',
        metadata={'help':
                  'Options: train/dev/test/same/none. \
                  Same means keep the same distribution if the data is already split. \
                  None means use only csv1 to create audio subfolder.'}
    )
    split_size: float = field(
        default= 0.3,
        metadata={'help': 'Percentage that goes into test.'}
    )
    speaker_id_1: str = field(
        default='',
        metadata={'help': 'Single speaker id to incude from csv1.'}
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
    symlink: bool = field(
        default=True,
        metadata={'help': 'os.path.symlink (true) or shutil.copy (false).'}
    )
    file_path_relative: bool = field(
        default=False,
        metadata={'help':
                  'Originally, it was expected for data to be clips/*.wav. \
                  only the file name was written in path_column. \
                  if it is not the case, i.e. the relative file path is written, \
                  then do not add clips/ to beginning'}
    )


class TrainingCreator:
    def __init__(self, data_args):
        self._csv1 = data_args.csv1
        self._csv2 = data_args.csv2
        self._split1 = data_args.split1
        self._split2 = data_args.split2
        self._split_size = data_args.split_size
        self._speaker_id_1 = data_args.speaker_id_1
        self._folder = data_args.folder
        self._label_column = data_args.label_column
        self._path_column = data_args.path_column
        self._speaker_column = data_args.speaker_column
        self._symlink = data_args.symlink
        self._file_path_relative = data_args.file_path_relative

        self._df1 = pd.read_csv(self._csv1)
        if self._speaker_id_1 != '':
            self._df1[self._speaker_column] = self._df1[self._speaker_column].astype(str)
            self._df1 = self._df1[self._df1[self._speaker_column] == self._speaker_id_1]
        if self._split2 != 'none':
            self._df2 = pd.read_csv(self._csv2)

    def _create_relative_symlink(self, src, dst):
        dir = os.path.dirname(dst)
        dir = os.path.realpath(dir)
        src = os.path.realpath(src)
        # src = os.path.relpath(src, dir)
        if self._symlink:
            return os.symlink(src, dst)
        else:
            return shutil.copy(src, dst)

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
                dir = os.path.join(self._folder, 'clips', split)
                split_dir = os.path.join(dirname, 'clips', split)
                if not os.path.exists(dir) and os.path.isdir(split_dir):
                    os.mkdir(dir)

        elif self._split1 == 'split':
            os.makedirs(os.path.join(self._folder + '_split_1', 'clips'))
            os.makedirs(os.path.join(self._folder + '_split_2', 'clips'))

        if self._split2 == 'same':
            dirname = os.path.dirname(self._csv2)
            splits = os.listdir(os.path.join(dirname, 'clips'))
            for split in splits:
                dir = os.path.join(self._folder, 'clips', split)
                split_dir = os.path.join(dirname, 'clips', split)
                if not os.path.exists(dir) and os.path.isdir(split_dir):
                    os.mkdir(dir)

    def _symlink_csv(self, csv_dir, df, split):
        """
        Take a single csv file (dataframe). Copy the audio contents to destination. Create metadata.
        """
        for i, row in df.iterrows():
            if self._file_path_relative:
                src = os.path.join(csv_dir, row[self._path_column])
            else:
                src = os.path.join(csv_dir, 'clips', row[self._path_column])

            if split == 'train' or split == 'dev' or split == 'test':
                dst = os.path.join(self._folder, 'clips', split, row[self._path_column])
            elif split == 'same':
                dst = os.path.join(self._folder, row[self._path_column])
                """
                TODO: remove after DatasetDownloader changes file names to relative file paths + file names
                """
                if row[self._path_column] == os.path.basename(row[self._path_column]):
                    dst = os.path.join(self._folder, 'clips', row[self._path_column])
            elif split.find('split') >= 0:
                dst = os.path.join(self._folder + split, row[self._path_column])
                """
                TODO: remove after DatasetDownloader changes file names to relative file paths + file names
                """
                if row[self._path_column] == os.path.basename(row[self._path_column]):
                    dst = os.path.join(self._folder + split, 'clips', row[self._path_column])
            else:
                raise ValueError('Split name incorrect!')

            self._create_relative_symlink(src, dst)

        if split == 'train' or split == 'dev' or split == 'test':
            df[self._path_column] = df.apply(lambda row: os.path.join('clips', split, row[self._path_column]), axis=1)

        return df

    def _create_metadata(self):
        if self._split1 == 'split':
            self._df1.to_csv(os.path.join(self._folder + '_split_1', 'metadata.csv'), index=False)
            self._df2.to_csv(os.path.join(self._folder + '_split_2', 'metadata.csv'), index=False)
            return
        if self._split2 != 'none':
            concat_df = pd.concat([self._df1, self._df2], ignore_index=True)
        else:
            concat_df = self._df1
        concat_df = self._remove_bad_transcriptions(concat_df)
        concat_df.to_csv(os.path.join(self._folder, 'metadata.csv'), index=False)

    def _remove_bad_transcriptions(self, df):
        """
        Delete [, ], (, ), ` characters.
        Change:
        - -> ' '
        & -> and
        á -> a,
        â -> ',
        ë -> e,
        é -> e,
        ñ -> n,
        ú -> u,
        ō -> o,
        ó -> o

        Examples (all from Common Voice):
        507      Variations (on a musical air) With great rapidity
        423      Play pop-rap off Google Music.
        1379     [What] a piece of work [is man]
        1477     A man and a woman on a motorcycle.`
        9396     table for five at Space Aliens Grill & Bar in FM

        3414     "Please put maimi yajima's song onto Operación Bikini."
        5766        add the best of guitar shorty in my playlist clásica
        6191                    Hunger is good mustard â the best sauce.
        6900                                      I canât understand it.
        7179                                            Play a Nóta song
        8581                Thatâs what you get for testing my patience.
        10721              Today Iâm making the Internet more inclusive.
        """
        df[self._label_column] = df[self._label_column].str.replace('`', ' ')
        df[self._label_column] = df[self._label_column].str.replace('[', ' ')
        df[self._label_column] = df[self._label_column].str.replace(']', ' ')
        df[self._label_column] = df[self._label_column].str.replace('(', ' ')
        df[self._label_column] = df[self._label_column].str.replace(')', ' ')
        df[self._label_column] = df[self._label_column].str.replace('-', ' ')

        df[self._label_column] = df[self._label_column].str.replace('â', '\'')
        df[self._label_column] = df[self._label_column].str.replace('á', 'a')
        df[self._label_column] = df[self._label_column].str.replace('ë', 'e')
        df[self._label_column] = df[self._label_column].str.replace('é', 'e')
        df[self._label_column] = df[self._label_column].str.replace('ñ', 'n')
        df[self._label_column] = df[self._label_column].str.replace('ú', 'u')
        df[self._label_column] = df[self._label_column].str.replace('ō', 'o')
        df[self._label_column] = df[self._label_column].str.replace('ó', 'o')
        df[self._label_column] = df[self._label_column].str.replace('&', ' and ')

        return df

    def create_training_data(self):
        self._create_filepath()
        if self._split1 != 'split':
            self._df1 = self._symlink_csv(os.path.dirname(self._csv1), self._df1, self._split1)
        else:
            train, test = train_test_split(self._df1, test_size=self._split_size, random_state=42)
            self._df1 = self._symlink_csv(os.path.dirname(self._csv1), train, '_split_1')
            self._df2 = self._symlink_csv(os.path.dirname(self._csv1), test, '_split_2')
        if self._split2 != 'none':
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
