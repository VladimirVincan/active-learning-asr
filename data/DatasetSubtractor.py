import os
import pickle
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
        default='final/metadata.csv',
        metadata={'help': 'Csv file of the large, original dataset.'}
    )
    csv2: str = field(
        default='cluster_sampler_inverse/metadata.csv',
        metadata={'help': 'Csv file of the subset.'}
    )
    folder: str = field(
        default='cluster_subtractor',
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

class DatasetSubtractor():
    """
    Inputs: train & whole dataset
    Output: new train dataset without original train
    """
    def __init__(self, data_args):
        self._csv1 = data_args.csv1
        self._csv2 = data_args.csv2
        self._folder = data_args.folder
        self._label_column = data_args.label_column
        self._path_column = data_args.path_column
        self._speaker_column = data_args.speaker_column

        self._df1 = pd.read_csv(self._csv1)
        self._df2 = pd.read_csv(self._csv2)
        print(self._df2)
        self._df2.drop(['cluster', 'file_name'], axis=1, inplace=True)
        self._df1_train, self._df1_others = self._divide_df_train_others(self._df1)
        self._df1_train = self._remove_subset_from_original(self._df1_train, self._df2)
        self._df_final = pd.concat([self._df1_train, self._df1_others]).reset_index(drop=True)
        self._df_final.drop(['split'], axis=1, inplace=True)
        self._df_final['file_name'] = self._df_final[self._path_column]
        self._symlink_csv(self._df_final, os.path.dirname(self._csv1), self._folder)

    def _divide_df_train_others(self, df, split='train'):
        """
        Select the train subgroup from df.

        We have train and dev and potentially test subgroups in the df.
        Only the train subgroup has clusters defined.
        """
        print('--- appending cluster numbers to df ---')
        if split == 'none':
            return df
        # https://stackoverflow.com/questions/37333299/splitting-a-pandas-dataframe-column-by-delimiter/52269469#52269469
        df['split'] = df['path'].str.split('/').str[1]
        condition = df['split'] == split
        df_train = pd.DataFrame(columns=df.columns)
        df_train = df_train.append(df[condition], ignore_index=True)

        condition = condition.apply(lambda x: not x)
        df_others = pd.DataFrame(columns=df.columns)
        df_others = df_others.append(df[condition], ignore_index=True)

        return df_train, df_others

    def _remove_subset_from_original(self, df_original, df_subset):
        df = df_original.merge(df_subset, how='left', indicator = True)
        df.reset_index(drop=True, inplace=True)
        df = df[df['_merge'] == 'left_only']
        print(df_original)
        print(df)
        df.drop(['_merge'], axis=1, inplace=True)
        return df

    def _create_relative_symlink(self, src, dst):
        directory = os.path.dirname(dst)
        src = os.path.realpath(src)
        shutil.copy(src, dst)
        # os.symlink(src, dst)

    def _symlink_csv(self, df, input_folder, output_folder):
        """
        Take a single csv file (dataframe). Copy the audio contents to destination. Create metadata.
        """
        for i, row in df.iterrows():
            src = os.path.join(input_folder, row[self._path_column])
            dst = os.path.join(output_folder, row[self._path_column])
            dirname = os.path.dirname(dst)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            self._create_relative_symlink(src, dst)
        df.to_csv(os.path.join(output_folder, 'metadata.csv'), index=False)


def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args = parser.parse_args_into_dataclasses()
    data_args = data_args[0]

    dataset_subtractor = DatasetSubtractor(data_args)


if __name__ == '__main__':
    main()
