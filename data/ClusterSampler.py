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
    clusters_dump_name: str = field(
        default='clusters/cv16.pkl',
        metadata={'help': 'Name of pkl dump file of clusters list.'}
    )
    split: str = field(
        default='train',
        metadata={'help': 'Options: train/dev/test/none. \
        Which split do the samples belong to. If none, then sample the whole df.'}
    )
    csv: str = field(
        default='common_voice/metadata.csv',
        metadata={'help': 'Csv file of data that is going to be merged.'}
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


class ClusterSampler():
    def __init__(self, data_args):
        self._clusters_dump_name = data_args.clusters_dump_name
        self._split = data_args.split
        self._csv = data_args.csv
        self._folder = data_args.folder
        self._label_column = data_args.label_column
        self._path_column = data_args.path_column
        self._speaker_column = data_args.speaker_column

        print('--- read csv ---')
        self._df = pd.read_csv(self._csv)
        print('--- opening clusters dicts ---')
        with open(self._clusters_dump_name, 'rb') as f:
            self._clusters_dicts = pickle.load(f)

        if os.path.exists(self._folder):
            shutil.rmtree(self._folder)

        self._df_train, self._df_others = self._divide_df_train_others()
        self._df_train = self.assign_cluster_to_df(self._df_train, self._clusters_dicts)
        # self._df_train.to_csv('clusters1.csv')
        self._df_train = self._get_sampled_df(self._df_train)

        self._df_output = pd.concat([self._df_train, self._df_others]).reset_index(drop=True)
        self._df_output = self._remove_bad_transcriptions(self._df_output)
        # self._df_output[self._path_column] = self._df_output.apply(lambda row: row[self._path_column].replace('clips', 'data'), axis=1)
        self._df_output['file_name'] = self._df_output[self._path_column]
        self._symlink_csv(self._df_output, os.path.dirname(self._csv), self._folder)

    def _divide_df_train_others(self):
        """
        Select the train subgroup from df.

        We have train and dev and potentially test subgroups in the df.
        Only the train subgroup has clusters defined.
        """
        print('--- appending cluster numbers to df ---')
        if self._split == 'none':
            return self._df
        # https://stackoverflow.com/questions/37333299/splitting-a-pandas-dataframe-column-by-delimiter/52269469#52269469
        self._df['split'] = self._df['path'].str.split('/').str[1]
        condition = self._df['split'] == self._split
        df_train = pd.DataFrame(columns=self._df.columns)
        df_train = df_train.append(self._df[condition], ignore_index=True)

        condition = condition.apply(lambda x: not x)
        df_others = pd.DataFrame(columns=self._df.columns)
        df_others = df_others.append(self._df[condition], ignore_index=True)

        return df_train, df_others

    def get_clusters_dicts(self):
        return self._clusters_dicts

    def get_clusters_list(self):
        clusters_list = []
        for i, cluster_dict in enumerate(self._clusters_dicts):
            clusters_list.append(cluster_dict['cluster'])
        return clusters_list

    def analyze_clusters_list(self):
        from collections import Counter
        clusters_list = self.get_clusters_list()
        element_count = Counter(clusters_list)
        for element, count in element_count.items():
            print(f'{element}: {count}')

    def get_number_of_elements(self):
        return len(self.get_clusters_list())

    def _convert_clusters_dicts_to_single_dict(self, clusters_dicts):
        """
        [{"key1": "value1", "key2": "value2"}, {"key1": "value3", "key2": "value4"}, {"key3": "value5", "key4": "value6"}]
        ->
        {'value1': 'value2', 'value3': 'value4', 'value5': 'value6'}
        """
        merged_dict = {}
        for d in clusters_dicts:
            merged_dict[d['filename']] = d['cluster']
        return merged_dict

    def _remove_bad_transcriptions(self, df):
        """
        Remove ( and ). Change á, ë to a, e
        """
        df = df[~df[self._label_column].str.contains('\(')]
        df = df[~df[self._label_column].str.contains('\)')]
        df[self._label_column] = df[self._label_column].str.replace('á', 'a')
        df[self._label_column] = df[self._label_column].str.replace('ë', 'e')
        return df

    def assign_cluster_to_df(self, df, clusters_dicts):
        """
        Create cluster column in df, add number of cluster.
        """
        clusters_dict = self._convert_clusters_dicts_to_single_dict(clusters_dicts)
        df['cluster'] = ''
        path = df.iloc[[0]]['path'][0]
        path=str(path)
        for i, row in df.iterrows():
            path = df.loc[i, 'path']
            basename = path.split('/')[-1]
            cluster_num = clusters_dict[basename]
            df.loc[i, 'cluster'] = cluster_num
        print('shape:')
        empty_string_count = (df['cluster'] == '').sum()
        print('Empty string count = ' + str(empty_string_count))
        return df

    def _get_sampled_df(self, df, frac=0.05):
        """
        Get df where train subset is sampled.

        https://www.geeksforgeeks.org/stratified-sampling-in-pandas/
        """
        df_stratified = df.groupby('cluster', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))
        return df_stratified

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

    cluster_sampler = ClusterSampler(data_args)



if __name__ == '__main__':
    main()
