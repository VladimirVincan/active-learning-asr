import os
import pickle
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import pandas as pd
from sklearn.metrics.pairwise import _num_samples
from sklearn.model_selection import train_test_split
from transformers import HfArgumentParser


@dataclass
class DataArguments:
    csv: str = field(
        default='final_inverse_2/metadata.csv',
        metadata={'help': 'Csv file of the dataset for which uncertainties have been calculated.'}
    )
    uncertainties: str = field(
        default='../batch_active_learning/uncertainties.csv',
        metadata={'help': 'Csv file of the dataset with file paths and uncertainties.'}
    )
    clusters_dump_name: str = field(
        default='../clusters/train_final.pkl',
        metadata={'help': 'Name of pkl dump file of clusters list.'}
    )
    value_counts: str = field(
        default='cluster_sampler_inverse/value_counts.csv',
        metadata={'help': 'how many files should be sampled. do not change'}
    )
    folder: str = field(
        default='cluster_sampler',
        metadata={'help': 'Name of folder where data will be stored.'}
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
    algorithm: str = field(
        default='inverse',
        metadata={'help': 'inverse, smca, random'}
    )
    symlink: bool = field(
        default=True,
        metadata={'help': 'os.path.symlink (true) or shutil.copy (false).'}
    )


class UncertaintySelector():
    """
    Inputs: whole dataset csv & csv with uncertainty
    Process: merge csvs, select from each cluster the same amount of samples
    Output: csv for training

    Where did cluster dissapear??

    Both csvs will later go to DatasetSubtractor
    Number of samples per cluster (from ClusterSampler):
    cluster   #  sample_percentage  affine_linear  num_samples
    -1    11555           0.897964       0.045610          527
     1      288           0.022381       0.093749           26
     3      182           0.014144       0.094202           17
     6      100           0.007771       0.094553            9
     9       93           0.007227       0.094583            8
     5       86           0.006683       0.094613            8
     2       82           0.006372       0.094630            7
    11       81           0.006295       0.094634            7
     8       80           0.006217       0.094638            7
     7       77           0.005984       0.094651            7
     0       72           0.005595       0.094672            6
    10       69           0.005362       0.094685            6
    12       52           0.004041       0.094758            4
     4       51           0.003963       0.094762            4
    """
    def __init__(self, data_args):
        self._csv = data_args.csv
        self._uncertainties = data_args.uncertainties
        self._clusters_dump_name = data_args.clusters_dump_name
        self._value_counts = data_args.value_counts
        with open(self._clusters_dump_name, 'rb') as f:
            self._clusters_dicts = pickle.load(f)
        self._folder = data_args.folder
        self._label_column = data_args.label_column
        self._path_column = data_args.path_column
        self._speaker_column = data_args.speaker_column
        self._algorithm = data_args.algorithm
        self._symlink = data_args.symlink

        self._df = pd.read_csv(self._csv)
        if self._algorithm != 'random':
            self._df_uncertainties = pd.read_csv(self._uncertainties)
        self._df_value_counts = pd.read_csv(self._value_counts)
        with open(self._clusters_dump_name, 'rb') as f:
            self._clusters_dicts = pickle.load(f)

        print(self._df_value_counts['num_samples'].sum())
        print('-------------------------------')

    def generate_dataset(self):
        df_train, df_val = self._divide_df_train_others(self._df)
        if self._algorithm != 'random':
            df_train = pd.merge(df_train, self._df_uncertainties, how='left', on=self._path_column)
            self._assert_uncertainties_not_nan(df_train)
        df_train = self._assign_cluster_to_df(df_train, self._clusters_dicts)
        # self._assert_value_counts(df_train)

        df_sampled = self._sample(df_train, self._df_value_counts)
        # do not do concat anymore
        # we do not want df_val.
        # df_val will be added from train_inverse_1
        # self._df_out = pd.concat([df_sampled, df_val])
        self._df_out = df_sampled

        # self._find_matching_rows(df_sampled, df_train)

        if os.path.exists(self._folder):
            shutil.rmtree(self._folder)
        os.mkdir(self._folder)
        self._symlink_csv(self._df_out, os.path.dirname(self._csv), self._folder)
        # df_train.to_csv(os.path.join(self._folder, 'whole_uncertainties.csv'), index=False)

    def _sample(self, df, value_counts):
        """
        Sample df w.r.t. value counts and uncertainties.

        E.g: Sample 527 samples from cluster -1 with maximum uncertainties (11555 samples).
        Sample 26 samples from cluster 1 with maximum uncertainties (288 samples).
        Etc.
        """
        # df_sampled = df.groupby('cluster', group_keys=False).apply(lambda x: x.sample(n=value_counts.loc[value_counts['index']==x.iloc[0]['cluster'], 'num_samples'].values[0], random_state=42))
        if self._algorithm == 'inverse':
            df_sampled = df.groupby('cluster', group_keys=False).apply(lambda x: x.nlargest(value_counts.loc[value_counts['index']==x.iloc[0]['cluster'], 'num_samples'].values[0], 'uncertainty'))
        elif self._algorithm == 'smca':
            num_samples = value_counts['num_samples'].sum()
            print(num_samples)
            df_sampled = df.nlargest(num_samples, 'uncertainty')
        else:  # if self._algorithm == 'random':
            num_samples = value_counts['num_samples'].sum()
            df_sampled = df.sample(n=num_samples, random_state=42)
        return df_sampled

    def _assert_uncertainties_not_nan(self, df):
        assert df['uncertainty'].isna().sum() == 0, 'NaN values in uncertainty column!'

    def _assert_value_counts(self, df_train):
        """
        We have the value counts from cluster_sampler_inverse_1.
        That's how many samples to take from each cluster.
        That number, plus final_inverse_n,
        should be equal to final_inverse_n-1.
        """
        value_counts = df_train['cluster'].value_counts().reset_index()
        # print(value_counts)
        # print(self._df_value_counts)
        total_rows = value_counts['cluster'].sum()
        # print(total_rows)
        df_diff = self._df_value_counts['cluster'] - self._df_value_counts['num_samples'] - value_counts['cluster']
        print(df_diff)
        assert (df_diff==0).all(), 'Not all values in df_diff are zero!'

    def _find_matching_rows(self, df1, df2):
        """
        df1 and df2 should be disjoint.
        If matching rows found, print them
        Only values in path column should match.
        """
        df_matching = pd.merge(df1, df2, on=self._path_column, how='inner')
        if not df_matching.empty:
            print(df_matching)
            raise ValueError('DFs should be disjoint!')

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
        df['split'] = df[self._path_column].str.split('/').str[1]
        condition = df['split'] == split
        df_train = pd.DataFrame(columns=df.columns)
        df_train = df_train.append(df[condition], ignore_index=True)

        condition = condition.apply(lambda x: not x)
        df_others = pd.DataFrame(columns=df.columns)
        df_others = df_others.append(df[condition], ignore_index=True)

        return df_train, df_others

    def _assign_cluster_to_df(self, df, clusters_dicts):
        """
        Create cluster column in df, add number of cluster.
        """
        clusters_dict = self._convert_clusters_dicts_to_single_dict(clusters_dicts)
        df['cluster'] = ''
        path = df.iloc[[0]][self._path_column][0]
        path=str(path)
        for i, row in df.iterrows():
            path = df.loc[i, self._path_column]
            basename = path.split('/')[-1]
            cluster_num = clusters_dict[basename]
            df.loc[i, 'cluster'] = cluster_num
        print('shape:')
        empty_string_count = (df['cluster'] == '').sum()
        print('Empty string count = ' + str(empty_string_count))
        return df

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

    def _create_relative_symlink(self, src, dst):
        directory = os.path.dirname(dst)
        src = os.path.realpath(src)
        if self._symlink:
            return os.symlink(src, dst)
        else:
            return shutil.copy(src, dst)

    def _symlink_csv(self, df, input_folder, output_folder):
        """
        Take a single csv file (dataframe). Copy the audio contents to destination. Create metadata.
        """
        df.to_csv(os.path.join(output_folder, 'metadata.csv'), index=False)
        value_counts = df['cluster'].value_counts().reset_index()
        value_counts.to_csv(os.path.join(self._folder, 'value_counts.csv'), index=False)
        for i, row in df.iterrows():
            src = os.path.join(input_folder, row[self._path_column])
            dst = os.path.join(output_folder, row[self._path_column])
            dirname = os.path.dirname(dst)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            self._create_relative_symlink(src, dst)


def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args = parser.parse_args_into_dataclasses()
    data_args = data_args[0]

    uncertainty_selector = UncertaintySelector(data_args)
    uncertainty_selector.generate_dataset()


if __name__ == '__main__':
    main()
