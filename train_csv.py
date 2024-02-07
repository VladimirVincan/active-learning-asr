import os
import pickle
import sys
from dataclasses import dataclass, field

import pandas as pd
import torch
from kmeans_pytorch import kmeans
from transformers import HfArgumentParser


@dataclass
class DataArguments:
    csv_filename: str = field(
        default='common_voice/validated.tsv',
        metadata={'help': 'The csv file which contains info about audio files.'}
    )
    output_csv_filename: str = field(
        default=None,
        metadata={'help': 'The output csv file - merged input csv file with cluster info. If name not defined, use csv_filename + output'}
    )
    random_csv_filename: str = field(
        default=None,
        metadata={'help': 'Randomly sampled rows from input csv. If name not defined, use csv_filename + random'}
    )
    num_rows_in_random: int = field(
        default=10,
        metadata={'help': 'Number of rows randomly sampled from input csv'}
    )
    cluster_csv_filename: str = field(
        default=None,
        metadata={'help': 'Randomly sampled rows from each cluster from input csv. If name not defined, use csv_filename + cluster'}
    )
    num_rows_in_cluster: int = field(
        default=2,
        metadata={'help': 'Number of rows randomly sampled from each cluster in input csv'}
    )
    audio_folder: str = field(
        default='cv_samples/',
        metadata={'help': 'Path to audio files.'}
    )
    clusters_dump_name: str = field(
        default='clusters/cv16.pkl',
        metadata={'help': 'Name of pkl dump file of clusters list.'}
    )


def create_data_args():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args = parser.parse_args_into_dataclasses()

    # Remove this line if multiple dataclasses exist in parser
    data_args = data_args[0]

    if data_args.output_csv_filename is None:
        left, right = data_args.csv_filename.split('.')
        data_args.output_csv_filename = left + '_output.' + right

    if data_args.random_csv_filename is None:
        left, right = data_args.csv_filename.split('.')
        data_args.random_csv_filename = left + '_random.' + right

    if data_args.cluster_csv_filename is None:
        left, right = data_args.csv_filename.split('.')
        data_args.cluster_csv_filename = left + '_cluster.' + right

    return data_args


def create_random_df(original_df, n):
    """
    Create a new DataFrame with n random rows from the original DataFrame.

    Parameters:
    - original_df: pandas DataFrame
    - n: int, number of random rows to select

    Returns:
    - new_df: pandas DataFrame with n random rows
    """
    # Check if n is greater than the number of rows in the original DataFrame
    if n > len(original_df):
        raise ValueError("Number of random rows (n) should be less than or equal to the number of rows in the original DataFrame.")

    # Use the sample method to select n random rows
    new_df = original_df.sample(n)

    return new_df


def create_cluster_df(original_df, n):
    """
    Create a new DataFrame by sampling n rows from each cluster in the original DataFrame.

    Parameters:
    - original_df: pandas DataFrame
    - n: int, number of rows to sample from each cluster

    Returns:
    - new_df: pandas DataFrame with n random rows from each cluster
    """
    # Check if n is greater than the maximum number of rows in any cluster
    if n > original_df.groupby('cluster').size().min():
        import warnings
        warnings.warn("Number of random rows (n) should be less than or equal to the size of the smallest cluster.")

    # Use groupby and apply to sample n rows from each cluster
    new_df = original_df.groupby('cluster', group_keys=False).apply(lambda group: group.sample(min(n, group.shape[0])))

    return new_df


def main():
    data_args = create_data_args()

    with open(data_args.clusters_dump_name, 'rb') as f:
        cluster_dicts = pickle.load(f)

    df = pd.read_csv(data_args.csv_filename, delimiter='\t')

    # Remove .mp3 extension from cluster dicts
    df_cluster = pd.DataFrame(cluster_dicts)
    df_cluster['path'] = df_cluster['filename'].str.split('.').str[0]
    df_cluster.drop(columns=['filename'], inplace=True)

    # change 'cluster' column from tensors to integers
    def tensor_to_int(tensor):
        return int(tensor.item())
    df_cluster['cluster'] = df_cluster['cluster'].apply(tensor_to_int)

    # df_intersection = pd.merge(df_intersection, df_cluster, on='path', how='left')
    df_intersection = pd.merge(df, df_cluster, on='path', how='inner')
    df_intersection.to_csv(data_args.output_csv_filename, sep='\t')
    print('Intersection df shape: ' + str(df_intersection.shape))

    df_random = create_random_df(df_intersection, data_args.num_rows_in_random)
    df_random.to_csv(data_args.random_csv_filename, sep='\t', index=False)
    print('Random df shape: ' + str(df_random.shape))

    df_cluster = create_cluster_df(df_intersection, data_args.num_rows_in_cluster)
    df_cluster.to_csv(data_args.cluster_csv_filename, sep='\t', index=False)
    print('Cluster df shape: ' + str(df_cluster.shape))


if __name__ == '__main__':
    main()
