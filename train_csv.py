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
        default='common_voice/validated_cluster.tsv',
        metadata={'help': 'The output csv file - merged input csv file with cluster info.'}
    )
    audio_folder: str = field(
        default='cv_samples/',
        metadata={'help': 'Path to audio files.'}
    )
    clusters_dump_name: str = field(
        default='clusters/cv16.pkl',
        metadata={'help': 'Name of pkl dump file of clusters list.'}
    )


def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args = parser.parse_args_into_dataclasses()

    # Remove this line if multiple dataclasses exist in parser
    data_args = data_args[0]

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
    print(df_intersection.head())
    print(df_intersection.shape)

    df_intersection.to_csv(data_args.output_csv_filename, sep='\t')

if __name__ == '__main__':
    main()
