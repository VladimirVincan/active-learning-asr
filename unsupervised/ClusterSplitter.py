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
    csv: str = field(
        default='common_voice/validated.tsv',
        metadata={'help': 'The csv file which contains info about audio files.'}
    )
    folder: str = field(
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


class ClusterSplitter():
    def __init__(self, data_args):
        self._csv = data_args.csv
