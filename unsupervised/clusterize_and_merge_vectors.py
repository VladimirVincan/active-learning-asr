import os
import pickle
import sys
from dataclasses import dataclass, field

import torch
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
from transformers import HfArgumentParser


@dataclass
class DataArguments:
    embeddings1_dump_name: str = field(
        default='embeddings/clips.pkl',
        metadata={'help': 'Name of pkl dump file of embeddings.'}
    )
    embeddings2_dump_name: str = field(
        default='embeddings/librispeech.pkl',
        metadata={'help': 'Name of pkl dump file of embeddings.'}
    )
    embeddings_dump_name: str = field(
        default='embeddings/concat.pkl',
        metadata={'help': 'Name of pkl dump file of embeddings.'}
    )
    pickle_save: bool = field(
        default=False,
        metadata={'help': 'If embeddings should be saved as binary file'}
    )
    clusters_dump_name: str = field(
        default='clusters/concat.pkl',
        metadata={'help': 'Name of pkl dump file of clusters list.'}
    )


def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        data_args = parser.parse_args_into_dataclasses()

    # Remove this line if multiple dataclasses exist in parser
    data_args = data_args[0]

    with open(data_args.embeddings1_dump_name, 'rb') as f:
        embedding1_dicts = pickle.load(f)

    with open(data_args.embeddings2_dump_name, 'rb') as f:
        embedding2_dicts = pickle.load(f)

    # Create clusters list
    clusters = []
    for i, emb_dict in enumerate(embedding1_dicts):
        cluster_dict = {
            'filename': emb_dict['filename'],
            'cluster': 0
        }
        clusters.append(cluster_dict)

    for i, emb_dict in enumerate(embedding2_dicts):
        cluster_dict = {
            'filename': emb_dict['filename'],
            'cluster': 1
        }
        clusters.append(cluster_dict)

    embedding_concat_dicts = embedding1_dicts + embedding2_dicts

    # Save clusters list with pickle
    if data_args.pickle_save:
        dirname = os.path.dirname(data_args.clusters_dump_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        with open(data_args.clusters_dump_name, 'wb') as f:
            pickle.dump(clusters, f)

        with open(data_args.embeddings_dump_name, 'wb') as f:
            pickle.dump(embedding_concat_dicts, f)


if __name__ == '__main__':
    main()
