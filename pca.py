import os
import pickle
import sys
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from transformers import HfArgumentParser


@dataclass
class DataArguments:
    embeddings_dump_name: str = field(
        default='embeddings/cv16.pkl',
        metadata={'help': 'Name of pkl dump file of embeddings.'}
    )
    num_clusters: int = field(
        default=2,
        metadata={'help': 'The k in k-means.'}
    )
    clusters_dump_name: str = field(
        default='clusters/cv16.pkl',
        metadata={'help': 'Name of pkl dump file of clusters list.'}
    )


def tensor2row(tensor):
    row = {}
    for i in range(len(tensor)):
        row[f'column{i}'] = tensor[i]
    return row


def embedding2df(embedding_dicts):
    df = pd.DataFrame()
    for embedding in embedding_dicts:
        tensor = embedding['embedding'].numpy()[0][0]
        row = tensor2row(tensor)
        df = df.append(row, ignore_index=True)
    return df


def get_cluster_list(clusters_dicts):
    cluster_list = []
    for cluster in clusters_dicts:
        cluster_number = cluster['cluster'].numpy()
        cluster_list.append(cluster_number)
    return cluster_list


def get_names(clusters_dicts):
    names = []
    for cluster in clusters_dicts:
        name = cluster['filename']
        names.append(name)
    return names


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

    with open(data_args.embeddings_dump_name, 'rb') as f:
        embedding_dicts = pickle.load(f)

    with open(data_args.clusters_dump_name, 'rb') as f:
        clusters_dicts = pickle.load(f)

    df = embedding2df(embedding_dicts)
    cluster_list = get_cluster_list(clusters_dicts)
    print(cluster_list)

    pca = PCA(n_components=2)
    pca.fit(df)
    x_pca = pca.transform(df)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_pca[:, 0], x_pca[:, 1],
            c=cluster_list,
            cmap='plasma')

    # labeling x and y axes
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

    # annotate:
    names = get_names(clusters_dicts)
    for i, txt in enumerate(names):
        plt.annotate(txt, (x_pca[i, 0], x_pca[i, 1]))

    plt.show()


if __name__ == '__main__':
    main()
