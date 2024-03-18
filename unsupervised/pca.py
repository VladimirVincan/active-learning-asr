import os
import pickle
import sys
from dataclasses import dataclass, field

import matplotlib.cm as cm
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
    projection_dims: int = field(
        default=2,
        metadata={'help': '2 for 2d projection or 3 for 3d projection'}
    )


def tensor2row(tensor):
    row = {}
    for i in range(len(tensor)):
        row[f'column{i}'] = tensor[i]
    return row


def embedding2df(embedding_dicts):
    df = pd.DataFrame()
    for embedding in embedding_dicts:
        tensor = embedding['embedding'].numpy()[0][0]  # if 1-d array problem, comment out last [0]!!
        row = tensor2row(tensor)
        df = df.append(row, ignore_index=True)
    return df


def get_cluster_list(clusters_dicts):
    cluster_list = []
    for cluster in clusters_dicts:
        cluster_number = cluster['cluster']
        cluster_list.append(cluster_number)
    return cluster_list


def get_names(clusters_dicts):
    names = []
    for cluster in clusters_dicts:
        name = cluster['filename']
        names.append(name)
    return names


def cluster_list_to_color_list(data_args, cluster_list):
    num_clusters = data_args.num_clusters
    color_list = []
    for i in cluster_list:
        color = cm.nipy_spectral(float(i) / num_clusters)
        color_list.append(color)

    return color_list


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

    with open(data_args.embeddings_dump_name, 'rb') as f:
        print('Opening embeddings')
        embedding_dicts = pickle.load(f)

    with open(data_args.clusters_dump_name, 'rb') as f:
        print('Opening clusters')
        clusters_dicts = pickle.load(f)

    print('creating df')
    df = embedding2df(embedding_dicts)
    cluster_list = get_cluster_list(clusters_dicts)
    color_list = cluster_list_to_color_list(data_args, cluster_list)

    print('pca start')
    pca = PCA(n_components=data_args.projection_dims).fit(df)
    x_pca = pca.transform(df)
    print(pca.explained_variance_ratio_)

    print('plot start')
    if data_args.projection_dims == 3:
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes(projection='3d')
        ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2],
                   c=color_list,
                   cmap='nipy_spectral')

        # labeling x and y axes
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
    else:
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(x_pca[:, 0], x_pca[:, 1],
                   c=color_list,
                   cmap='nipy_spectral')

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
