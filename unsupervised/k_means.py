import os
import pickle
import sys
from dataclasses import dataclass, field

import torch
from sklearn.cluster import DBSCAN, KMeans
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
    pickle_save: bool = field(
        default=False,
        metadata={'help': 'If embeddings should be saved as binary file'}
    )
    clusters_dump_name: str = field(
        default='clusters/cv16.pkl',
        metadata={'help': 'Name of pkl dump file of clusters list.'}
    )


class Model:
    def __init__(self, data_args, embedding_dicts):
        self.__num_clusters = data_args.num_clusters

        self.__embedding_tensor = torch.zeros(1, 1, 512)
        for i, emb_dict in enumerate(embedding_dicts):
            if i == 0:
                self.__embedding_tensor = emb_dict['embedding']
            else:
                self.__embedding_tensor = torch.cat((self.__embedding_tensor, emb_dict['embedding']))
        self.__embedding_array = self.__embedding_tensor.squeeze().numpy()
        # print(self.__embedding_array)

        # self.__kmeans = KMeans(
        #     n_clusters=self.__num_clusters,
        #     random_state=None,
        #     n_init='auto'
        # )
        self.__kmeans = DBSCAN(eps=63, min_samples=50)
        self.__kmeans.fit(self.__embedding_array)

    def get_cluster_ids(self):
        return self.__kmeans.labels_

    def get_embedding_tensor(self):
        return self.__embedding_tensor


def analyze_clusters_list(clusters_dicts):
    from collections import Counter
    clusters_list = get_clusters_list(clusters_dicts)
    element_count = Counter(clusters_list)
    for element, count in element_count.items():
        print(f'{element}: {count}')


def get_clusters_list(clusters_dicts):
    clusters_list = []
    for i, cluster_dict in enumerate(clusters_dicts):
        clusters_list.append(cluster_dict['cluster'])
    return clusters_list


def is_librispeech(filename):
    """
    common voice filename example: 5653997f2267bc3c86ef58ab780538510de60885b6039c12bfd729d357f575da0c3399af4c41f3c96527fa2a9d28c541ab0b55d27bc9e2008905df5c774c3935.wav
    librispeech filename example: 5652-39938-0066.wav
    """
    if len(filename) < 30:
        return True


def minmax_clusters(clusters_dicts):
    """
    Get the min(librispeech, commonvoice) from each group/cluster.

    Metric for analyzing how well clusters are separated.
    Create dict of dicts:
    {cluster_number: {'common_voice': value, 'librispeech': value}}
    """
    metric_dict = {}
    for i, cluster_dict in enumerate(clusters_dicts):
        cluster = cluster_dict['cluster']
        filename = cluster_dict['filename']
        if cluster in metric_dict:
            if is_librispeech(filename):
                metric_dict[cluster]['librispeech'] += 1
            else:
                metric_dict[cluster]['common_voice'] += 1
        else:
            d = {'librispeech': 0, 'common_voice': 0}
            if is_librispeech(filename):
                d['librispeech'] = 1
            else:
                d['common_voice'] = 1
            metric_dict[cluster] = d

    for key in metric_dict.keys():
        metric_dict[key]['overlap'] = min(metric_dict[key]['librispeech'], metric_dict[key]['common_voice'])
    from pprint import pprint
    pprint(metric_dict)


def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        data_args = parser.parse_args_into_dataclasses()

    # Remove this line if multiple dataclasses exist in parser
    data_args = data_args[0]

    print('--- opening embeddings ---')
    with open(data_args.embeddings_dump_name, 'rb') as f:
        embedding_dicts = pickle.load(f)

    print('--- init model ---')
    model = Model(data_args, embedding_dicts)

    cluster_array = model.get_cluster_ids()
    print(cluster_array)

    # Create clusters list
    clusters = []
    total = len(embedding_dicts)
    for i, emb_dict in enumerate(embedding_dicts):
        # print(str(i) + ' / ' + str(total))
        cluster_dict = {
            'filename': emb_dict['filename'],
            'cluster': cluster_array[i]
        }
        clusters.append(cluster_dict)
    # print(clusters)

    analyze_clusters_list(clusters)
    minmax_clusters(clusters)

    # Save clusters list with pickle
    if data_args.pickle_save:
        dirname = os.path.dirname(data_args.clusters_dump_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        with open(data_args.clusters_dump_name,'wb') as f:
            pickle.dump(clusters, f)


if __name__ == '__main__':
    main()
