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

        self.__kmeans = KMeans(
            n_clusters=self.__num_clusters,
            random_state=None,
            n_init='auto'
        )
        self.__kmeans.fit(self.__embedding_array)

    def get_cluster_ids(self):
        return self.__kmeans.labels_

    def get_embedding_tensor(self):
        return self.__embedding_tensor


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
        embedding_dicts = pickle.load(f)

    model = Model(data_args, embedding_dicts)

    cluster_array = model.get_cluster_ids()
    print(cluster_array)

    # Create clusters list
    clusters = []
    for i, emb_dict in enumerate(embedding_dicts):
        cluster_dict = {
            'filename': emb_dict['filename'],
            'cluster': cluster_array[i]
        }
        clusters.append(cluster_dict)
    # print(clusters)

    # Save clusters list with pickle
    if data_args.pickle_save:
        dirname = os.path.dirname(data_args.clusters_dump_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        with open(data_args.clusters_dump_name,'wb') as f:
            pickle.dump(clusters, f)


if __name__ == '__main__':
    main()
