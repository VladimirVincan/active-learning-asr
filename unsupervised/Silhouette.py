import os
import pickle
import sys
from dataclasses import dataclass, field

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
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


class Silhouette():
    def __init__(self, data_args):
        with open(data_args.embeddings_dump_name, 'rb') as f:
            print('Opening embeddings')

            embeddings_dicts = pickle.load(f)
            self._embeddings_tensor = torch.zeros(1, 1, 512)
            for i, emb_dict in enumerate(embeddings_dicts):
                if i == 0:
                    self._embeddings_tensor = emb_dict['embedding']
                else:
                    self._embeddings_tensor = torch.cat((self._embeddings_tensor, emb_dict['embedding']))
            self._embeddings_array = self._embeddings_tensor.squeeze().numpy()

        with open(data_args.clusters_dump_name, 'rb') as f:
            print('Opening clusters')
            clusters_dicts = pickle.load(f)
            self._cluster_array = [d['cluster'] for d in clusters_dicts]

        self._num_clusters = data_args.num_clusters

    def get_slihouette_score(self):
        self._silhouette_avg = silhouette_score(self._embeddings_array, self._cluster_array)
        return self._silhouette_avg

    def silhouette_plot(self):
        fig, (ax1) = plt.subplots(1, 1)
        sample_silhouette_values = silhouette_samples(self._embeddings_array, self._cluster_array)

        y_lower = 10
        for i in range(self._num_clusters):
            """
            Write np.array(self._cluster_array)

            https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
            This is the input of sample_silhouette_values:
                clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                cluster_labels = clusterer.fit_predict(X)
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            This is the input shape:
                kmeans.predict([[0, 0], [12, 3]])
                array([1, 0], dtype=int32)
            """
            ith_cluster_silhouette_values = sample_silhouette_values[np.array(self._cluster_array) == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / self._num_clusters)

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
               )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=self._silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.show()



def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        data_args = parser.parse_args_into_dataclasses()
    data_args = data_args[0]

    silhouette = Silhouette(data_args)
    print(silhouette.get_slihouette_score())
    silhouette.silhouette_plot()


if __name__ == '__main__':
    main()
