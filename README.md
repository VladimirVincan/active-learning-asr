# How to run x-vectors
Run the following command to create embeddings: `python3 x-vectors.py x-vectors.json` \
The embeddings will be created in the `embeddings` folder. 

Run the following command to clusterize embeddings: `python3 k-means.py k-means.json` \
The clusters will be created in the `clusters` folder. 

Run the following command to see clusters: `python3 read-clusters.py read-clusters.json`

Please correct the `audio_folder` in the `x-vectors.json` file with the path to your data.

Install https://github.com/subhadarship/kmeans_pytorch

If Nvidia is creating problems, add to .bashrc:

```bash
NVIDIA_PATH=/usr/local/lib/python3.10/dist-packages/nvidia
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NVIDIA_PATH}/cufft/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NVIDIA_PATH}/cuda_cupti/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NVIDIA_PATH}/cusparse/lib/
```
