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

conda setup:
    conda create -n alasr2 python=3.10 numpy pandas
    python3 -m pip install torch ray jiwer baal transformers psutil
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


## Citation

If you use this work, please cite:

**Text format:**

> Ognjen Kundacina, Vladimir Vincan, Dragisa Miskovic, "Combining X-Vectors and Bayesian Batch Active Learning: Two-Stage Active Learning Pipeline for Speech Recognition," IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 1862-1876, 2025. DOI: 10.1109/TASLPRO.2025.3565216

**BibTeX format:**

```bibtex
@ARTICLE{10979459,
  author={Kundacina, Ognjen and Vincan, Vladimir and Miskovic, Dragisa},
  journal={IEEE Transactions on Audio, Speech and Language Processing}, 
  title={Combining X-Vectors and Bayesian Batch Active Learning: Two-Stage Active Learning Pipeline for Speech Recognition}, 
  year={2025},
  volume={33},
  number={},
  pages={1862-1876},
  keywords={Data models;Training;Active learning;Uncertainty;Labeling;Bayes methods;Recording;Deep learning;Pipelines;Tuning;Active learning;batch active learning;Bayesian active learning;speech recognition;x-vectors},
  doi={10.1109/TASLPRO.2025.3565216}}
