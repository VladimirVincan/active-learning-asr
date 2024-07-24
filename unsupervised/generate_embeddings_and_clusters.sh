rm -rf ../embeddings
rm -rf ../clusters
# embeddings and clusters are called final_train instead of just final
# because we will calculate embeddings and clusters for the train split of the dataset

mkdir ../embeddings
mkdir ../clusters

python3 x_vectors.py \
        --audio_folder="../data/final/clips/train" \
        --embeddings_dump_name="../embeddings/final_train.pkl" \
        --pickle_save="True"

python3 k_means.py \
        --embeddings_dump_name="../embeddings/final_train.pkl" \
        --clusters_dump_name="../clusters/final_train.pkl" \
        --pickle_save="True"
