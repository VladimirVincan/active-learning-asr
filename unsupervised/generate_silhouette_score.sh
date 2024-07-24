echo "----- Starting x vectors -----" > xvector_with_noise.txt

for i in $(seq 1 30);
do
    echo "xvector iteration ${i}"
    START=$(date +%s.%N)
    python3 x_vectors.py \
            --audio_folder "../data/librispeech/clips" \
            --embeddings_dump_name "../embeddings/librispeech_snr${i}.pkl" \
            --pickle_save="True" \
            --snr=$i
    END=$(date +%s.%N)
    DIFF_LS=$(echo "$END - $START" | bc)
    START=$(date +%s.%N)
    python3 x_vectors.py \
            --audio_folder "../data/common_voice/clips" \
            --embeddings_dump_name "../embeddings/common_voice_snr${i}.pkl" \
            --pickle_save="True" \
            --snr=$i
    END=$(date +%s.%N)
    DIFF_CV=$(echo "$END - $START" | bc)
    python3 clusterize_and_merge_vectors.py \
            --embeddings1_dump_name "../embeddings/common_voice_snr${i}.pkl" \
            --embeddings2_dump_name "../embeddings/librispeech_snr${i}.pkl" \
            --embeddings_dump_name "../embeddings/x_vectors_merge_snr${i}.pkl" \
            --clusters_dump_name "../clusters/x_vectors_merge_snr${i}.pkl" \
            --pickle_save="True"
    score=$(python3 Silhouette.py \
                    --embeddings_dump_name "../embeddings/x_vectors_merge_snr${i}.pkl" \
                    --clusters_dump_name "../clusters/x_vectors_merge_snr${i}.pkl" \
                    --num_clusters="2" \
                    --projection_dims="2")
    # echo "xvector iteration ${i}: ${score}. LS TIME: ${DIFF_LS}, CV_TIME: ${DIFF_CV}"
    # echo "xvector iteration ${i}: ${score}. LS TIME: ${DIFF_LS}, CV_TIME: ${DIFF_CV}" >> xvector_with_noise.txt
    echo "xvector iteration ${i}: ${score}."
    echo "xvector iteration ${i}: ${score}." >> xvector_with_noise.txt
done
