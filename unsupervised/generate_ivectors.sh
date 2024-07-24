#!/bin/bash

# echo "----- Starting i vectors -----" > ivector_with_noise.txt
AL_ROOT="${HOME}/active-learning-asr"
IXV_ROOT="${HOME}/ivector-xvector"
dataset=("common_voice" "librispeech")
# dataset=("common_voice")

for i in $(seq 2 2);
do
    echo "----- Iteration $i -----"
    for j in ${dataset[@]}; do
        START=$(date +%s.%N)
        echo "----- Dataset $j -----"
        if [ -d "../data/$j/snr" ]; then
            rm -Rf "../data/$j/snr";
        fi
        python3 ${AL_ROOT}/data/NoiseAdder.py \
                --input_audio_folder "${AL_ROOT}/data/$j/clips" \
                --output_audio_folder "${AL_ROOT}/data/$j/snr" \
                --snr=$i

        cd ${IXV_ROOT}/ivector
        # source enroll.sh ${AL_ROOT}/data/$j/snr
        source path.sh
        cd ${IXV_ROOT}
        pwd
        ${KALDI_ROOT}/src/bin/copy-vector ark:ivector/data/feat/ivectors_enroll_mfcc/ivector.1.ark ark,t:- >ivector.txt
        python3 format_norm.py --vector_path='ivector.txt' --save_path="${i}_${j}.npz"
        cd ${AL_ROOT}/unsupervised
        pwd
        python3 i_vector.py \
                --npz_file="${IXV_ROOT}/${i}_${j}.npz" \
                --pickle_save="True" \
                --embeddings_dump_name="${AL_ROOT}/embeddings/${j}_isnr${i}.pkl"
        END=$(date +%s.%N)
        DIFF=$(echo "$END - $START" | bc)
        echo "ivector snr ${i} for ${j} time: ${DIFF}."
        echo "ivector snr ${i} for ${j} time: ${DIFF}." >> ivector_with_noise_timings.txt
        echo ""
    done

    python3 clusterize_and_merge_vectors.py \
            --embeddings1_dump_name "${AL_ROOT}/embeddings/common_voice_isnr${i}.pkl" \
            --embeddings2_dump_name "${AL_ROOT}/embeddings/librispeech_isnr${i}.pkl" \
            --embeddings_dump_name "${AL_ROOT}/embeddings/i_vectors_merge_snr${i}.pkl" \
            --clusters_dump_name "${AL_ROOT}/clusters/i_vectors_merge_snr${i}.pkl" \
            --pickle_save="True"
    score=$(python3 Silhouette.py \
                    --embeddings_dump_name "../embeddings/i_vectors_merge_snr${i}.pkl" \
                    --clusters_dump_name "../clusters/i_vectors_merge_snr${i}.pkl" \
                    --num_clusters="2" \
                    --projection_dims="2")

    echo "ivector snr ${i} for ${j}: ${score}." >> ivector_with_noise.txt
    echo "ivector snr ${i} for ${j}: ${score}."
done
