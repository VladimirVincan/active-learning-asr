#!/bin/bash

speaker_ids="211 4014 730 2989 8063 4195 27 125 118 1867 5339 1502 7302 5561 1926 4051 6531 6078 1455 7190 1081 2289 4406 5652 1447 1594 1963 481 1898 4640 8629 201 2136 3240 5049 198 460 1116 1737 3259 1235 5393 307 831 4441 39 83 89 2007 2391 3242 3374 4680 5688 8238 254 311 322 1970 2817 2911 5750 7635 6081 7278 2764 3486 5808 8088 2952 3436 3723 5192 7367 26 78 669 1578 3112 3830 4088 6476 8226 8465 8468 32 405 587 1246 3699 3879 6147 200 426 2002 2436 2691 2843 3214 5390 6415 6836 6880 8051 8098 441 887 1841 2182 5022 5463 6272 7078 7148 7178 7505 7800 8609 8975 40 150 248 250 403 3526 4137 6181 6209 7447 8425 298 302 374 911 2196 3235 3440 3982 3983 5322 5703 5867 6367 6385 8324 163 1088 1723 2518 4267 5104 5456 5789 6019 7402 8095 8419 19 446 4297 4397 4813 4853 7059 7067 7113 7226 7264 8770 233 625 909 2159 2416 3857 4018 4898 6064 6454 7859 8108 8747 1263 2836 6848 8123 87 196 696 1069 2514 3168 4160 5163 5678 6437 7511 226 1334 2893 3664 4362 4788 4830 6529 6818 8838 2092 7780 7794 8312 8797 1743 3947 7517 103 1553 1624 4340 839 8630 328 3807 5778 1355 1363 3607 60 4481 1034 412 2910 229 4859 1098 6563 2384 332 6000 5514 8580 289 6925 458 1040 4214 8014 1183 445 1992 7312"

speakers_list=($speaker_ids)

for speaker in "${speakers_list[@]}"; do
    echo "$speaker"
    python3 x_vectors.py \
            --audio_folder "../data/ls_single/$speaker/clips" \
            --embeddings_dump_name "../embeddings/ls_single/$speaker.pkl" \
            --pickle_save="True"
    python3 clusterize_and_merge_vectors.py \
            --embeddings1_dump_name "../embeddings/common_voice.pkl" \
            --embeddings2_dump_name "../embeddings/ls_single/$speaker.pkl" \
            --embeddings_dump_name "../embeddings/x_vectors_merge.pkl" \
            --clusters_dump_name "../clusters/x_vectors_merge.pkl" \
            --pickle_save="True"
    python3 Silhouette.py \
            --embeddings_dump_name "../embeddings/x_vectors_merge.pkl" \
            --clusters_dump_name "../clusters/x_vectors_merge.pkl" \
            --num_clusters="2" \
            --projection_dims="3"
done
