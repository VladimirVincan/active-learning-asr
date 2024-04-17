if [ ! -f "librispeech/metadata_sorted_selected_dbscan.csv" ]; then
    cp metadata_sorted_selected_dbscan.csv librispeech/metadata_sorted_selected_dbscan.csv
fi

rm -rf ls_dump_split_1
rm -rf ls_dump_split_2
rm -rf ls_cv_train_dump
rm -rf final

python3 TrainingCreator.py \
        --csv1="librispeech/metadata_sorted_selected_dbscan.csv" \
        --split1="split" \
        --split2="none" \
        --split_size="0.3" \
        --folder="ls_dump"

python3 TrainingCreator.py \
        --csv1="ls_dump_split_1/metadata.csv" \
        --csv2="common_voice/metadata.csv" \
        --split1="same" \
        --split2="same" \
        --folder="ls_cv_train_dump"

python3 TrainingCreator.py \
        --csv1="ls_cv_train_dump/metadata.csv" \
        --csv2="ls_dump_split_2/metadata.csv" \
        --split1="train" \
        --split2="dev" \
        --folder="final"

rm -rf ls_dump
rm -rf ls_dump_split_1
rm -rf ls_dump_split_2
rm -rf ls_cv_train_dump
