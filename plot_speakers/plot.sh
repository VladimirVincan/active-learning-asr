#! /bin/bash

# x vectors ------------------------------

python3 SpeakerSelector.py SpeakerSelector.json

python3 ../unsupervised/x_vectors.py x_vectors_1.json
python3 ../unsupervised/x_vectors.py x_vectors_2.json

python3 ../unsupervised/clusterize_and_merge_vectors.py x_vectors.json
python3 ../unsupervised/pca.py x_vectors.json


# i vectors ------------------------------
location=$pwd
echo $location

cd ~/ivector-xvector/ivector
./enroll.sh ~/active-learning-asr/plot_speakers/speaker_data/speaker1
cd ..
~/kaldi/src/bin/copy-vector ark:ivector/data/feat/ivectors_enroll_mfcc/ivector.1.ark ark,t:- >ivector_speaker1.txt
python3 format_norm.py --vector_path='ivector_speaker1.txt' --save_path='ivector_speaker1.npz'

cd ~/ivector-xvector/ivector
./enroll.sh ~/active-learning-asr/plot_speakers/speaker_data/speaker2
cd ..
~/kaldi/src/bin/copy-vector ark:ivector/data/feat/ivectors_enroll_mfcc/ivector.1.ark ark,t:- >ivector_speaker2.txt
python3 format_norm.py --vector_path='ivector_speaker2.txt' --save_path='ivector_speaker2.npz'

echo "===================================="

cd ~/active-learning-asr/plot_speakers
python3 ../unsupervised/i_vector.py i_vector_1.json
python3 ../unsupervised/i_vector.py i_vector_2.json

python3 ../unsupervised/clusterize_and_merge_vectors.py i_vector.json
python3 ../unsupervised/pca.py i_vector.json
