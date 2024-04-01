import os
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from transformers import HfArgumentParser

"""
Expected library format:
home/
  kaldi/
  ivector-xvector/
  active-learning-asr/
"""


@dataclass
class DataArguments:
    npz_file: str = field(
        default='~/ivector-xvector/i_vector.npz',
        metadata={'help': 'Audio file which will be encoded'}
    )
    pickle_save: bool = field(
        default=False,
        metadata={'help': 'If embeddings should be saved as binary file'}
    )
    embeddings_dump_name: str = field(
        default='../embeddings/i_vector.pkl',
        metadata={'help': 'Name of pkl dump file of embeddings. Used if pickle_save is True.'}
    )


def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args = parser.parse_args_into_dataclasses()
    data_args = data_args[0]

    ivector_numpy = np.load(data_args.npz_file, allow_pickle=True)
    ivector_tensor = torch.from_numpy(ivector_numpy['features'])

    embeddings = torch.zeros(1, 1, 400)

    embeddings = []
    for i, name in enumerate(ivector_numpy['data_path']):
        basename = name['pic_path'] + '.wav'
        print(basename)
        embedding_dict = {
            'filename': basename,
            'embedding': ivector_tensor[i].unsqueeze(0).unsqueeze(0).to(torch.float32)
        }
        embeddings.append(embedding_dict)

    if data_args.pickle_save:
        import pickle

        dirname = os.path.dirname(data_args.embeddings_dump_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        with open(data_args.embeddings_dump_name,'wb') as f:
            pickle.dump(embeddings, f)


if __name__ == '__main__':
    main()
