import os
import sys
from dataclasses import dataclass, field

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from transformers import HfArgumentParser


@dataclass
class ModelArguments:
    source: str = field(
        default='speechbrain/spkrec-xvect-voxceleb'
    )
    savedir: str = field(
        default='pretrained_models/spkrec-xvect-voxceleb'
    )


@dataclass
class DataArguments:
    audio_file: str = field(
        default=None,
        metadata={'help': 'Audio file which will be encoded'}
    )
    audio_folder: str = field(
        default=None,
        metadata={'help': 'Audio folder which will be encoded'}
    )
    pickle_save: bool = field(
        default=False,
        metadata={'help': 'If embeddings should be saved as binary file'}
    )
    embeddings_dump_name: str = field(
        default='embeddings/cv16.pkl',
        metadata={'help': 'Name of pkl dump file of embeddings. Used if pickle_save is True.'}
    )


class Model:
    def __init__(self, model_args):
        self.__classifier = EncoderClassifier.from_hparams(source=model_args.source, savedir=model_args.savedir)

    def get_embedding(self, audio_file):
        signal, fs = torchaudio.load(audio_file)
        if fs != 16000:
            signal = torchaudio.functional.resample(signal,orig_freq=fs, new_freq=16000)
        embedding = self.__classifier.encode_batch(signal)
        return embedding


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    model = Model(model_args)
    embeddings = torch.zeros(1, 1, 512)

    embeddings = []
    if data_args.audio_folder is not None:
        for i, basename in enumerate(sorted(os.listdir(data_args.audio_folder))):
            filename = os.path.join(data_args.audio_folder, basename)
            if os.path.isdir(filename):
                continue
            print(basename)
            embedding_dict = {}
            embedding_dict['filename'] = os.path.basename(filename)
            embedding_dict['embedding'] = model.get_embedding(filename)
            embeddings.append(embedding_dict)

    elif data_args.audio_file is not None:
        embeddings = model.get_embedding(data_args.audio_file)

    else:
        raise Exception('Data path not defined!')

    # print(embeddings)
    # print(len(embeddings))

    if data_args.pickle_save:
        import pickle

        dirname = os.path.dirname(data_args.embeddings_dump_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        with open(data_args.embeddings_dump_name,'wb') as f:
            pickle.dump(embeddings, f)


if __name__ == '__main__':
    main()


