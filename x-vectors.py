import os
import sys
from dataclasses import dataclass, field

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


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    classifier = EncoderClassifier.from_hparams(source=model_args.source, savedir=model_args.savedir)
    signal, fs = torchaudio.load(data_args.audio_file)
    embeddings = classifier.encode_batch(signal)

    print(embeddings)
    print(embeddings.shape)


if __name__ == '__main__':
    main()


