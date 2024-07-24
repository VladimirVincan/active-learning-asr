import os
import shutil
import sys
from dataclasses import dataclass, field

import torch
import torchaudio
import torchaudio.functional as F
from transformers import HfArgumentParser


@dataclass
class DataArguments:
    input_audio_folder: str = field(
        default=None,
        metadata={'help': 'Audio folder which will be encoded'}
    )
    output_audio_folder: str = field(
        default=None,
        metadata={'help': 'Audio folder which will be encoded'}
    )
    noise_file: str = field(
        default='/home/bici/Downloads/whitenoise3.wav'
    )
    audio_file: str = field(
        default=None
    )
    snr: int = field(
        default=None
    )


def add_noise(audio_file, noise_file, snr):
        signal, fs = torchaudio.load(audio_file)
        if fs != 16000:
            signal = torchaudio.functional.resample(signal,orig_freq=fs, new_freq=16000)
        if snr is not None:
            noise, fs = torchaudio.load(noise_file)
            if fs != 16000:
                noise = torchaudio.functional.resample(noise, orig_freq=fs, new_freq=16000)
            noise = noise[:, :signal.shape[1]]
            snr_dbs = torch.tensor([snr])
            signal = F.add_noise(signal, noise, snr_dbs)
        return signal


def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]), allow_extra_keys=True)
    else:
        data_args = parser.parse_args_into_dataclasses()
    data_args = data_args[0]

    if os.path.exists(data_args.output_audio_folder) and os.path.isdir(data_args.output_audio_folder):
        shutil.rmtree(data_args.output_audio_folder)
    os.mkdir(data_args.output_audio_folder)

    if data_args.input_audio_folder is not None:
        for i, basename in enumerate(sorted(os.listdir(data_args.input_audio_folder))):
            filename = os.path.join(data_args.input_audio_folder, basename)
            if os.path.isdir(filename):
                continue
            print(basename)
            signal = add_noise(filename, data_args.noise_file, data_args.snr)
            filename_o = os.path.join(data_args.output_audio_folder, basename)
            # Format required: PCM Linear with 16 bits
            # https://groups.google.com/g/kaldi-help/c/4ckyu7wSw1s
            # https://groups.google.com/g/kaldi-help/c/p4Cog5_NqVo
            torchaudio.save(filename_o, signal, 16_000, format='wav', encoding='PCM_S', bits_per_sample=16)
    elif data_args.audio_file is not None:
        embeddings = add_noise(data_args.audio_file, data_args.noise_file, data_args.snr)
        basename = os.path.basename(data_args.audio_file)
        filename_o = os.path.join(data_args.output_audio_folder, basename)
        torchaudio.save(filename_o, signal, 16_000)

    else:
        raise Exception('Data path not defined!')


if __name__ == '__main__':
    main()
