import csv
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psutil
import ray
import torch
import torch.multiprocessing as mp
from baal.bayesian.dropout import MCDropoutModule
from datasets import load_dataset
from jiwer import wer
from transformers import HfArgumentParser, Wav2Vec2ForCTC, Wav2Vec2Processor


@dataclass
class DataArguments:
    dataset_dir: str = field(
        default='../data/cluster_subtractor'
    )
    model_dir: str = field(
        default='../model/0706f64c-4f4b-4d26-ba24-28e841bfa371'
    )
    csv: str = field(
        default='../uncertainties/results.csv',
        metadata={'help': 'Name of folder where data will be stored.'})
    label_column: str = field(
        default='sentence',
        metadata={'help': 'Name of column name that has text labels of corresponding audio files.'}
    )
    path_column: str = field(
        default='path',
        metadata={'help': 'Name of column name that has text labels of corresponding audio files.'}
    )
    speaker_column: str = field(
        default='speaker_id',
        metadata={'help': 'Name of column name that has names/ids of speakers.'}
    )


def load_dataset_fn(data_args):
    # ds = load_dataset(data_args.dataset_dir, split='train+validation')
    ds = load_dataset(data_args.dataset_dir, split='validation[:2%]')
    ds = (
        ds.map(
            lambda u:
            {k: v[0] for k, v in processor(
                u["audio"]["array"],
                return_tensors="pt",
                padding="longest",
                sampling_rate=16000).items()}
        ).with_format("torch"))
    # num_rows = len(ds)
    return ds


torch.manual_seed(42)
np.random.seed(42)

parser = HfArgumentParser((DataArguments))
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else:
    data_args = parser.parse_args_into_dataclasses()
data_args = data_args[0]

processor = Wav2Vec2Processor.from_pretrained(data_args.model_dir)
ds = load_dataset_fn(data_args)


def transcribe_using_base_model(model, processor, speech_sample):
    input_values = speech_sample['input_values']

    # Ensure input is of correct shape [batch_size, sequence_length]
    input_values = torch.squeeze(input_values)

    # Ensure input_values has a batch dimension
    if input_values.dim() == 1:
        input_values = input_values.unsqueeze(0)

    with torch.no_grad():  # Disable gradient calculation for inference
        logits = model(input_values).logits  # Get logits from the model

    # Decode logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription

def transcribe_using_dropout(model, processor, speech_sample):
    mc_dropout_model = MCDropoutModule(model)
    input_values = speech_sample['input_values']

    # Ensure input is of correct shape [batch_size, sequence_length]
    input_values = torch.squeeze(input_values)

    # Ensure input_values has a batch dimension
    if input_values.dim() == 1:
        input_values = input_values.unsqueeze(0)

    with torch.no_grad():  # Disable gradient calculation for inference
        logits = mc_dropout_model(input_values).logits  # Get logits from the model

    # Decode logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription

@ray.remote
def calculate_uncertainty_for_sample(processor_id, speech_sample, NUM_ITERATIONS=20):
    model = Wav2Vec2ForCTC.from_pretrained(data_args.model_dir)
    base_transcription = transcribe_using_base_model(model, processor_id, speech_sample)
    wer_list = []
    for i in range(NUM_ITERATIONS):
        transcription = transcribe_using_dropout(model, processor_id, speech_sample)
        wer_list.append(wer(base_transcription, transcription))
    return sum(wer_list)/len(wer_list)

def calculate_uncertainty_for_all_samples_parallel():
    print('--- STARTING UNCERTAINTY PARALLEL ---')
    results = pd.DataFrame(columns=['path', 'uncertainty'])

    # procesor moze da se stavi u ray.put, a model ne moze zbog serijalizacije, pa njega saljemo kao argument
    processor = Wav2Vec2Processor.from_pretrained(data_args.model_dir)
    processor_id = ray.put(processor)

    future_uncertainties = []
    for i, speech_sample in enumerate(ds):
        print(i)
        future_uncertainty = calculate_uncertainty_for_sample.remote(processor_id, speech_sample)
        future_uncertainties.append(future_uncertainty)

    # collect results
    uncertainties = ray.get(future_uncertainties)
    print('--- AFTER RAY GET ---')

    for i, result in enumerate(uncertainties):
        speech_sample = ds[i]
        dict = {'path': speech_sample['path'], 'uncertainty': result.get()}
        # results = results.append(dict, ignore_index=True)
        results = pd.concat([results, dict], axis=1, ignore_index=True)
        print(dict)

    results.to_csv(data_args.csv, index=False)

    ray.shutdown()


def main():
    num_cpus = psutil.cpu_count(logical=True)
    print(num_cpus)
    ray.init(num_cpus=2)
    calculate_uncertainty_for_all_samples_parallel()


if __name__ == '__main__':
    main()
