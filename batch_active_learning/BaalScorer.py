import csv
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import psutil
import ray
import torch
import torch.multiprocessing as mp
from baal.bayesian.dropout import MCDropoutModule
from datasets import load_dataset
from jiwer import cer, wer
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
        default='file_name',
        metadata={'help': 'Name of column name that has text labels of corresponding audio files.'}
    )
    speaker_column: str = field(
        default='speaker_id',
        metadata={'help': 'Name of column name that has names/ids of speakers.'}
    )
    algorithm: str = field(
        default='inverse',
        metadata={'help': 'inverse or smca'}
    )
    num_cpus: int = field(
        default=2
    )


def load_dataset_fn(data_args):
    ds = load_dataset(data_args.dataset_dir, split='train')
    # ds = load_dataset(data_args.dataset_dir, split='validation[:8%]')
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

df = pd.read_csv(os.path.join(data_args.dataset_dir, 'metadata.csv'))
df['path'] = df['file_name']
df.to_csv(os.path.join(data_args.dataset_dir, 'metadata.csv'), index=False)
processor = Wav2Vec2Processor.from_pretrained(data_args.model_dir)
ds = load_dataset_fn(data_args)

# ============================= TRANSCRIPTION ============================

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

# ============================= END TRANSCRIPTION ============================


# ============================= INVERSE ============================

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
    results = pd.DataFrame(columns=[data_args.path_column, 'uncertainty'])

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
        dict = {data_args.path_column: speech_sample['path'], 'uncertainty': result}
        dict = pd.DataFrame(dict, index=[0])
        print(dict)
        results = pd.concat([results, dict], axis=0, ignore_index=True)

    results.to_csv(data_args.csv, index=False)

    ray.shutdown()

def calculate_uncertainty_for_all_samples_sequential():
    """
    Comment @ray.remote in calculate_uncertainty_for_sample in order to enable this fn to work.
    """
    print('--- STARTING UNCERTAINTY SEQUENTIAL ---')
    results = pd.DataFrame(columns=[data_args.path_column, 'uncertainty'])
    for i, speech_sample in enumerate(ds):
        result = calculate_uncertainty_for_sample(speech_sample)
        dict = {data_args.path_column: speech_sample['path'], 'uncertainty': result}
        results = pd.concat([results, dict], axis=0, ignore_index=True)
        print(dict)

    results.to_csv(data_args.csv, index=False)

# ============================= END INVERSE ============================


# ============================= SMCA ============================

if data_args.algorithm == 'smca':
    model_smca = Wav2Vec2ForCTC.from_pretrained(data_args.model_dir)
    mc_dropout_model_smca = MCDropoutModule(model_smca)

def calculate_uncertainty_for_sample_SMCA(speech_sample):
    base_transcription = transcribe_using_base_model(model_smca, processor, speech_sample)
    dropout_transcription = transcribe_using_dropout(mc_dropout_model_smca, processor, speech_sample)
    return cer(base_transcription, dropout_transcription)

def calculate_uncertainty_for_all_samples_sequential_SMCA():
    future_uncertainties = []
    print('--- STARTING UNCERTAINTY SEQUENTIAL SMCA ---')
    results = pd.DataFrame(columns=[data_args.path_column, 'uncertainty'])
    for i, speech_sample in enumerate(ds):
        uncertainty = calculate_uncertainty_for_sample_SMCA(speech_sample)
        dict = {data_args.path_column: speech_sample['path'], 'uncertainty': uncertainty}
        new_row = pd.DataFrame([dict])
        results = pd.concat([results, new_row], ignore_index=True)
        print(dict)

    results.to_csv(data_args.csv, index=False)

# ============================= END SMCA ============================

def main():
    # num_cpus = psutil.cpu_count(logical=True)
    print('num cpus: ' + str(data_args.num_cpus))
    if data_args.algorithm == 'inverse':
        print('starting inverse parallel')
        start_time = time.time()
        ray.init(num_cpus=data_args.num_cpus)
        calculate_uncertainty_for_all_samples_parallel()
        print("duration =", time.time() - start_time)
    elif data_args.algorithm == 'smca':
        print('starting smca sequential')
        calculate_uncertainty_for_all_samples_sequential_SMCA()
    elif data_args.alogithm == 'inverse_sequential':
        print('starting inverse sequential')
        start_time = time.time()
        calculate_uncertainty_for_all_samples_sequential()
        print("duration =", time.time() - start_time)
    else:
        raise ValueError('Algorithm not defined!')


if __name__ == '__main__':
    main()
