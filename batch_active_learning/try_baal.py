# Wav2Vec in Baal

from datasets import load_dataset, DatasetDict
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments

from baal.active.heuristics import BALD
from baal.bayesian.dropout import patch_module, MCDropoutModule
from baal.transformers_trainer_wrapper import BaalTransformersTrainer
from jiwer import wer
import multiprocessing
import torch.multiprocessing as mp

PARALLEL = False

# load model and tokenizer
model_path = "facebook/wav2vec2-base-960h"
#model_path = "philschmid/tiny-random-wav2vec2"
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
#ds = load_dataset("mozilla-foundation/common_voice_16_1", "sr", split="test")

# Preprocess the audio and set format to torch.
ds_processed = (
    ds.map(
        lambda u: {k: v[0] for k, v in processor(u["audio"]["array"], return_tensors="pt", padding="longest").items()})
    .with_format("torch"))


speech_sample = ds_processed[2]
label = speech_sample["text"]
#print("Label:", label)

def TranscribeUsingDropout(processor, model, speech_sample):
    with MCDropoutModule(model) as mcdropout_model:
        input_values = speech_sample["input_values"]

        # Ensure input is of correct shape [batch_size, sequence_length]
        input_values = torch.squeeze(input_values)

        # Ensure input_values has a batch dimension
        if input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)

        with torch.no_grad():  # Disable gradient calculation for inference
            logits = mcdropout_model(input_values).logits  # Get logits from the model

        # Decode logits to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        return transcription



def TranscribeUsingBaseModel(processor, model, speech_sample):
    input_values = speech_sample["input_values"]

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


def CalculateUncertaintyForSample(processor, speech_sample, model):
    
    base_transcription = TranscribeUsingBaseModel(processor, model, speech_sample)
    wer_list = []
    for i in range(20):
        transcription = TranscribeUsingDropout(processor, model, speech_sample)
        wer_list.append(wer(base_transcription, transcription))
    return sum(wer_list) / len(wer_list)




def CalculateUncertaintyFor_N_Samples_Sequential(ds_processed, n_samples):
    results = []
    for i in range(n_samples):
        speech_sample = ds_processed[i]
        uncertainty = CalculateUncertaintyForSample(processor, speech_sample, model)
        results.append(uncertainty)
    
    for i, result in enumerate(results):
        speech_sample = ds_processed[i]
        print("Uncertainty for sample", speech_sample['file'], "is:", result)

#----------------- Parallel version -----------------#
        
def CalculateUncertaintyForSampleParallel(speech_sample):
    print("Hi from parallel function")
    # we use global model and processor, because we cannot pass them as arguments to the parallel function
    # due to the fact that the parallel function is called by the pool.apply_async function
    # and the arguments to the parallel function must be picklable (and torch model is not picklable)
    
    return CalculateUncertaintyForSample(processor, speech_sample, model)


def CalculateUncertaintyFor_N_Samples_Parallel(ctx, ds_processed, n_samples):
    
    pool = ctx.Pool(processes=n_samples)
    #pool = multiprocessing.Pool()
    results = []
    for i in range(n_samples):
        speech_sample = ds_processed[i]
        result = pool.apply_async(CalculateUncertaintyForSampleParallel, (([speech_sample])))
        results.append(result)
    
    pool.close()
    pool.join()
    uncertainties = [result.get() for result in results]

    
    # TODO for loop above is necessary to be parallelized, 
    # but the for loop will be replaced with a sort and select top results
    for i, uncertainty in enumerate(uncertainties):
        speech_sample = ds_processed[i]
        print("Uncertainty for sample", speech_sample['file'], "is:", uncertainty)


def CalculateUncertaintyFor_N_Samples(ctx, ds_processed, n_samples, parallel=False):
    if parallel:
        CalculateUncertaintyFor_N_Samples_Parallel(ctx, ds_processed, n_samples)
    else:
        CalculateUncertaintyFor_N_Samples_Sequential(ds_processed, n_samples)


def main():
    ctx = mp.get_context('spawn') 
    CalculateUncertaintyFor_N_Samples(ctx, ds_processed, n_samples=4, parallel=PARALLEL)


if __name__ == '__main__':
    main()


'''
for i in range(5):
    transcription = TranscribeUsingDropout(processor, model, speech_sample)
    print("Transcription with monte carlo dropout:", transcription)
    print("WER given the label:", wer(label, transcription))


transcription = TranscribeUsingBaseModel(processor, model, speech_sample)
print("Transcription with base model:", transcription)
print("WER given the label:", wer(label, transcription))
'''

