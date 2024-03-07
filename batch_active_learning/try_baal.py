# Wav2Vec in Baal

from datasets import load_dataset, DatasetDict
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments

from baal.active.heuristics import BALD
from baal.bayesian.dropout import patch_module, MCDropoutModule
from baal.transformers_trainer_wrapper import BaalTransformersTrainer
from jiwer import wer

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

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


def CalculateUncertaintyForSample(processor, model, speech_sample):
    base_transcription = TranscribeUsingBaseModel(processor, model, speech_sample)
    wer_list = []
    for i in range(20):
        transcription = TranscribeUsingDropout(processor, model, speech_sample)
        wer_list.append(wer(base_transcription, transcription))
    return sum(wer_list) / len(wer_list)


for i in range(20):
    speech_sample = ds_processed[i]
    print("Uncertainty for sample ", speech_sample['file'], " is: ", CalculateUncertaintyForSample(processor, model, speech_sample))
'''
for i in range(5):
    transcription = TranscribeUsingDropout(processor, model, speech_sample)
    print("Transcription with monte carlo dropout:", transcription)
    print("WER given the label:", wer(label, transcription))


transcription = TranscribeUsingBaseModel(processor, model, speech_sample)
print("Transcription with base model:", transcription)
print("WER given the label:", wer(label, transcription))
'''

