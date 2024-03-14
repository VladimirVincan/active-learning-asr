import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint

import datasets
import pandas as pd
import soundfile
from datasets import Audio, load_dataset
from pydub import AudioSegment
from transformers import HfArgumentParser

from DatasetDownloader import DataArguments, DatasetDownloader


class DatasetAnalyzer(DatasetDownloader):
    def __init__(self, data_args):
        super().__init__(data_args)
        self._df = pd.read_csv(os.path.join(self._folder, self._csv))

    def file_number(self):
        files = os.listdir(os.path.join(self._folder, 'clips'))
        num_files = len(files)
        return num_files

    def duration(self):
        """
        Returns the total duration of the audio folder in seconds.
        """
        total_duration_seconds = 0

        for filename in os.listdir(os.path.join(self._folder, 'clips')):
            if filename.endswith(self._extension):
                audio = AudioSegment.from_file(os.path.join(self._folder, 'clips', filename))
                total_duration_seconds += audio.duration_seconds

        return total_duration_seconds

    def speaker_counts(self, speaker_num=-1):
        """
        Prints the speaker id and number of audio files that the speaker appeared in.
        """
        speaker_counts = self._df[self._speaker_column].value_counts()
        speaker_counts = {k: v for k, v in sorted(speaker_counts.items(), key=lambda item: item[1], reverse=True)}
        if speaker_num >= 0:
            # create a shorter dict with the top speaker_num speakers
            speaker_counts = dict(list(speaker_counts.items())[:speaker_num])
        return speaker_counts

    def duration_per_speaker(self, speaker_num=-1):
        """
        Returns a dictionary where key is speaker id and value is total duration in seconds.
        """
        duration_dict = {}
        i = 0
        for index, row in self._df.iterrows():
            audio = AudioSegment.from_file(os.path.join(self._folder, 'clips', row[self._path_column]))
            duration = audio.duration_seconds

            if duration_dict.get(row[self._speaker_column]) is not None:
                duration_dict[row[self._speaker_column]] += duration
            else:
                duration_dict[row[self._speaker_column]] = duration

        # sort the dict
        duration_dict = dict(sorted(duration_dict.items(), key=lambda item: item[1], reverse=True))

        if speaker_num >= 0:
            # create a shorter dict with the top speaker_num speakers
            duration_dict = dict(list(duration_dict.items())[:speaker_num])

        return duration_dict


def main():
    parser = HfArgumentParser((DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args = parser.parse_args_into_dataclasses()
    data_args = data_args[0]

    dataset_analyzer = DatasetAnalyzer(data_args)

    print("----- Number of audio files per speaker -----")
    speaker_counts = dataset_analyzer.speaker_counts(5)
    pprint(speaker_counts, sort_dicts=False)

    print("----- Duration per speaker -----")
    duration_per_speaker = dataset_analyzer.duration_per_speaker(5)
    pprint(duration_per_speaker, sort_dicts=False)

    total_duration_seconds = dataset_analyzer.duration()
    print(f"Total duration of audio files in the folder: {total_duration_seconds:.2f} seconds")

    num_files = dataset_analyzer.file_number()
    print(f"Total number of audio files in the folder: {num_files}")


if __name__ == '__main__':
    main()
