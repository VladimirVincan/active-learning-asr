import os

from pydub import AudioSegment

# Input and output directories
input_folder = 'common_voice/clips48'
output_folder = 'common_voice/clips16'
desired_sample_rate = 16000
format = 'mp3'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.' + format):  # Assuming the files are in WAV format
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load the audio file
        try:
            audio = AudioSegment.from_file(input_path, format=format)
        except:
            continue

        # Set the desired sample rate
        audio = audio.set_frame_rate(desired_sample_rate)
        if audio.channels == 2:
            audio = audio.set_channels(1)
        # Export the converted audio to the output folder
        audio.export(output_path, format=format)

print("Conversion complete.")
