import glob
import os
import pyedflib
from datetime import timedelta

from utils.utils import get_root


def generate_txt_from_edf(file_path):
    print(f"Processing file: {file_path}")
    
    # Read each EDF file
    with pyedflib.EdfReader(file_path) as edf_reader:
        n_channels = edf_reader.signals_in_file
        signal_labels = edf_reader.getSignalLabels()
        print(f"Channels found: {signal_labels}")
        
        for types in signal_labels:
            channel_index = signal_labels.index(types)
            data = edf_reader.readSignal(channel_index)
            sfreq = edf_reader.getSampleFrequency(channel_index)
            total_duration = timedelta(seconds=len(data) / sfreq)  # Convert total duration to timedelta

            # Define interval and marker pattern
            start_time = timedelta(hours=0)  # Start from 0:00:00
            interval = timedelta(seconds=20)  # Each interval is 20 seconds
            markers = ['EMPTY']  # Example marker pattern

            # Generate rows for the required format
            output_rows = []
            current_time = start_time
            index = 1

            while current_time < total_duration:
                marker = markers[(index - 1) % len(markers)]
                row = f"{index}\t{str(current_time)}\t{marker}"
                output_rows.append(row)
                
                # Update for the next row
                current_time += interval
                index += 1

            # Join rows with newline characters
            output_txt = "\n".join(output_rows)

            # Save the output to a .txt file with the same base name
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            output_path = file_path[:-4]+".txt"
            with open(output_path, "w") as f:
                f.write(output_txt)
                
            print(f"Saved output to {output_path}")


def generate_files_for_folder(folder_path):
# Iterate through each EDF file in the folder
    for file_path in glob.glob(os.path.join(folder_path, "*.edf")):
        generate_txt_from_edf(file_path)
