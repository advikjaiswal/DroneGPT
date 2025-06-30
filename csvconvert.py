import os
from pyulog.ulog2csv import convert_ulog2csv

ulog_folder = "drone_data"

for filename in os.listdir(ulog_folder):
    if filename.endswith(".ulg"):
        ulog_path = os.path.join(ulog_folder, filename)
        output_folder = ulog_folder  # Output to same folder

        print(f"🔄 Processing {filename}...")

        try:
            convert_ulog2csv(
                ulog_file_name=ulog_path,
                messages=None,                # All messages
                output=output_folder,         # Output directory
                delimiter=',',                # Standard CSV
                time_s=None,                  # Start time (None = beginning)
                time_e=None                   # End time (None = full log)
            )
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")
