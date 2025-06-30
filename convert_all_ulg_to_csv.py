import os
from pyulog.ulog2csv import convert_ulog2csv

ulog_folder = "drone_data"

for filename in os.listdir(ulog_folder):
    if filename.endswith(".ulg"):
        ulog_path = os.path.join(ulog_folder, filename)
        output_dir = ulog_folder  # save CSVs in same folder

        print(f"🔄 Processing {filename}...")

        try:
            # Use correct function signature with all 6 required args
            convert_ulog2csv(
                ulog_file_name=ulog_path,
                messages=None,                # None = export all topics
                output=output_dir,            # Where to save .csvs
                delimiter=",",                # Standard CSV delimiter
                time_s=None,                  # Start from beginning
                time_e=None                   # End at last log time
            )
            print(f"✅ Successfully converted {filename}")
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")
