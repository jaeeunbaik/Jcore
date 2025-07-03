import csv

def csv_to_scp(csv_file, scp_file):
    """
    Converts a CSV file to an SCP file with the specified format.

    Args:
        csv_file (str): Path to the input CSV file.
        scp_file (str): Path to the output SCP file.
    """
    with open(csv_file, 'r', encoding='utf-8') as infile, open(scp_file, 'w', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        for row in reader:
            if len(row) == 2:
                wav_path, transcription = row
                outfile.write(f"{wav_path}|{transcription}\n")
            else:
                print(f"Skipping row with unexpected number of columns: {row}")

if __name__ == "__main__":
    csv_file_path = '/home/nas4/user/jiwon/medical_asr/medical_db_new_olkavs_vad/olkavs_tts_results.csv'  # Replace with your CSV file path
    scp_file_path ='./순천향대test/testclean.scp' # Replace with your desired SCP file path
    csv_to_scp(csv_file_path, scp_file_path)
    print(f"Successfully converted {csv_file_path} to {scp_file_path}")