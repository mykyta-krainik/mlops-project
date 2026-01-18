import csv
import os
import glob
from pathlib import Path

def clean_csv_files(input_dir, output_dir):
    """
    Reads CSV files from input_dir, merges multi-line comments into single lines,
    and writes them to output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = glob.glob(str(input_path / "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} files to process.")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        target_path = output_path / filename
        
        print(f"Processing {filename}...")
        
        with open(file_path, 'r', encoding='utf-8', newline='') as infile, \
             open(target_path, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Read header
            try:
                header = next(reader)
                writer.writerow(header)
            except StopIteration:
                continue

            for row in reader:
                # Merge multi-line content in all cells (primarily targeting comment_text)
                # Replacing \r\n, \n, \r with a single space
                cleaned_row = [
                    col.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').strip() 
                    if col else col 
                    for col in row
                ]
                writer.writerow(cleaned_row)
        
        print(f"Saved cleaned file to {target_path}")

if __name__ == "__main__":
    BASE_DIR = "/home/mykyta-krainik/uni/mlops/mlops-project"
    INPUT_DIR = os.path.join(BASE_DIR, "data/batches")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data/batches_cleaned")
    
    clean_csv_files(INPUT_DIR, OUTPUT_DIR)
