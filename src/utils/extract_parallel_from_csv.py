import csv
import unicodedata
from pathlib import Path
import argparse


def strip_diacritics(s: str) -> str:
    nfkd = unicodedata.normalize('NFD', s)
    stripped = ''.join(ch for ch in nfkd if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize('NFC', stripped)


def extract(csv_path, target_column, out_dir, source_column=None, delimiter=',', lowercase=True, strip_source=True, max_lines=None, prefix=None):
    """Extract parallel files from a CSV.

    Parameters
    - csv_path: path to csv file
    - target_column: name of column with diacritized (target) text
    - out_dir: directory to write outputs
    - source_column: optional name of column to use as source (if provided, that column will be used verbatim)
    - strip_source: if True and source_column is None, source will be produced by stripping diacritics from target
    - prefix: optional prefix for output files (e.g., 'train' or 'test'); otherwise csv basename is used
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = prefix if prefix else csv_path.stem
    tgt_path = out_dir / f'{base}_diacritic.txt'
    src_path = out_dir / f'{base}_undiacritic.txt'
    seen = 0
    with open(csv_path, 'r', encoding='utf-8') as f, open(tgt_path, 'w', encoding='utf-8') as ft, open(src_path, 'w', encoding='utf-8') as fs:
        reader = csv.DictReader(f, delimiter=delimiter)
        if target_column not in reader.fieldnames:
            raise ValueError(f"Target column '{target_column}' not found in CSV. Available columns: {reader.fieldnames}")
        if source_column and source_column not in reader.fieldnames:
            raise ValueError(f"Source column '{source_column}' not found in CSV. Available columns: {reader.fieldnames}")
        for row in reader:
            tgt = row[target_column].strip()
            if not tgt:
                continue
            if lowercase:
                tgt = tgt.lower()
            ft.write(tgt + '\n')
            if source_column:
                src = row[source_column].strip()
                if lowercase:
                    src = src.lower()
                fs.write(src + '\n')
            else:
                if strip_source:
                    fs.write(strip_diacritics(tgt) + '\n')
                else:
                    fs.write(tgt + '\n')
            seen += 1
            if max_lines and seen >= max_lines:
                break
    print(f'Wrote {seen} lines to {tgt_path} and {src_path}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('csv_path')
    p.add_argument('--target-column', '-t', required=True, help='Name of the CSV column containing diacritized text (target)')
    p.add_argument('--source-column', '-s', default=None, help='Name of the CSV column to use as source (undiacritic). If omitted, source is created by stripping diacritics from target')
    p.add_argument('--out-dir', '-o', default='data', help='Output directory for parallel files')
    p.add_argument('--prefix', '-p', default=None, help='Prefix for output files (defaults to csv basename)')
    p.add_argument('--delimiter', default=',')
    p.add_argument('--no-lowercase', dest='lowercase', action='store_false')
    p.add_argument('--no-strip', dest='strip_source', action='store_false', help='When source-column is not provided, do not strip diacritics (i.e., copy target into source)')
    p.add_argument('--max-lines', type=int, default=None)
    args = p.parse_args()
    extract(args.csv_path, args.target_column, args.out_dir, source_column=args.source_column, delimiter=args.delimiter, lowercase=args.lowercase, strip_source=args.strip_source, max_lines=args.max_lines, prefix=args.prefix)
