import argparse
from pathlib import Path


def check_alignment(src_path: Path, tgt_path: Path, show_examples: int = 5, trim: bool = False):
    src_lines = [l.rstrip('\n') for l in src_path.open(encoding='utf-8')]
    tgt_lines = [l.rstrip('\n') for l in tgt_path.open(encoding='utf-8')]
    len_src = len(src_lines)
    len_tgt = len(tgt_lines)
    print(f'source: {src_path} -> {len_src} lines')
    print(f'target: {tgt_path} -> {len_tgt} lines')
    if len_src != len_tgt:
        print('Lengths differ by', abs(len_src - len_tgt))
    else:
        print('Lengths match.')

    # show first N example pairs and first mismatches
    n = min(show_examples, min(len_src, len_tgt))
    if n > 0:
        print('\nFirst %d aligned examples:' % n)
        for i in range(n):
            print(f'[{i}] SRC: {src_lines[i]}')
            print(f'    TGT: {tgt_lines[i]}')

    # find first mismatches
    mismatches = []
    limit = min(len_src, len_tgt)
    for i in range(limit):
        if src_lines[i] != tgt_lines[i]:
            mismatches.append(i)
            if len(mismatches) >= show_examples:
                break
    if mismatches:
        print(f'\nFirst {len(mismatches)} mismatched indices (src != tgt):')
        for i in mismatches:
            print(f'[{i}] SRC: {src_lines[i]}')
            print(f'    TGT: {tgt_lines[i]}')
    else:
        print('\nNo mismatches in the first %d positions.' % limit)

    if trim and len_src != len_tgt:
        m = min(len_src, len_tgt)
        new_src = src_lines[:m]
        new_tgt = tgt_lines[:m]
        src_path_backup = src_path.with_suffix(src_path.suffix + '.bak')
        tgt_path_backup = tgt_path.with_suffix(tgt_path.suffix + '.bak')
        src_path.rename(src_path_backup)
        tgt_path.rename(tgt_path_backup)
        with src_path.open('w', encoding='utf-8') as fs:
            for l in new_src:
                fs.write(l + '\n')
        with tgt_path.open('w', encoding='utf-8') as ft:
            for l in new_tgt:
                ft.write(l + '\n')
        print(f'Files trimmed to {m} lines. Backups: {src_path_backup}, {tgt_path_backup}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('src')
    p.add_argument('tgt')
    p.add_argument('--show', type=int, default=5, help='How many examples/mismatches to show')
    p.add_argument('--trim', action='store_true', help='Trim both files to the shortest length (backups are created)')
    args = p.parse_args()
    check_alignment(Path(args.src), Path(args.tgt), show_examples=args.show, trim=args.trim)
