import os
from typing import List, Tuple, Dict
import glob
import torch
from torch.utils.data import Dataset

class CharVocab:
    def __init__(self, specials=['<pad>','<unk>','<sos>','<eos>']):
        self.idx2char = list(specials)
        self.char2idx = {c:i for i,c in enumerate(self.idx2char)}

    def add_texts(self, texts: List[str]):
        for t in texts:
            for ch in t:
                if ch not in self.char2idx:
                    self.char2idx[ch] = len(self.idx2char)
                    self.idx2char.append(ch)

    def encode(self, text: str) -> List[int]:
        return [self.char2idx.get(ch, self.char2idx['<unk>']) for ch in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.idx2char[i] if i < len(self.idx2char) else '<unk>' for i in ids)

    @property
    def pad_index(self):
        return 0

    @property
    def unk_index(self):
        return 1

    @property
    def sos_index(self):
        return 2

    @property
    def eos_index(self):
        return 3

    def __len__(self):
        return len(self.idx2char)

class DiacriticDataset(Dataset):
    def __init__(self, undiacritic_texts: List[str], diacritic_texts: List[str], vocab: CharVocab, max_len: int=128):
        if len(undiacritic_texts) != len(diacritic_texts):
            # Trim to the shortest list and warn the user rather than raising immediately.
            min_len = min(len(undiacritic_texts), len(diacritic_texts))
            print(f"Warning: source/target length mismatch ({len(undiacritic_texts)} vs {len(diacritic_texts)}). Trimming to {min_len} examples.")
            undiacritic_texts = undiacritic_texts[:min_len]
            diacritic_texts = diacritic_texts[:min_len]
        # encode raw source and target
        self.src = [vocab.encode(t)[:max_len] for t in undiacritic_texts]
        tgt_raw = [vocab.encode(t)[:max_len] for t in diacritic_texts]
        # build decoder input (sos + tgt[:-1]) and decoder target (tgt + eos)
        self.dec_input = []
        self.dec_target = []
        for t in tgt_raw:
            # truncate to max_len-1 so we can add sos/eos and keep overall <= max_len
            t_trunc = t[:max_len-1]
            dec_in = [vocab.sos_index] + t_trunc
            dec_t = t_trunc + [vocab.eos_index]
            self.dec_input.append(dec_in)
            self.dec_target.append(dec_t)
        self.vocab = vocab

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return (torch.tensor(self.src[idx], dtype=torch.long),
                torch.tensor(self.dec_input[idx], dtype=torch.long),
                torch.tensor(self.dec_target[idx], dtype=torch.long))

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    srcs, dec_ins, dec_tgts = zip(*batch)
    src_lens = [s.size(0) for s in srcs]
    dec_in_lens = [s.size(0) for s in dec_ins]
    dec_tgt_lens = [s.size(0) for s in dec_tgts]
    max_src = max(src_lens)
    max_dec_in = max(dec_in_lens)
    max_dec_tgt = max(dec_tgt_lens)
    padded_src = torch.full((len(srcs), max_src), fill_value=0, dtype=torch.long)
    padded_dec_in = torch.full((len(dec_ins), max_dec_in), fill_value=0, dtype=torch.long)
    padded_dec_tgt = torch.full((len(dec_tgts), max_dec_tgt), fill_value=0, dtype=torch.long)
    for i, s in enumerate(srcs):
        padded_src[i, :s.size(0)] = s
    for i, s in enumerate(dec_ins):
        padded_dec_in[i, :s.size(0)] = s
    for i, s in enumerate(dec_tgts):
        padded_dec_tgt[i, :s.size(0)] = s
    return padded_src, padded_dec_in, padded_dec_tgt

def load_pair_files(data_dir: str, undiacritic_fname: str='sample_undiacritic.txt', diacritic_fname: str='sample_diacritic.txt'):
    """Backward-compatible loader: load a single pair of files.

    Returns (undiacritic_lines, diacritic_lines)
    """
    ud_path = os.path.join(data_dir, undiacritic_fname)
    di_path = os.path.join(data_dir, diacritic_fname)
    with open(ud_path, 'r', encoding='utf-8') as f:
        und = [l.strip() for l in f if l.strip()]
    with open(di_path, 'r', encoding='utf-8') as f:
        dia = [l.strip() for l in f if l.strip()]
    return und, dia


def _read_txt_lines(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def load_splits(data_dir: str) -> Dict[str, Tuple[List[str], List[str]]]:
    """Auto-detect and load parallel train/dev/test splits.

    Looks for files named `<prefix>_diacritic.txt` and `<prefix>_undiacritic.txt` for prefixes
    'train', 'dev', 'test'. Returns a dict mapping available prefixes to (src_list, tgt_list).
    If none of those files are found, falls back to the legacy `sample_undiacritic.txt`/
    `sample_diacritic.txt` pair and returns a dict with key 'train'.
    """
    out = {}
    prefixes = ['train', 'dev', 'test']
    for p in prefixes:
        di_fn = os.path.join(data_dir, f'{p}_diacritic.txt')
        ud_fn = os.path.join(data_dir, f'{p}_undiacritic.txt')
        if os.path.exists(di_fn) and os.path.exists(ud_fn):
            out[p] = (_read_txt_lines(ud_fn), _read_txt_lines(di_fn))
    if out:
        return out
    # fallback to older defaults
    sample_ud = os.path.join(data_dir, 'sample_undiacritic.txt')
    sample_di = os.path.join(data_dir, 'sample_diacritic.txt')
    if os.path.exists(sample_ud) and os.path.exists(sample_di):
        return {'train': (_read_txt_lines(sample_ud), _read_txt_lines(sample_di))}
    # also try generic names often produced by extractor
    generic_pairs = glob.glob(os.path.join(data_dir, '*_undiacritic.txt'))
    if generic_pairs:
        for ud in generic_pairs:
            base = os.path.basename(ud).replace('_undiacritic.txt', '')
            di = os.path.join(data_dir, f'{base}_diacritic.txt')
            if os.path.exists(di):
                out[base] = (_read_txt_lines(ud), _read_txt_lines(di))
        if out:
            return out
    # nothing found
    raise FileNotFoundError('No parallel files found in data directory. Expected files like `train_undiacritic.txt`/`train_diacritic.txt` or `sample_undiacritic.txt`/`sample_diacritic.txt`.')
