import argparse
import torch
from src.data import load_splits, CharVocab, DiacriticDataset, collate_fn
from src.models.bilstm import Seq2SeqBiLSTM
from src.eval import compute_diacritic_metrics
from torch.utils.data import DataLoader


def build_vocab_from_list(idx2char_list):
    vocab = CharVocab()
    vocab.idx2char = list(idx2char_list)
    vocab.char2idx = {c: i for i, c in enumerate(vocab.idx2char)}
    return vocab


def ids_to_string(ids, vocab):
    chars = []
    for i in ids:
        if i == vocab.pad_index:
            break
        if i < len(vocab.idx2char):
            ch = vocab.idx2char[i]
        else:
            ch = '<unk>'
        if ch in ('<pad>', '<unk>'):
            continue
        chars.append(ch)
    return ''.join(chars)


def inspect(model_path, data_dir, split='dev', n=10, device='cpu'):
    ckpt = torch.load(model_path, map_location=device)
    idx2char = ckpt.get('vocab')
    if idx2char is None:
        raise RuntimeError('No vocab found in checkpoint')
    vocab = build_vocab_from_list(idx2char)
    # build model with defaults
    model = Seq2SeqBiLSTM(len(vocab)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    splits = load_splits(data_dir)
    if split not in splits:
        raise RuntimeError(f"Split '{split}' not found in data_dir")
    src_lines, tgt_lines = splits[split]
    ds = DiacriticDataset(src_lines, tgt_lines, vocab)
    dl = DataLoader(ds, batch_size=n, shuffle=False, collate_fn=collate_fn)
    src_batch, tgt_batch = next(iter(dl))
    with torch.no_grad():
        src_batch = src_batch.to(device)
        # greedy decode path in model when tgt is None
        pred_ids = model(src_batch)
    pred_ids = pred_ids.cpu().tolist()
    tgt_ids = tgt_batch.tolist()
    src_ids = src_batch.cpu().tolist()
    for i in range(len(pred_ids)):
        p = ids_to_string(pred_ids[i], vocab)
        t = ids_to_string(tgt_ids[i], vocab)
        s = ids_to_string(src_ids[i], vocab)
        dm = compute_diacritic_metrics(p, t)
        print('--- Example', i)
        print('SRC:', s)
        print('TGT:', t)
        print('PRED:', p)
        print('Diacritic P/R/F1:', round(dm['precision'],3), round(dm['recall'],3), round(dm['f1'],3))
        print()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', '-m', default='models/bilstm.pt')
    p.add_argument('--data_dir', '-d', default='data')
    p.add_argument('--split', default='dev')
    p.add_argument('-n', type=int, default=10)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    inspect(args.model, args.data_dir, split=args.split, n=args.n, device=args.device)
