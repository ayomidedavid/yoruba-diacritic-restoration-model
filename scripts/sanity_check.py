import sys
import os
import traceback
import torch

# make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_splits, CharVocab, DiacriticDataset, collate_fn
from torch.utils.data import DataLoader
from src.models.bilstm import Seq2SeqBiLSTM
from src.models.transformer import Seq2SeqTransformer
from src.models.hybrid import HybridModel


def main():
    try:
        splits = load_splits('data')
        train_src, train_tgt = splits['train']
    except Exception as e:
        print('Could not load splits from data/:', e)
        return

    # take a tiny subset
    n = min(8, len(train_src))
    src_sub = train_src[:n]
    tgt_sub = train_tgt[:n]

    vocab = CharVocab()
    vocab.add_texts(src_sub + tgt_sub)
    ds = DiacriticDataset(src_sub, tgt_sub, vocab)
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

    batch = next(iter(dl))
    if len(batch) == 3:
        src, dec_in, dec_tgt = batch
    else:
        print('Unexpected batch format from collate_fn')
        return

    print('Batch shapes: src', src.shape, 'dec_in', dec_in.shape, 'dec_tgt', dec_tgt.shape)

    device = torch.device('cpu')

    # BiLSTM
    bilstm = Seq2SeqBiLSTM(len(vocab), sos_index=vocab.sos_index, eos_index=vocab.eos_index).to(device)
    bilstm.eval()
    with torch.no_grad():
        logits = bilstm(src, dec_in)
        print('BiLSTM logits shape (teacher):', logits.shape)
        gen = bilstm(src)
        print('BiLSTM generated shape:', gen.shape)

    # Transformer
    transformer = Seq2SeqTransformer(len(vocab), sos_index=vocab.sos_index, eos_index=vocab.eos_index).to(device)
    transformer.eval()
    with torch.no_grad():
        logits = transformer(src, dec_in)
        print('Transformer logits shape (teacher):', logits.shape)
        gen = transformer(src)
        print('Transformer generated shape:', gen.shape)

    # Hybrid
    hybrid = HybridModel(len(vocab), sos_index=vocab.sos_index, eos_index=vocab.eos_index).to(device)
    hybrid.eval()
    with torch.no_grad():
        logits = hybrid(src, dec_in)
        print('Hybrid logits shape (teacher):', logits.shape)
        gen = hybrid(src)
        print('Hybrid generated shape:', gen.shape)

    print('\nSample decoded (first example):')
    print('src:', ''.join([vocab.idx2char[i] for i in src[0].tolist() if i != vocab.pad_index]))
    print('tgt:', ''.join([vocab.idx2char[i] for i in dec_tgt[0].tolist() if i != vocab.pad_index]))
    print('gen_bilstm ids:', gen[0].tolist() if isinstance(gen, torch.Tensor) else gen)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
