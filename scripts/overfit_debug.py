import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import load_splits, CharVocab, DiacriticDataset, collate_fn
from src.models.bilstm import Seq2SeqBiLSTM


def ids_to_string(ids, vocab):
    s = []
    for i in ids:
        if i == vocab.eos_index:
            break
        if i == vocab.pad_index:
            continue
        s.append(vocab.idx2char[i] if i < len(vocab.idx2char) else '<unk>')
    return ''.join(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--n_examples', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--print_n', type=int, default=5)
    parser.add_argument('--save_model', default=None)
    args = parser.parse_args()

    try:
        splits = load_splits(args.data_dir)
        train_src, train_tgt = splits['train']
    except Exception as e:
        print('Error loading splits:', e)
        return

    n = min(args.n_examples, len(train_src))
    src_small = train_src[:n]
    tgt_small = train_tgt[:n]

    vocab = CharVocab()
    vocab.add_texts(src_small + tgt_small)

    ds = DiacriticDataset(src_small, tgt_small, vocab)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device(args.device)
    model = Seq2SeqBiLSTM(len(vocab), hidden_dim=args.hidden, sos_index=vocab.sos_index, eos_index=vocab.eos_index).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_index)

    print(f'Overfitting on {n} examples, {len(dl)} batches per epoch, epochs={args.epochs}, device={device}')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_tokens = 0
        for batch in dl:
            src, dec_in, dec_tgt = batch
            src = src.to(device)
            dec_in = dec_in.to(device)
            dec_tgt = dec_tgt.to(device)
            optim.zero_grad()
            logits = model(src, dec_in)
            logits_flat = logits.view(-1, logits.size(-1))
            tgt_flat = dec_tgt.view(-1)
            loss = loss_fn(logits_flat, tgt_flat)
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_tokens += (tgt_flat != vocab.pad_index).sum().item()
        avg_loss = total_loss / len(dl)
        print(f'Epoch {epoch:03d} loss={avg_loss:.4f}')

        # print sample predictions on the whole small set
        model.eval()
        with torch.no_grad():
            # prepare full-batch for generation and targets
            all_src, all_dec_in, all_dec_tgt = next(iter(DataLoader(ds, batch_size=n, collate_fn=collate_fn)))
            all_src = all_src.to(device)
            preds = model(all_src)
            # preds: [n, seq_len]
            print('--- Samples ---')
            for i in range(min(args.print_n, n)):
                src_ids = all_src[i].cpu().tolist()
                tgt_ids = all_dec_tgt[i].cpu().tolist()
                pred_ids = preds[i].cpu().tolist()
                src_s = ''.join([vocab.idx2char[j] for j in src_ids if j != vocab.pad_index])
                tgt_s = ids_to_string(tgt_ids, vocab)
                pred_s = ids_to_string(pred_ids, vocab)
                print(f'[{i}] SRC: {src_s}')
                print(f'    TGT: {tgt_s}')
                print(f'    PRED:{pred_s}')
            print('---------------')

    if args.save_model:
        torch.save({'model_state': model.state_dict(), 'vocab': vocab.idx2char}, args.save_model)
        print('Saved model to', args.save_model)


if __name__ == '__main__':
    main()
