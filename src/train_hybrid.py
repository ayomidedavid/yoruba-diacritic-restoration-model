import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import load_pair_files, load_splits, CharVocab, DiacriticDataset, collate_fn
from src.models.hybrid import HybridModel
from src.eval import evaluate_model


def train(args):
    try:
        splits = load_splits(args.data_dir)
        train_src, train_tgt = splits['train']
    except Exception:
        train_src, train_tgt = load_pair_files(args.data_dir)
    vocab = CharVocab()
    vocab.add_texts(train_src+train_tgt)
    ds = DiacriticDataset(train_src, train_tgt, vocab)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                    num_workers=args.num_workers, pin_memory=False,
                    persistent_workers=(args.num_workers>0))
    dev_dl = None
    try:
        splits = load_splits(args.data_dir)
        if 'dev' in splits:
            dev_src, dev_tgt = splits['dev']
            dev_ds = DiacriticDataset(dev_src, dev_tgt, vocab)
            dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    except Exception:
        dev_dl = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(args.num_threads)
    model = HybridModel(len(vocab), sos_index=vocab.sos_index, eos_index=vocab.eos_index).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_index)
    best_f1 = 0.0
    accum_steps = max(1, args.accum_steps)
    for epoch in range(args.epochs):
        model.train()
        prog = tqdm(dl, desc=f"Epoch {epoch+1}")
        step = 0
        optim.zero_grad()
        for batch in prog:
            if len(batch) == 3:
                src, dec_in, dec_tgt = batch
            else:
                src, dec_tgt = batch
                dec_in = torch.full((dec_tgt.size(0), 1), fill_value=vocab.sos_index, dtype=torch.long)
                dec_in = torch.cat([dec_in, dec_tgt[:, :-1]], dim=1)
            src = src.to(device)
            dec_in = dec_in.to(device)
            dec_tgt = dec_tgt.to(device)
            logits = model(src, dec_in)
            logits_flat = logits.view(-1, logits.size(-1))
            tgt_flat = dec_tgt.view(-1)
            loss = loss_fn(logits_flat, tgt_flat)
            loss = loss / accum_steps
            loss.backward()
            step += 1
            if step % accum_steps == 0:
                optim.step()
                optim.zero_grad()
            prog.set_postfix(loss=loss.item()*accum_steps)
        if dev_dl is not None and not getattr(args, 'no_eval', False):
            try:
                if getattr(args, 'eval_subset', 0) and args.eval_subset > 0:
                    subset_n = min(args.eval_subset, len(dev_src))
                    dev_sub_src = dev_src[:subset_n]
                    dev_sub_tgt = dev_tgt[:subset_n]
                    dev_sub_ds = DiacriticDataset(dev_sub_src, dev_sub_tgt, vocab)
                    from torch.utils.data import DataLoader as _DL
                    dev_sub_dl = _DL(dev_sub_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
                    metrics = evaluate_model(model, dev_sub_dl, vocab)
                else:
                    metrics = evaluate_model(model, dev_dl, vocab)
                print(f'Epoch {epoch+1} Dev metrics:', metrics)
                f1 = metrics.get('diacritic_f1_micro', 0.0)
                if f1 > best_f1:
                    best_f1 = f1
                    ckpt_path = os.path.join(args.output_dir, 'hybrid_best.pt')
                    torch.save({'model_state': model.state_dict(), 'vocab': vocab.idx2char}, ckpt_path)
                    print(f'Saved best model to {ckpt_path} (f1={best_f1:.4f})')
            except Exception as e:
                import traceback
                print(f"Warning: exception during dev evaluation or checkpointing at epoch {epoch+1}: {e}")
                traceback.print_exc()
                print("Continuing to next epoch despite the above error.")
    last_path = os.path.join(args.output_dir, 'hybrid_last.pt')
    torch.save({'model_state': model.state_dict(), 'vocab': vocab.idx2char}, last_path)
    print(f'Saved final model to {last_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_dir', default='models')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--accum_steps', type=int, default=1)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--no_eval', action='store_true', help='Skip dev evaluation between epochs')
    parser.add_argument('--eval_subset', type=int, default=0, help='If >0, evaluate on first N dev examples (faster)')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
