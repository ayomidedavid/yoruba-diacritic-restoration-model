import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data import load_pair_files, load_splits, CharVocab, DiacriticDataset, collate_fn
from src.models.bilstm import Seq2SeqBiLSTM
from src.eval import evaluate_model


def train(args):
    # load splits if available (train/dev/test), otherwise fall back to legacy pair files
    try:
        splits = load_splits(args.data_dir)
        train_src, train_tgt = splits['train']
    except Exception:
        train_src, train_tgt = load_pair_files(args.data_dir)
    vocab = CharVocab()
    vocab.add_texts(train_src + train_tgt)
    ds = DiacriticDataset(train_src, train_tgt, vocab)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                    num_workers=args.num_workers, pin_memory=False,
                    persistent_workers=(args.num_workers>0))
    # optional dev set
    dev_dl = None
    try:
        if 'dev' in locals() or 'splits' in locals():
            splits = load_splits(args.data_dir)
            if 'dev' in splits:
                dev_src, dev_tgt = splits['dev']
                dev_ds = DiacriticDataset(dev_src, dev_tgt, vocab)
                dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    except Exception:
        dev_dl = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # control intra-op threads to avoid oversubscription
    torch.set_num_threads(args.num_threads)
    model = Seq2SeqBiLSTM(len(vocab), sos_index=vocab.sos_index, eos_index=vocab.eos_index).to(device)
    if args.optimizer == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_index)
    best_f1 = 0.0
    accum_steps = max(1, args.accum_steps)
    for epoch in range(args.epochs):
        model.train()
        print(f"Starting epoch {epoch+1}/{args.epochs}")
        prog = tqdm(dl, desc=f"Epoch {epoch+1}")
        step = 0
        optim.zero_grad()
        # scheduled sampling probability for this epoch
        if args.ss_anneal_epochs > 0:
            # linear annealing from ss_start -> ss_end over ss_anneal_epochs
            frac = min(epoch / float(args.ss_anneal_epochs), 1.0)
            ss_prob = args.ss_start + (args.ss_end - args.ss_start) * frac
        else:
            ss_prob = args.ss_start
        for batch in prog:
            # dataloader yields (src, dec_input, dec_target)
            if len(batch) == 3:
                src, dec_in, dec_tgt = batch
            else:
                src, dec_tgt = batch
            src = src.to(device)
            dec_tgt = dec_tgt.to(device)
            batch_size = src.size(0)
            max_len = dec_tgt.size(1)
            # iterative decoding with scheduled sampling: build inputs step-by-step
            logits_steps = []
            # start token
            cur_input = torch.full((batch_size, 1), vocab.sos_index, dtype=torch.long, device=device)
            h = None
            c = None
            for t in range(max_len):
                # forward with current input to get logits for last position
                logits_all = model(src, cur_input)
                last_logits = logits_all[:, -1, :]
                logits_steps.append(last_logits.unsqueeze(1))
                # predicted token
                pred_tok = last_logits.argmax(dim=-1)
                # ground truth token for this timestep
                true_tok = dec_tgt[:, t]
                # decide per-sample whether to use ground truth (teacher forcing)
                if ss_prob >= 1.0:
                    use_true_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                elif ss_prob <= 0.0:
                    use_true_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                else:
                    use_true_mask = torch.rand(batch_size, device=device) < ss_prob
                # choose next input token
                next_tok = torch.where(use_true_mask, true_tok, pred_tok)
                cur_input = torch.cat([cur_input, next_tok.unsqueeze(1)], dim=1)
            # stack logits steps -> [batch, seq, vocab]
            logits_full = torch.cat(logits_steps, dim=1)
            tgt_flat = dec_tgt.view(-1)
            logits_flat = logits_full.view(-1, logits_full.size(-1))
            loss = loss_fn(logits_flat, tgt_flat)
            loss = loss / accum_steps
            loss.backward()
            # gradient clipping
            if args.clip_norm and args.clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            step += 1
            if step % accum_steps == 0:
                optim.step()
                optim.zero_grad()
            prog.set_postfix(loss=loss.item()*accum_steps, ss_prob=ss_prob)
        # end epoch: evaluate on dev if available
        if dev_dl is not None and not getattr(args, 'no_eval', False):
            try:
                # allow evaluating on a small dev subset to save time
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
                # save best
                if f1 > best_f1:
                    best_f1 = f1
                    ckpt_path = os.path.join(args.output_dir, 'bilstm_best.pt')
                    torch.save({'model_state': model.state_dict(), 'vocab': vocab.idx2char}, ckpt_path)
                    print(f'Saved best model to {ckpt_path} (f1={best_f1:.4f})')
            except Exception as e:
                import traceback
                print(f"Warning: exception during dev evaluation or checkpointing at epoch {epoch+1}: {e}")
                traceback.print_exc()
                print("Continuing to next epoch despite the above error.")
    # final save (last)
    last_path = os.path.join(args.output_dir, 'bilstm_last.pt')
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
    parser.add_argument('--optimizer', choices=['adam','adamw'], default='adam')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--clip_norm', type=float, default=0.0, help='If >0, clip gradients to this norm')
    parser.add_argument('--ss_start', type=float, default=1.0, help='Scheduled sampling start prob (teacher forcing prob)')
    parser.add_argument('--ss_end', type=float, default=1.0, help='Scheduled sampling end prob')
    parser.add_argument('--ss_anneal_epochs', type=int, default=0, help='Number of epochs to anneal scheduled sampling over')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
