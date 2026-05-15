import os
import sys
import argparse
import torch

# make repo importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import load_splits, CharVocab, DiacriticDataset, collate_fn
from src.eval import evaluate_model, ids_to_string
from torch.utils.data import DataLoader
from src.models.bilstm import Seq2SeqBiLSTM
from src.models.transformer import Seq2SeqTransformer
from src.models.hybrid import HybridModel


def build_vocab_from_idx2char(idx2char):
    v = CharVocab()
    v.idx2char = list(idx2char)
    v.char2idx = {c:i for i,c in enumerate(v.idx2char)}
    return v


def load_checkpoint(path, model_type, device):
    ckpt = torch.load(path, map_location=device)
    idx2char = ckpt.get('vocab')
    if idx2char is None:
        raise ValueError('Checkpoint does not contain `vocab`')
    vocab = build_vocab_from_idx2char(idx2char)
    if model_type == 'bilstm':
        model = Seq2SeqBiLSTM(len(vocab), sos_index=vocab.sos_index, eos_index=vocab.eos_index)
    elif model_type == 'transformer':
        model = Seq2SeqTransformer(len(vocab), sos_index=vocab.sos_index, eos_index=vocab.eos_index)
    elif model_type == 'hybrid':
        model = HybridModel(len(vocab), sos_index=vocab.sos_index, eos_index=vocab.eos_index)
    else:
        raise ValueError('Unknown model_type')
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model, vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--model_type', choices=['bilstm','transformer','hybrid'], default='bilstm')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_subset', type=int, default=0, help='If >0 evaluate on first N dev examples')
    parser.add_argument('--num_examples', type=int, default=10, help='Number of sample predictions to print')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_preds', action='store_true', help='Save predictions to CSV')
    parser.add_argument('--preds_path', default=None, help='Path to write predictions CSV when --save_preds is set')
    args = parser.parse_args()

    device = torch.device(args.device)
    try:
        splits = load_splits(args.data_dir)
        if 'dev' not in splits:
            print('No dev split found in', args.data_dir)
            return
        dev_src, dev_tgt = splits['dev']
    except Exception as e:
        print('Error loading splits:', e)
        return

    # load model and vocab
    if args.model_path and os.path.exists(args.model_path):
        model, vocab = load_checkpoint(args.model_path, args.model_type, device)
    else:
        # try defaults
        for candidate in ['models/bilstm_best.pt','models/bilstm_last.pt','models/bilstm.pt']:
            if os.path.exists(candidate):
                model, vocab = load_checkpoint(candidate, args.model_type, device)
                break
        else:
            print('No checkpoint found. Aborting.')
            return

    # build dev dataset (optionally subset)
    if args.eval_subset and args.eval_subset > 0:
        dev_src = dev_src[:args.eval_subset]
        dev_tgt = dev_tgt[:args.eval_subset]
    dev_ds = DiacriticDataset(dev_src, dev_tgt, vocab)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # run evaluation
    metrics = evaluate_model(model, dev_dl, vocab, device=device)
    print('Dev metrics:')
    for k,v in metrics.items():
        print(f'  {k}: {v}')

    # print sample predictions and optionally save CSV
    import csv
    print('\nSample predictions:')
    preds_out = []
    with torch.no_grad():
        for i, batch in enumerate(dev_dl):
            src, dec_in, dec_tgt = batch
            src = src.to(device)
            dec_tgt = dec_tgt.to(device)
            preds = model(src)
            for j in range(src.size(0)):
                idx = i*args.batch_size + j
                p_ids = preds[j].cpu().tolist()
                t_ids = dec_tgt[j].cpu().tolist()
                s_ids = src[j].cpu().tolist()
                s_str = ''.join([vocab.idx2char[id] for id in s_ids if id != vocab.pad_index])
                t_str = ids_to_string(t_ids, vocab)
                p_str = ids_to_string(p_ids, vocab)
                if idx < args.num_examples:
                    print(f'[{idx}] SRC: {s_str}')
                    print(f'    TGT: {t_str}')
                    print(f'    PRED: {p_str}\n')
                preds_out.append({'idx': idx, 'src': s_str, 'tgt': t_str, 'pred': p_str})
    if args.save_preds:
        out_path = args.preds_path or os.path.join('models', f'{args.model_type}_preds.csv')
        print('Saving predictions to', out_path)
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['idx','src','tgt','pred'])
            writer.writeheader()
            for r in preds_out:
                writer.writerow(r)

if __name__ == '__main__':
    main()
