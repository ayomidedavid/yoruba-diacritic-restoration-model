import unicodedata
from typing import Tuple, List, Dict
import torch
import math


def strip_diacritics(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s)
    stripped = ''.join(ch for ch in nfkd if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize('NFC', stripped)


def ids_to_string(ids: List[int], vocab) -> str:
    # stop at pad if encountered; skip special tokens when decoding
    chars = []
    for i in ids:
        if i == vocab.pad_index:
            break
        # if index out of range guard
        if i < len(vocab.idx2char):
            ch = vocab.idx2char[i]
        else:
            ch = '<unk>'
        # skip explicit special tokens in the output
        if ch in ('<pad>', '<unk>', '<sos>', '<eos>'):
            continue
        chars.append(ch)
    return ''.join(chars)


def _char_level_counts(pred: str, target: str) -> Tuple[int, int]:
    # returns (match_count, total_chars_compared)
    n = min(len(pred), len(target))
    match = sum(1 for i in range(n) if pred[i] == target[i])
    # count remaining target chars as mismatches
    total = max(len(pred), len(target))
    return match, total


def compute_char_accuracy(pred: str, target: str) -> float:
    match, total = _char_level_counts(pred, target)
    if total == 0:
        return 1.0
    return match / total


def compute_diacritic_metrics(pred: str, target: str) -> Dict[str, float]:
    # We treat each character position as a binary decision: does the character have a diacritic?
    # True positives: target has diacritic and pred equals target (exact char match)
    # Predicted positives: positions where pred char has a diacritic (pred != stripped(pred))
    # Actual positives: positions where target char has a diacritic
    # We'll compare over the full span: handle tails by counting their diacritics appropriately.
    pred_len = len(pred)
    tgt_len = len(target)
    n = max(pred_len, tgt_len)
    tp = 0
    pred_pos = 0
    actual_pos = 0
    for i in range(n):
        p = pred[i] if i < pred_len else ''
        t = target[i] if i < tgt_len else ''
        if t and t != strip_diacritics(t):
            actual_pos += 1
            if p == t:
                tp += 1
        if p and p != strip_diacritics(p):
            pred_pos += 1
    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / actual_pos if actual_pos > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'pred_pos': pred_pos, 'actual_pos': actual_pos}


def _levenshtein(a: List[str], b: List[str]) -> int:
    # classic dynamic programming distance
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def wer(pred: str, target: str) -> float:
    # Word Error Rate = edit distance(w_pred, w_ref) / number of words in reference
    w_pred = pred.split()
    w_ref = target.split()
    if len(w_ref) == 0:
        return 0.0 if len(w_pred) == 0 else 1.0
    dist = _levenshtein(w_pred, w_ref)
    return dist / len(w_ref)


def evaluate_model(model, dataloader, vocab, device=None) -> Dict[str, float]:
    """Evaluate model over a dataloader. Returns aggregated metrics.

    Expects dataloader to yield (src_tensor, tgt_tensor) where tensors are LongTensor of ids.
    The function will compute cross-entropy (using model(src, tgt) logits), greedy predictions
    (by argmax over logits) and then compute char-accuracy, diacritic precision/recall/F1, WER.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_index, reduction='sum')
    total_loss = 0.0
    total_tokens = 0
    total_char_matches = 0
    total_chars = 0
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    n_batches = 0
    sum_wer = 0.0
    n_examples = 0
    total_tp = 0
    total_pred_pos = 0
    total_actual_pos = 0
    with torch.no_grad():
        for batch in dataloader:
            # expect dataloader to yield (src, dec_input, dec_target)
            if len(batch) == 3:
                src, dec_in, dec_tgt = batch
            else:
                # backward-compatibility: if loader yields (src,tgt)
                src, dec_tgt = batch
                # build dec_in by prepending sos
                dec_in = torch.full((dec_tgt.size(0), 1), fill_value=vocab.sos_index, dtype=torch.long)
                dec_in = torch.cat([dec_in, dec_tgt[:, :-1]], dim=1)
            src = src.to(device)
            dec_in = dec_in.to(device)
            dec_tgt = dec_tgt.to(device)
            # get logits by passing dec_in (teacher-forcing)
            logits = model(src, dec_in)
            # logits shape: [batch, seq, vocab]
            bsz, seq_len, vocab_sz = logits.size()
            logits_flat = logits.view(-1, vocab_sz)
            tgt_flat = dec_tgt.view(-1)
            loss = loss_fn(logits_flat, tgt_flat)
            total_loss += loss.item()
            total_tokens += (tgt_flat != vocab.pad_index).sum().item()
            # Predictions: greedy generation from model
            pred_ids = model(src).cpu().tolist()
            tgt_ids = dec_tgt.cpu().tolist()
            for p_ids, t_ids in zip(pred_ids, tgt_ids):
                p_str = ids_to_string(p_ids, vocab)
                t_str = ids_to_string(t_ids, vocab)
                cmatch, ctotal = _char_level_counts(p_str, t_str)
                total_char_matches += cmatch
                total_chars += ctotal
                dm = compute_diacritic_metrics(p_str, t_str)
                sum_precision += dm['precision']
                sum_recall += dm['recall']
                sum_f1 += dm['f1']
                total_tp += dm['tp']
                total_pred_pos += dm['pred_pos']
                total_actual_pos += dm['actual_pos']
                sum_wer += wer(p_str, t_str)
                n_examples += 1
            n_batches += 1
    avg_xent = total_loss / total_tokens if total_tokens > 0 else 0.0
    char_acc = total_char_matches / total_chars if total_chars > 0 else 0.0
    # aggregate diacritic precision/recall/f1 from totals (micro)
    precision = total_tp / total_pred_pos if total_pred_pos > 0 else 0.0
    recall = total_tp / total_actual_pos if total_actual_pos > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_precision_per_example = sum_precision / n_examples if n_examples > 0 else 0.0
    avg_recall_per_example = sum_recall / n_examples if n_examples > 0 else 0.0
    avg_f1_per_example = sum_f1 / n_examples if n_examples > 0 else 0.0
    avg_wer = sum_wer / n_examples if n_examples > 0 else 0.0
    return {
        'cross_entropy_per_token': avg_xent,
        'char_accuracy': char_acc,
        'diacritic_precision_micro': precision,
        'diacritic_recall_micro': recall,
        'diacritic_f1_micro': f1,
        'diacritic_precision_macro_per_example': avg_precision_per_example,
        'diacritic_recall_macro_per_example': avg_recall_per_example,
        'diacritic_f1_macro_per_example': avg_f1_per_example,
        'wer': avg_wer,
        'examples_evaluated': n_examples
    }
