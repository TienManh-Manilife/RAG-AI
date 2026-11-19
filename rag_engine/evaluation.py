"""Reference-set evaluation: lexical answer metrics, retrieval & escalation metrics.

Token F1 is a reproducible proxy, NOT semantic correctness. Human feedback is separate.
"""
import re
from collections import Counter


def normalize(s):
    return ' '.join(re.findall(r'[^\W_]+', (s or '').casefold(), flags=re.UNICODE))


def answer_scores(prediction, reference):
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    exact = int(pred == ref)
    if not pred or not ref:
        return {'exact_match': exact, 'token_f1': float(exact)}
    overlap = sum((Counter(pred) & Counter(ref)).values())
    precision, recall = overlap/len(pred), overlap/len(ref)
    return {'exact_match': exact, 'token_f1': (2*precision*recall/(precision+recall) if precision+recall else 0.0)}


def bool_value(v):
    return str(v).strip().casefold() in {'1', 'yes', 'true', 'y', 'có', 'x'}


def evaluate_rows(rows: list[dict]):
    if not rows:
        return {}
    n = len(rows)
    metric = {'cases': n, 'exact_match': 0.0, 'token_f1': 0.0,
              'handoff_rate': 0.0, 'citation_rate': 0.0}
    retrieval = []
    labels = []
    for row in rows:
        score = answer_scores(row.get('prediction', ''), row.get('reference_answer', ''))
        metric['exact_match'] += score['exact_match']/n
        metric['token_f1'] += score['token_f1']/n
        metric['handoff_rate'] += int(row.get('handoff', False))/n
        metric['citation_rate'] += int(row.get('has_citation', False))/n
        if row.get('expected_source'):
            retrieval.append(int(any(str(row['expected_source']).casefold() in str(h.get('source', '')).casefold() for h in row.get('hits', []))))
        if row.get('should_handoff') is not None:
            labels.append((bool(row['handoff']), bool(row['should_handoff'])))
    metric['retrieval_recall_at_k'] = sum(retrieval)/len(retrieval) if retrieval else None
    if labels:
        tp = sum(a and b for a, b in labels)
        fp = sum(a and not b for a, b in labels)
        fn = sum(not a and b for a, b in labels)
        metric['handoff_precision'] = tp/(tp+fp) if tp+fp else None
        metric['handoff_recall'] = tp/(tp+fn) if tp+fn else None
    return metric


def suggest_evidence_threshold(rows: list[dict]):
    """Optimize triage F1 on labeled cases. Tune on train, verify on holdout!"""
    labeled = [r for r in rows if r.get('should_handoff') is not None]
    if len(labeled) < 10 or len({bool(r['should_handoff']) for r in labeled}) < 2:
        return None
    candidates = [i / 100 for i in range(20, 81, 5)]
    results = []
    for threshold in candidates:
        tp = fp = fn = 0
        for r in labeled:
            # Honor hard safety reasons independently of evidence threshold.
            hard_flags = bool(r.get('hard_handoff', False))
            pred = hard_flags or float(r.get('evidence_score', 0)) < threshold
            true = bool(r['should_handoff'])
            tp += int(pred and true)
            fp += int(pred and not true)
            fn += int(not pred and true)
        precision = tp/(tp+fp) if (tp+fp) else 0
        recall = tp/(tp+fn) if (tp+fn) else 0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
        results.append({'threshold': threshold, 'precision': precision, 'recall': recall, 'f1': f1})
    # Conservative tie-breaker: higher recall for missed escalations.
    return max(results, key=lambda x: (x['f1'], x['recall'], x['precision']))
