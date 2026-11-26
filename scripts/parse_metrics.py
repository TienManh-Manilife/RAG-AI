"""
python scripts/parse_metrics.py
"""

"""Summarize RAG-AI Resume Patch reports without Docker or third-party libraries."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
import math


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def flatten(data, prefix=''):
    for key, value in data.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            yield from flatten(value, name + '.')
        else:
            yield name, value


def metrics(summary, cases, report):
    out = dict(flatten(summary))
    times = sorted(r['latency_s'] for r in cases if r.get('latency_s') is not None)
    out['median_latency_s'] = median(times) if times else None
    out['p95_latency_s'] = times[math.ceil(.95 * len(times)) - 1] if times else None
    out['sum_case_latency_s'] = round(sum(times), 3) if times else None
    if 'confusion' in summary:
        c = summary['confusion']
        tp, fp, fn, tn = (c[k] for k in ('tp', 'fp', 'fn', 'tn'))
        out['false_answer_rate_on_insufficient'] = fn / (tp + fn) if tp + fn else None
        out['false_handoff_rate_on_answerable'] = fp / (fp + tn) if fp + tn else None
    return out


def fmt(key, value):
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if key.endswith('_s'):
            return f'{value:.3f}'
        if key == 'mrr_at_5':
            return f'{value:.4f}'
        if key.startswith('retrieval_recall_') or 'rate' in key or key.endswith('_proxy') or key in {'handoff_precision','handoff_recall','handoff_specificity','handoff_f1'}:
            return f'{value * 100:.2f}%'
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='?', type=Path, default=Path('results'))
    parser.add_argument('--include-partial', action='store_true')
    args = parser.parse_args()
    root = args.results
    reports = []
    # Ignore the root-level latest report, which duplicates the most recent run.
    for path in sorted(root.glob('*/benchmark_latest.json')):
        try:
            report = read_json(path)
            cases = report['cases']
            ids = [r['id'] for r in cases]
            done, total = report['completed_cases'], report['total_cases']
            if len(ids) != len(set(ids)) or done != len(ids) or done > total:
                raise ValueError('Invalid case count or duplicate IDs')
            checkpoint = path.with_name('checkpoint.txt')
            if checkpoint.exists():
                state = read_json(checkpoint)
                if (set(state['completed']) != set(ids) or len(state['case_ids']) != total
                    or state['identity'] != report['identity']
                    or any(state['completed'][r['id']] != r for r in cases)
                    or state['failure_count'] != report.get('failed_attempts')):
                    raise ValueError('Report stale: rerun the same benchmark command to regenerate it')
            complete = report['complete'] is True and done == total
            if not complete and not args.include_partial:
                print(f'SKIP {path.parent.name}: incomplete {done}/{total}')
                continue
            if any(r.get('error') or 'system_error' in r.get('errors', []) for r in cases):
                raise ValueError('Unexpected system error in completed cases')
            reports.append((path.parent.name, report, cases, complete))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f'SKIP {path}: {exc}')
    if not reports:
        raise SystemExit('No usable reports found. Check the results folder and benchmark status.')
    rows = []
    for name, report, cases, complete in reports:
        for scope, summary in [('all', report['summary']), *report.get('by_role', {}).items()]:
            selected = cases if scope == 'all' else [r for r in cases if r['actor_role'] == scope]
            row = {'run': name, 'scope': scope, 'complete': complete,
                   'run_completed_cases': report['completed_cases'], 'run_total_cases': report['total_cases'],
                   'failed_attempts_run': report.get('failed_attempts'), 'mode': report['mode'],
                   'method': 'hybrid' if report['dense_requested'] else 'lexical',
                   'dataset_sha256': report.get('identity', {}).get('dataset_sha256'),
                   'corpus_sha256': report.get('identity', {}).get('corpus_sha256'),
                   'source_sha256': report.get('identity', {}).get('source_sha256'),
                   'min_relevance': report.get('min_relevance'),
                   **metrics(summary, selected, report)}
            rows.append(row)
    cols = list(dict.fromkeys(k for row in rows for k in row))
    output = root / 'metrics_summary.csv'
    with output.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row['scope'] != 'all':
            continue
        print(f"\n=== {row['run']} | {row['run_completed_cases']}/{row['run_total_cases']} | {'COMPLETE' if row['complete'] else 'PARTIAL'} ===")
        for key, value in row.items():
            if key in {'run','scope','complete','mode','method','run_completed_cases','run_total_cases','dataset_sha256','corpus_sha256','source_sha256'}:
                continue
            print(f'{key:42} {fmt(key, value)}')
    print(f'\nSaved: {output.resolve()} (raw ratios 0–1; scopes all/user/staff)')


if __name__ == '__main__':
    main()
