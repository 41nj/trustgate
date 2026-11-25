"""Run a model on the MMLU-Pro sample and save predictions for offline reuse.

Examples:
  python3 evaluate_model.py --list-models               # models offered by the endpoint
  python3 evaluate_model.py --questions 14              # 1 per category, quick check
  python3 evaluate_model.py                             # all 140 questions (resumable)
  python3 evaluate_model.py --publish                   # also copy to evaluations/

Every answer is appended to the local cache results/mmlu-pro-<model>-<provider>.jsonl (with the
raw model output) and reused by later runs with the same provider, model, settings and data.
A sanitized, dashboard-ready JSON file is written at the end; --publish copies only that file to
evaluations/. Endpoint URLs and keys are never written; raw outputs are never published.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from benchmark import SUBSET_SIZES, subset
from provider import RESULTS_DIR, ProviderError, cache_path, cached_records, dashboard_payload, evaluate_run, list_models, load_config, run_identity, run_name

PUBLISHED_DIR = Path(__file__).resolve().parent / 'evaluations'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', type=int, choices=sorted(SUBSET_SIZES), default=140, help='1, 5 or 10 per category')
    parser.add_argument('--model', help='model ID (default: TRUSTGATE_MODEL)')
    parser.add_argument('--list-models', action='store_true', help='list models offered by the endpoint and exit')
    parser.add_argument('--skip-model-check', action='store_true', help='do not verify the model via /models')
    parser.add_argument('--publish', action='store_true', help='copy the finished run to evaluations/')
    args = parser.parse_args(argv)
    try:
        cfg = load_config()
    except (ValueError, OSError) as e:
        sys.exit(f'Configuration error: {str(e)[:200]}')
    if not cfg.key:
        sys.exit('No API key configured. Set TRUSTGATE_API_KEY or TRUSTGATE_API_KEY_FILE (see README).')
    model = args.model or cfg.model
    dataset, items = subset(SUBSET_SIZES[args.questions])
    base = f'mmlu-pro-{run_name(model)}'
    identity = run_identity(cfg, model, dataset)
    jsonl = cache_path(cfg, model)
    try:
        if args.list_models or not args.skip_model_check:
            models = list_models(cfg)
            if args.list_models:
                print('\n'.join(models) or '(no models listed)')
                return
            if model not in models:
                sys.exit(f'Model {model!r} is not offered by the endpoint. Available: {", ".join(models) or "none"}')
        records = cached_records(jsonl, identity)
        cached = sum(q['id'] in records for q in items)
        print(f'{len(items)} questions: {cached} cached in {jsonl.name}, {len(items) - cached} new requests')
        meta, records, failures = evaluate_run(cfg, items, model, jsonl, identity)
    except (ProviderError, ValueError) as e:
        sys.exit(f'Error: {str(e)[:200]}')
    payload = dashboard_payload(meta, records, failures, dataset, len(items))
    out = RESULTS_DIR / (f'{base}.json' if args.questions == 140 else f'{base}-n{args.questions}.json')
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding='utf-8')
    c, rows = payload['counts'], payload['rows']
    acc = sum(r['correct'] for r in rows) / len(rows) if rows else 0
    print(f'{c["valid"]} valid, {c["unparseable"]} unparseable, {c["failed"]} failed of {c["questions"]} · accuracy (valid) {acc:.1%}')
    print(f'Saved {out.relative_to(RESULTS_DIR.parent)}')
    if failures:
        print('Re-run the same command to retry failed questions.')
    elif args.publish:
        # Only the sanitized JSON is published; raw outputs stay in the git-ignored cache.
        PUBLISHED_DIR.mkdir(exist_ok=True)
        shutil.copy(out, PUBLISHED_DIR / out.name)
        print(f'Published {PUBLISHED_DIR.name}/{out.name} (raw outputs remain in {jsonl.relative_to(RESULTS_DIR.parent)})')


if __name__ == '__main__':
    main()
