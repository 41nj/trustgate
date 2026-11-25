"""Download MMLU-Pro (test split) and write the fixed TrustGate sample to data/mmlu_pro_sample.json.

Sampling: for each category (alphabetical), the category's questions are sorted by question_id
and 10 are drawn with one shared random.Random(42) via rng.sample. Within each category the
first 5 drawn are the 'calibration' half and the last 5 the 'test' half (stratified 70/70 split).
Model performance plays no role in sampling. Ground-truth explanations (cot_content) are dropped.

Revision handling: the datasets-server JSON API (standard library only, no pyarrow) always serves
the dataset's *current* revision; it cannot download a specific one. The script therefore checks
that the current revision equals REVISION before and after the download and aborts otherwise.
It verifies the revision; it cannot fetch an older one. If the dataset has changed, the committed
data/mmlu_pro_sample.json remains the reference; regenerating from a newer revision requires
--accept-revision and changes the cache identity of all runs.

  python3 fetch_mmlu_pro.py
"""
import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATASET = 'TIGER-Lab/MMLU-Pro'
REVISION = 'b189ec765aa7ed75c8acfea42df31fdae71f97be'
SEED, PER_CATEGORY, CAL_PER_CATEGORY = 42, 10, 5
OUT = Path(__file__).resolve().parent / 'data' / 'mmlu_pro_sample.json'


def get(url):
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'trustgate-fetch'}), timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 7:
                raise
            retry = getattr(e, 'headers', None) and e.headers.get('Retry-After')
            time.sleep(int(retry) if retry and retry.isdigit() else 10 * (attempt + 1))


def revision():
    return get(f'https://huggingface.co/api/datasets/{DATASET}')['sha']


def fetch_test_rows():
    q = urllib.parse.quote(DATASET, safe='')
    total = get(f'https://datasets-server.huggingface.co/info?dataset={q}')['dataset_info']['default']['splits']['test']['num_examples']
    rows = []
    for offset in range(0, total, 100):
        page = get(f'https://datasets-server.huggingface.co/rows?dataset={q}&config=default&split=test&offset={offset}&length=100')
        rows += [r['row'] for r in page['rows']]
        time.sleep(1)  # be polite to the public API
        print(f'\r{len(rows)}/{total}', end='', flush=True)
    print()
    if len(rows) != total or len({r['question_id'] for r in rows}) != total:
        raise RuntimeError('Incomplete or duplicated download.')
    return rows


def sample(rows, seed=SEED):
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r['category']].append(r)
    rng = random.Random(seed)
    items = []
    for cat in sorted(by_cat):
        drawn = rng.sample(sorted(by_cat[cat], key=lambda r: r['question_id']), PER_CATEGORY)
        for i, r in enumerate(drawn):
            opts = list(r['options'])
            if not 2 <= len(opts) <= 10 or r['answer'] != chr(65 + r['answer_index']) or r['answer_index'] >= len(opts):
                raise RuntimeError(f'Unexpected record {r["question_id"]}.')
            items.append({'id': f'mmlu-pro-{r["question_id"]}', 'question_id': r['question_id'], 'category': cat,
                          'src': r['src'], 'split': 'calibration' if i < CAL_PER_CATEGORY else 'test',
                          'question': r['question'], 'options': opts, 'answer': r['answer']})
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--accept-revision', help='allow this (newer) revision instead of REVISION')
    expected = parser.parse_args().accept_revision or REVISION
    rev = revision()
    if rev != expected:
        raise SystemExit(f'Hugging Face now serves revision {rev}, not {expected}. The datasets-server API cannot '
                         'fetch older revisions; keep the committed sample or re-run with --accept-revision.')
    rows = fetch_test_rows()
    if revision() != rev:
        raise SystemExit('Dataset changed during download; run again.')
    items = sample(rows)
    meta = {'dataset': DATASET, 'split': 'test', 'revision': rev, 'license': 'MIT',
            'url': f'https://huggingface.co/datasets/{DATASET}', 'paper': 'https://arxiv.org/abs/2406.01574',
            'fetched': datetime.now(timezone.utc).date().isoformat(), 'test_split_size': len(rows),
            'sampling': f'{PER_CATEGORY} per category, random.Random({SEED}).sample over question_id-sorted rows, '
                        f'categories in alphabetical order; first {CAL_PER_CATEGORY} drawn = calibration, rest = test',
            'excluded_fields': ['cot_content', 'answer_index']}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({'meta': meta, 'items': items}, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    cats = sorted({i['category'] for i in items})
    print(f'Wrote {len(items)} questions from {len(cats)} categories (revision {rev[:10]}) to {OUT.name}')


if __name__ == '__main__':
    main()
