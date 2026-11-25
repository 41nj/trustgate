"""The fixed MMLU-Pro sample (see fetch_mmlu_pro.py) and its reproducible per-category subsets."""
import hashlib
import json
from pathlib import Path

MMLU_PRO_FILE = Path(__file__).resolve().parent / 'data' / 'mmlu_pro_sample.json'
SUBSET_SIZES = {14: 1, 70: 5, 140: 10}  # questions -> questions per category


def mmlu_pro():
    """(meta, items) of the committed 140-question sample. Items contain no explanations."""
    raw = MMLU_PRO_FILE.read_bytes()
    data = json.loads(raw)
    return {**data['meta'], 'sample_sha256': hashlib.sha256(raw).hexdigest()[:16]}, data['items']


def subset(per_category):
    """(meta, items) with the same number of questions from every category.

    Within each category the sample stores 5 calibration then 5 test questions in draw order.
    Taking them alternately (cal 1, test 1, cal 2, …) keeps both halves represented, so a
    subset of 5 per category still supports the held-out check. The result is deterministic,
    and every smaller subset is contained in the larger ones, so cached answers carry over.
    """
    if per_category not in SUBSET_SIZES.values():
        raise ValueError('Choose 14, 70 or 140 questions.')
    meta, items = mmlu_pro()
    by_cat = {}
    for item in items:
        by_cat.setdefault(item['category'], {'calibration': [], 'test': []})[item['split']].append(item)
    chosen = []
    for cat in sorted(by_cat):
        cal, test = by_cat[cat]['calibration'], by_cat[cat]['test']
        interleaved = [q for pair in zip(cal, test) for q in pair]
        chosen += interleaved[:per_category]
    n = len(chosen)
    note = f'{per_category} per category ({n} questions), alternating calibration/test within each category'
    return {**meta, 'sampling': meta['sampling'] + (f'; subset: {note}' if n < len(items) else '')}, chosen
