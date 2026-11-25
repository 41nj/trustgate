"""Synthetic questions and prediction records for tests only. Never used by the application."""
import random

LETTERS = 'ABCD'


def questions(n=30):
    """Minimal four-option questions with a known answer key."""
    return [{'id': i + 1, 'question': f'Synthetic question {i + 1}?', 'options': [f'option {c}' for c in LETTERS],
             'answer': LETTERS[i % 4]} for i in range(n)]


def synthetic_rows(seed=42):
    """Deterministic, deliberately overconfident prediction records."""
    rng = random.Random(seed)
    rows = []
    for q in questions():
        p = round(rng.uniform(.38, .99), 2)
        correct = rng.random() < max(.15, p - .16)
        answer = q['answer'] if correct else rng.choice([x for x in LETTERS if x != q['answer']])
        rows.append({**q, 'prediction': answer, 'confidence': p, 'correct': correct})
    return rows
