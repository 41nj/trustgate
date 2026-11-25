"""Binary correctness evaluation and empirical abstention policies."""
import math


def validate(rows):
    if not isinstance(rows, list) or not rows or len(rows) > 10000:
        raise ValueError('Provide 1–10,000 prediction records.')
    for r in rows:
        if not isinstance(r, dict) or type(r.get('correct')) is not bool:
            raise ValueError('Each record needs a boolean correct field.')
        p = r.get('confidence')
        if type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Confidence must be a finite number in [0, 1].')
    return rows


def auc(rows):
    ordered = sorted(rows, key=lambda r: r['confidence'])
    pos = sum(r['correct'] for r in rows)
    neg = len(rows) - pos
    if not pos or not neg:
        return None
    # Mann–Whitney U, using average ranks for ties.
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j]['confidence'] == ordered[i]['confidence']:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(r['correct'] for r in ordered[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def _policy(accepted, n, wrong_cost, defer_cost):
    return _counts(len(accepted), sum(not r['correct'] for r in accepted), n, wrong_cost, defer_cost)


def _counts(a, e, n, wrong_cost, defer_cost):
    return {'accepted':a,'errors':e,'deferred':n-a,'coverage':a/n,
            'risk':e/a if a else None,'cost':e*wrong_cost+(n-a)*defer_cost}


def policy_curve(rows, wrong_cost, defer_cost):
    """All attainable policies, from 'defer all' (threshold None) to 'accept all'.
    Tied confidence scores must move together, so each distinct score is one step."""
    n = len(rows)
    curve = [dict(_counts(0,0,n,wrong_cost,defer_cost),threshold=None)]
    ordered = sorted(rows,key=lambda r:r['confidence'],reverse=True)
    a = e = 0
    for i,r in enumerate(ordered):
        a += 1; e += not r['correct']
        if i == n-1 or ordered[i+1]['confidence'] != r['confidence']:
            curve.append(dict(_counts(a,e,n,wrong_cost,defer_cost),threshold=r['confidence']))
    return curve


def holdout(rows, wrong_cost, defer_cost):
    """Choose the cost-minimizing threshold on split=='calibration' rows and apply it unchanged to
    split=='test' rows. Returns None unless both halves are present. Ties in cost prefer higher coverage."""
    cal = [r for r in rows if r.get('split') == 'calibration']
    test = [r for r in rows if r.get('split') == 'test']
    if not cal or not test:
        return None
    chosen = min(policy_curve(cal,wrong_cost,defer_cost),key=lambda p:(p['cost'],-p['coverage']))
    t = chosen['threshold']
    accepted = [] if t is None else [r for r in test if r['confidence'] >= t]
    return {'threshold':t,'n_calibration':len(cal),'n_test':len(test),'calibration':chosen,
            'test':_policy(accepted,len(test),wrong_cost,defer_cost),
            'test_answer_all_cost':sum(not r['correct'] for r in test)*wrong_cost,
            'test_defer_all_cost':len(test)*defer_cost}


def evaluate(rows, threshold=.8, wrong_cost=10., defer_cost=1.):
    validate(rows)
    if not all(type(v) in (int,float) and math.isfinite(v) for v in (threshold,wrong_cost,defer_cost)) or not 0 <= threshold <= 1 or not 0 <= wrong_cost <= 1000 or not 0 <= defer_cost <= 1000:
        raise ValueError('Invalid threshold or costs.')
    n = len(rows)
    accuracy = sum(r['correct'] for r in rows) / n
    bins = []
    groups = [[] for _ in range(10)]
    for r in rows:
        groups[min(9, int(r['confidence'] * 10))].append(r)
    for i, group in enumerate(groups):
        k = len(group)
        bins.append({'low':i/10,'high':(i+1)/10,'count':k,
                     'confidence':sum(r['confidence'] for r in group)/k if k else None,
                     'accuracy':sum(r['correct'] for r in group)/k if k else None})
    selected = _policy([r for r in rows if r['confidence'] >= threshold], n, wrong_cost, defer_cost)
    curve = policy_curve(rows, wrong_cost, defer_cost)
    best = min(curve,key=lambda p:(p['cost'],-p['coverage']))
    return {'n':n,'accuracy':accuracy,'mean_confidence':sum(r['confidence'] for r in rows)/n,
            'ece':sum(b['count']/n*abs(b['accuracy']-b['confidence']) for b in bins if b['count']),
            'brier':sum((r['confidence']-r['correct'])**2 for r in rows)/n,
            'auroc':auc(rows),'bins':bins,'curve':curve,'selected':selected,
            'best':best,'baseline_cost':sum(not r['correct'] for r in rows)*wrong_cost,
            'theoretical_threshold':max(0,1-defer_cost/wrong_cost) if wrong_cost else 0,
            'holdout':holdout(rows,wrong_cost,defer_cost)}
