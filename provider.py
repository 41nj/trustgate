"""OpenAI-compatible chat-completions client, answer parsing and resumable result storage.

Standard library only. Works with any endpoint implementing ``POST {base}/chat/completions``
and (optionally) ``GET {base}/models``. No structured-output / JSON-schema features are
assumed, so models such as Gemma served via vLLM or Open WebUI are supported.

Configuration (environment variables, never stored in the repository):
  TRUSTGATE_API_BASE      base URL, default https://api.openai.com/v1
  TRUSTGATE_MODEL         default model ID
  TRUSTGATE_API_KEY       API key (fallback: OPENAI_API_KEY)
  TRUSTGATE_API_KEY_FILE  alternative: file containing the key, plain or as NAME=value lines
  TRUSTGATE_API_KEY_NAME  variable name to read from the key file (e.g. KEY_1)
  TRUSTGATE_TEMPERATURE   default 0
  TRUSTGATE_MAX_TOKENS    default 4096 (an upper bound; reasoning on hard questions needs room)
  TRUSTGATE_TIMEOUT       seconds per request, default 300
  TRUSTGATE_WORKERS       parallel requests, 1–4, default 4
"""
import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROMPT_VERSION = 'mcq-v2'
LETTERS = 'ABCDEFGHIJ'


def system_prompt(letters):
    return ('You answer multiple-choice questions. Think step by step if needed, then finish with exactly two lines:\n'
            f'Answer: <one letter from {letters[0]} to {letters[-1]}>\n'
            'Confidence: <integer from 0 to 100: your probability, in percent, that this answer is correct>')
MODEL_RE = re.compile(r'[A-Za-z0-9._:/@-]{1,120}')
RESULTS_DIR = Path(__file__).resolve().parent / 'results'


class ProviderError(Exception):
    """Transport/API failure. Messages never contain the key, headers or the base URL."""


@dataclass(frozen=True)
class Config:
    base_url: str
    model: str
    key: str | None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 300.0
    workers: int = 4

    def __repr__(self):  # keep the key out of tracebacks and logs
        return f'Config(model={self.model!r}, temperature={self.temperature}, max_tokens={self.max_tokens})'

    def provider_id(self):
        """Short hash of the base URL. Distinguishes providers in the local cache; never published."""
        return hashlib.sha256(self.base_url.encode()).hexdigest()[:12]


def run_identity(cfg, model, dataset):
    """Everything that must match before cached answers may be reused."""
    return {'model': model, 'provider_id': cfg.provider_id(), 'temperature': cfg.temperature,
            'max_tokens': cfg.max_tokens, 'prompt_version': PROMPT_VERSION, 'benchmark': 'mmlu-pro',
            'dataset_revision': dataset.get('revision'), 'sample_sha256': dataset.get('sample_sha256')}


def cache_path(cfg, model):
    """Local, git-ignored cache file: one per model and provider."""
    return RESULTS_DIR / f'mmlu-pro-{run_name(model)}-{cfg.provider_id()}.jsonl'


def cached_records(path, identity):
    """Records stored under this identity. Raises if the file was written under another one."""
    stored, records = load_run(path)
    if stored and {k: stored.get(k) for k in identity} != identity:
        diff = sorted(k for k in identity if stored.get(k) != identity[k])
        raise ValueError(f'Cached answers in {Path(path).name} were produced with different settings ({", ".join(diff)}); move the file away to start fresh.')
    return records


def _read_key_file(path, name):
    text = Path(path).expanduser().read_text(encoding='utf-8')
    if not name:
        return text.strip() or None
    for line in text.splitlines():
        k, sep, v = line.strip().partition('=')
        if sep and k.strip() == name:
            return v.strip().strip('"\'') or None
    raise ValueError(f'{name} not found in key file.')


def _number(env, name, default, cast, low, high):
    raw = env.get(name)
    if raw in (None, ''):
        return default
    value = cast(raw)
    if not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}.')
    return value


def load_config(env=None):
    env = os.environ if env is None else env
    base = (env.get('TRUSTGATE_API_BASE') or 'https://api.openai.com/v1').rstrip('/')
    parsed = urllib.parse.urlparse(base)
    local = parsed.hostname in ('localhost', '127.0.0.1', '::1')
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and local):
        raise ValueError('TRUSTGATE_API_BASE must use https (http only for localhost).')
    key = env.get('TRUSTGATE_API_KEY') or env.get('OPENAI_API_KEY')
    if not key and env.get('TRUSTGATE_API_KEY_FILE'):
        key = _read_key_file(env['TRUSTGATE_API_KEY_FILE'], env.get('TRUSTGATE_API_KEY_NAME'))
    model = env.get('TRUSTGATE_MODEL') or 'gpt-5.4-mini'
    if not MODEL_RE.fullmatch(model):
        raise ValueError('Invalid TRUSTGATE_MODEL.')
    return Config(base, model, key,
                  _number(env,'TRUSTGATE_TEMPERATURE',0.0,float,0,2),
                  _number(env,'TRUSTGATE_MAX_TOKENS',4096,int,16,32768),
                  _number(env,'TRUSTGATE_TIMEOUT',300.0,float,1,900),
                  _number(env,'TRUSTGATE_WORKERS',4,int,1,4))


THINK_RE = re.compile(r'<think>.*?(</think>|$)', re.S | re.I)
JSON_RE = re.compile(r'\{[^{}]*\}')
NUM = r'(\d+(?:\.\d+)?|\.\d+)\s*(%?)'
# Strict: the requested format on its own line (Markdown emphasis/brackets tolerated).
ANSWER_LINE = re.compile(r'^[\s*_#>`-]*answer[\s*_`]*[:=][\s*_`]*\(?\[?([A-J])\]?\)?[\s*_`.]*$', re.I | re.M)
CONF_LINE = re.compile(r'^[\s*_#>`-]*confidence[\s*_`]*[:=][\s*_`]*' + NUM + r'[\s*_`.]*$', re.I | re.M)
# Lenient fallback inside prose. The letter must be upper case ("answer is a ..." is not a letter).
ANSWER_LOOSE = re.compile(r'(?i:answer)\s*(?i:is)?[\s*_:="\'`(\[]{0,8}([A-J])(?![A-Za-z0-9])')
CONF_LOOSE = re.compile(r'(?i:confidence)[\s*_:="\'`]{0,8}' + NUM)


def _confidence(value, percent=False):
    """Confidence was requested on a 0–100 scale; returns a probability in [0, 1].
    A non-integer below 1 (e.g. 0.85) is ambiguous between scales and rejected rather than guessed."""
    if isinstance(value, bool):
        raise ValueError('Confidence is not numeric.')
    if isinstance(value, str):
        value = value.strip()
        percent = percent or value.endswith('%')
        value = float(value.rstrip('%').strip())
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Confidence is not numeric.')
    if not percent and 0 < value < 1 and value != int(value):
        raise ValueError('Ambiguous confidence scale (value below 1 on a 0–100 scale).')
    if not 0 <= value <= 100:
        raise ValueError('Confidence outside 0–100.')
    return value / 100


def parse_prediction(text, letters='ABCD'):
    """Extract (letter, probability) from free-form model output. Raises ValueError if missing/invalid.

    Order: JSON object {"answer": "B", "confidence": 80}; then 'Answer: B' / 'Confidence: 80' on their
    own lines; then a lenient search in prose. Reasoning in <think> blocks is ignored and the *last*
    occurrence wins, because models often discuss options before committing.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty model output.')
    text = THINK_RE.sub('', text)

    def check(letter):
        letter = str(letter).strip().strip('()[]').upper()
        if len(letter) != 1 or letter not in letters:
            raise ValueError(f'Answer is not one of {letters[0]}–{letters[-1]}.')
        return letter

    for blob in reversed(JSON_RE.findall(text)):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        obj = {str(k).lower(): v for k, v in obj.items()} if isinstance(obj, dict) else {}
        if 'answer' in obj and 'confidence' in obj:
            return check(obj['answer']), _confidence(obj['confidence'])
    answers = ANSWER_LINE.findall(text) or ANSWER_LOOSE.findall(text)
    confs = CONF_LINE.findall(text) or CONF_LOOSE.findall(text)
    if not answers:
        raise ValueError('No answer letter found.')
    if not confs:
        raise ValueError('No confidence found.')
    value, pct = confs[-1]
    return check(answers[-1]), _confidence(float(value), bool(pct))


def _redact(text, cfg):
    """Remove anything that could identify the endpoint or leak credentials from provider text."""
    text = str(text)
    if cfg.key:
        text = text.replace(cfg.key, '[secret]')
    text = re.sub(r'https?://\S+', '[url]', text)
    text = re.sub(r'[\w.+-]+@[\w-]+(\.[\w-]+)+', '[email]', text)
    text = re.sub(r'\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b', '[host]', text)
    text = re.sub(r'\b(?:sk|pk|key|token|bearer)[-_ ]?[A-Za-z0-9._-]{8,}', '[secret]', text, flags=re.I)
    text = re.sub(r'\b[A-Za-z0-9_-]{24,}\b', '[secret]', text)
    return re.sub(r'\s+', ' ', text).strip()[:120]


def _safe_http_error(e, cfg):
    """'API returned HTTP <code>' plus, at most, a short redacted message from the response body."""
    detail = ''
    try:
        body = e.read(2000).decode('utf-8', 'replace')
        data = json.loads(body) if body.lstrip().startswith('{') else {}
        err = data.get('error')
        detail = (err.get('message') if isinstance(err, dict) else err) or data.get('detail') or data.get('message') or ''
        if not isinstance(detail, str):
            detail = ''
    except Exception:
        detail = ''
    detail = _redact(detail, cfg) if detail else ''
    return f'API returned HTTP {e.code}' + (f': {detail}' if detail else '')


def _request(cfg, method, path, body=None):
    if not cfg.key:
        raise ProviderError('No API key configured.')
    req = urllib.request.Request(cfg.base_url + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + cfg.key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise ProviderError(_safe_http_error(e, cfg)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ProviderError(f'Connection failed: {type(e).__name__}') from None
    except json.JSONDecodeError:
        raise ProviderError('API returned non-JSON response.') from None


def list_models(cfg):
    data = _request(cfg, 'GET', '/models')
    items = data.get('data', data) if isinstance(data, dict) else data
    return sorted(m['id'] for m in items if isinstance(m, dict) and isinstance(m.get('id'), str))


def chat(cfg, model, messages, retries=4, backoff=5.0):
    """Return (content, finish_reason). Retries transient failures with linear backoff."""
    body = {'model': model, 'messages': messages, 'temperature': cfg.temperature, 'max_tokens': cfg.max_tokens}
    err = None
    for attempt in range(retries):
        try:
            data = _request(cfg, 'POST', '/chat/completions', body)
            choice = data['choices'][0]
            return choice.get('message', {}).get('content') or '', choice.get('finish_reason')
        except ProviderError as e:
            err = e
            code = re.match(r'API returned HTTP (\d+)', str(e))
            if code and code.group(1) in ('400', '401', '403', '404'):
                break  # not transient
        except (KeyError, IndexError, TypeError, AttributeError):
            err = ProviderError('Unexpected response format.')
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    raise err


def format_question(q):
    return q['question'] + '\n' + '\n'.join(f'{chr(65+i)}. {opt}' for i, opt in enumerate(q['options']))


def predict(cfg, q, model, retries=4, backoff=5.0):
    """One prediction record. Unparseable outputs are kept with status 'invalid' (never silently dropped)."""
    letters = LETTERS[:len(q['options'])]
    content, finish = chat(cfg, model, [{'role':'system','content':system_prompt(letters)},
                                        {'role':'user','content':format_question(q)}], retries, backoff)
    rec = {**q, 'finish_reason': finish, 'raw_output': content[:20000]}
    try:
        if finish == 'length':
            raise ValueError('Output truncated (finish_reason=length); increase TRUSTGATE_MAX_TOKENS.')
        letter, p = parse_prediction(content, letters)
    except ValueError as e:
        return {**rec, 'status': 'invalid', 'error': str(e)[:200]}
    return {**rec, 'status': 'ok', 'prediction': letter, 'confidence': p, 'correct': letter == q['answer']}


def run_name(model):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', model).strip('_')[:80] or 'run'


def load_run(path):
    """Return (meta, records keyed by question id). Later lines overwrite earlier ones."""
    meta, records = None, {}
    path = Path(path)
    if not path.exists():
        return meta, records
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get('type') == 'run':
            meta = meta or item
        elif item.get('type') == 'prediction':
            records[item['id']] = {k: v for k, v in item.items() if k != 'type'}
    return meta, records


def evaluate_run(cfg, qs, model, path, identity, log=None, retries=4, backoff=5.0):
    """Evaluate questions, appending each finished record to a JSONL cache file. Resumable: questions
    with a stored record (ok or invalid) are skipped; transport failures are not stored and are
    retried on the next run. Reuse requires an identical run identity (see run_identity)."""
    if not MODEL_RE.fullmatch(model):
        raise ValueError('Invalid model ID.')
    log = log or print
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = cached_records(path, identity)
    stored_meta = load_run(path)[0]
    meta = {**identity, 'started': (stored_meta or {}).get('started') or datetime.now(timezone.utc).isoformat(timespec='seconds')}
    lock = threading.Lock()
    with path.open('a', encoding='utf-8') as f:
        if not stored_meta:
            f.write(json.dumps({'type':'run', **meta}) + '\n')
            f.flush()
        todo = [q for q in qs if q['id'] not in records]
        log(f'model={model} temperature={cfg.temperature} max_tokens={cfg.max_tokens} '
            f'prompt={PROMPT_VERSION} · {len(qs)-len(todo)} stored, {len(todo)} to run')
        failures = []

        def work(q):
            try:
                rec = predict(cfg, q, model, retries, backoff)
            except ProviderError as e:
                with lock:
                    failures.append(q['id'])
                    log(f'  #{q["id"]}: request failed ({e})')
                return
            with lock:
                f.write(json.dumps({'type':'prediction', **rec}, ensure_ascii=False) + '\n')
                f.flush()
                records[q['id']] = rec
                status = f'{rec["prediction"]} @ {rec["confidence"]:.2f}' if rec['status'] == 'ok' else f'invalid: {rec["error"]}'
                log(f'  #{q["id"]}: {status}')

        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            list(pool.map(work, todo))
    return meta, [records[q['id']] for q in qs if q['id'] in records], failures


# Only these fields leave the local cache (dashboard responses, results/*.json, evaluations/).
PUBLIC_RUN = ('model', 'temperature', 'max_tokens', 'prompt_version', 'benchmark', 'dataset_revision', 'started')
PUBLIC_ROW = ('id', 'question_id', 'category', 'split', 'src', 'question', 'options', 'answer', 'prediction', 'confidence', 'correct')
PUBLIC_EXCLUDED = ('id', 'question_id', 'category', 'split', 'status', 'error')


def dashboard_payload(meta, records, failures=(), dataset=None, total=None):
    """Sanitized, dashboard-ready result: no raw model output, provider identity or endpoint.
    Unparseable answers are listed (without their text) so exclusions stay transparent."""
    pick = lambda d, keys: {k: d[k] for k in keys if k in d}
    rows = [pick(r, PUBLIC_ROW) for r in records if r.get('status', 'ok') == 'ok']
    excluded = [pick(r, PUBLIC_EXCLUDED) for r in records if r.get('status') == 'invalid']
    note = []
    if excluded:
        note.append(f'{len(excluded)} unparseable excluded')
    if failures:
        note.append(f'{len(failures)} failed requests')
    source = f'Live: {meta["model"]}' + (f' ({", ".join(note)})' if note else '')
    counts = {'questions': total if total is not None else len(records) + len(failures), 'valid': len(rows),
              'unparseable': len(excluded), 'failed': len(failures)}
    return {'source': source, 'kind': 'real', 'run': pick(meta, PUBLIC_RUN), 'dataset': dataset, 'counts': counts,
            'rows': rows, 'excluded': excluded, 'failed_ids': list(failures)}
