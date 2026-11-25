"""TrustGate: local-only HTTP server, zero third-party dependencies."""
import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmark import SUBSET_SIZES, subset
from engine import evaluate
from provider import MODEL_RE, ProviderError, cache_path, cached_records, dashboard_payload, evaluate_run, load_config, run_identity

ROOT = Path(__file__).resolve().parent
PUBLISHED = ROOT / 'evaluations'


def published():
    """Evaluations shipped in evaluations/*.json, viewable offline."""
    out=[]
    for f in sorted(PUBLISHED.glob('*.json')):
        try: data=json.loads(f.read_text(encoding='utf-8'))
        except (OSError,ValueError): continue
        out.append({'name':f.stem,'model':(data.get('run') or {}).get('model'),'dataset':(data.get('dataset') or {}).get('dataset'),'n':len(data.get('rows',[]))})
    return out


def prepare(model, count):
    """Validate a run request; return (cfg, dataset meta, items, cache path, cached question ids)."""
    cfg=load_config()
    if not cfg.key:
        raise ProviderError('No API provider configured. Set TRUSTGATE_API_BASE and TRUSTGATE_API_KEY, then restart.')
    if count not in SUBSET_SIZES:
        raise ValueError('Choose 14, 70 or 140 questions.')
    if not isinstance(model,str) or not MODEL_RE.fullmatch(model):
        raise ValueError('Invalid model ID.')
    dataset,items=subset(SUBSET_SIZES[count])
    # Same cache as evaluate_model.py; reuse only under an identical identity (provider, settings, data).
    identity=run_identity(cfg,model,dataset)
    path=cache_path(cfg,model)
    records=cached_records(path,identity)
    cached={q['id'] for q in items if q['id'] in records}
    return cfg,dataset,items,path,cached,identity


def plan_live(model, count):
    _,_,items,_,cached,_=prepare(model,count)
    return {'total':len(items),'cached':len(cached),'new':len(items)-len(cached)}


def run_live(model, count):
    cfg,dataset,items,path,cached,identity=prepare(model,count)
    errors=[]
    def log(line):
        print(line,flush=True)
        if 'request failed' in line: errors.append(line.split('request failed',1)[1].strip(' ()'))
    meta,records,failures=evaluate_run(cfg,items,model,path,identity,log=log)
    if failures:
        if not records: path.unlink(missing_ok=True)  # nothing but the header line
        raise ProviderError(f'{len(failures)} of {len(items)} requests failed ({errors[-1] if errors else "unknown error"}). Completed answers were saved; run again to resume.')
    payload=dashboard_payload(meta,records,dataset=dataset,total=len(items))
    for r in payload['rows']+payload['excluded']:
        r['origin']='cached' if r['id'] in cached else 'new'
    payload['counts'].update(cached=len(cached),new=len(items)-len(cached))
    if not payload['rows']:
        raise ValueError('No parseable model answers.')
    return payload


class Handler(BaseHTTPRequestHandler):
    def respond(self, value, code=200):
        body=json.dumps(value,allow_nan=False).encode()
        self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/status':
            try: cfg=load_config()
            except (ValueError,OSError): cfg=None
            return self.respond({'live_available':bool(cfg and cfg.key),'default_model':cfg.model if cfg else None,'evaluations':published()})
        m=re.fullmatch(r'/api/evaluations/([A-Za-z0-9._-]{1,120})',self.path)
        if m:
            f=PUBLISHED/f'{m.group(1)}.json'
            if not f.is_file(): return self.send_error(404)
            data=json.loads(f.read_text(encoding='utf-8')); data['source']='Real evaluation: '+data.get('run',{}).get('model','?')
            return self.respond(data)
        assets={'/':('index.html','text/html; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8')}
        if self.path not in assets:
            return self.send_error(404)
        name,mime=assets[self.path]; data=(ROOT/'static'/name).read_bytes()
        self.send_response(200); self.send_header('Content-Type',mime); self.send_header('Content-Length',str(len(data))); self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(data)

    def do_POST(self):
        try:
            # Same-origin requests only; JSON content required.
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://localhost:{self.server.server_port}',f'http://127.0.0.1:{self.server.server_port}'):
                return self.respond({'error':'Cross-origin request rejected.'},403)
            if self.headers.get_content_type()!='application/json':
                return self.respond({'error':'Expected application/json.'},415)
            size=int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 5_000_000:
                return self.respond({'error':'Invalid payload size.'},400)
            data=json.loads(self.rfile.read(size))
            if self.path=='/api/metrics':
                return self.respond(evaluate(data['rows'],data['threshold'],data['wrong_cost'],data['defer_cost']))
            if self.path=='/api/live/plan':
                return self.respond(plan_live(data['model'],data['count']))
            if self.path=='/api/live':
                return self.respond(run_live(data['model'],data['count']))
            return self.respond({'error':'Not found.'},404)
        except ProviderError as e:
            return self.respond({'error':str(e)[:300]},502)
        except OSError:
            return self.respond({'error':'Could not read configuration or write results.'},500)
        except (ValueError,KeyError,TypeError,AttributeError) as e:
            return self.respond({'error':'Invalid input or model response: '+str(e)[:200]},400)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--port',type=int,default=8000); args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'TrustGate → http://127.0.0.1:{server.server_port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
