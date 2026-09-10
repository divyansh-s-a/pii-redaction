#!/usr/bin/env python3
"""Local web app for the redaction workflow — stdlib only, localhost only.

    python3 tools/webapp.py [--port 8788]

Two views:
  Upload & Redact   drop an audio file -> the production workflow runs
                    (all gates) -> download the redacted outputs, or see the
                    quarantine notice.
  Review quarantine per held file: each flagged beep edge with an audio
                    snippet player (hear exactly what the listener flagged),
                    Extend-beep buttons, and Release / Keep with a mandatory
                    reviewer note. Actions call tools/release.py functions;
                    every decision lands in the manifest.

Security: binds 127.0.0.1 only; API keys stay in .env on this machine; this
is a single-operator tool, not a hosted service (see the report's product
path for that).
"""
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "tools"))
import audio_utils      # noqa: E402
import common           # noqa: E402
import gemini_transcribe  # noqa: E402  (cut_wav_chunk)
import release as rel   # noqa: E402

UPLOADS = ROOT / "media" / "incoming" / "uploads"
JOBS = {}               # job_id -> {file, fid, state, log_tail, started}
JOBS_LOCK = threading.Lock()


def fid_for(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_")
    return stem or "upload"


def run_job(job_id, media: Path):
    with JOBS_LOCK:
        JOBS[job_id]["state"] = "running"
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "redact.py"),
                        str(media)], cwd=str(ROOT),
                       capture_output=True, text=True)
    fid = fid_for(media)
    if (ROOT / "deliveries" / fid / "manifest.json").exists():
        state = "delivered"
    elif (ROOT / "quarantine" / fid / "manifest.json").exists():
        state = "quarantined"
    else:
        state = "error"
    with JOBS_LOCK:
        JOBS[job_id].update(state=state,
                            log_tail=(p.stdout + p.stderr)[-1500:])


def quarantine_summary():
    out = []
    qd = ROOT / "quarantine"
    if not qd.exists():
        return out
    for d in sorted(qd.iterdir()):
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        ev = m.get("edge_verification") or {}
        out.append({
            "fid": d.name,
            "reason": m.get("quarantined_reason"),
            "qc": m.get("qc_status"),
            "audit_leaks": (m.get("judge_audit") or {}).get("leak_candidates"),
            "unresolved": ev.get("unresolved") or [],
            "unchecked": ev.get("unchecked") or [],
            "actions": (m.get("human_review") or {}).get("actions", []),
        })
    return out


def snippet_wav(fid: str, a_ms: int, b_ms: int) -> bytes:
    """A WAV slice of the quarantined file's CURRENT beeped audio."""
    src = ROOT / "quarantine" / fid / common.beeped_name(fid)
    tmp = ROOT / "quarantine" / fid / "_web_snip_src.wav"
    out = ROOT / "quarantine" / fid / "_web_snip.wav"
    audio_utils.decode_to_wav(src, tmp, rate=16000, mono=True)
    gemini_transcribe.cut_wav_chunk(tmp, out, max(0, a_ms) / 1000,
                                    max(300, b_ms - a_ms) / 1000)
    data = out.read_bytes()
    tmp.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return data


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Audio PII Redaction</title><style>
body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;background:#f5f7fa;color:#16324f}
header{background:#14213d;color:#fff;padding:14px 28px;display:flex;gap:24px;align-items:center}
header h1{font-size:18px;margin:0}
.tab{cursor:pointer;padding:6px 14px;border-radius:8px;color:#cbd5e1}
.tab.active{background:#2b3a67;color:#fff}
main{max-width:980px;margin:24px auto;padding:0 20px}
.card{background:#fff;border:1px solid #dbe1ea;border-radius:14px;padding:20px;margin-bottom:18px}
button{background:#159f7f;color:#fff;border:0;border-radius:8px;padding:8px 14px;cursor:pointer;font-size:14px}
button.warn{background:#e8a020}button.ghost{background:#5b7290}
button:disabled{opacity:.45;cursor:default}
.badge{display:inline-block;border-radius:12px;padding:2px 10px;font-size:12px;font-weight:700}
.b-del{background:#d9f2e2;color:#14532d}.b-q{background:#fdeed2;color:#92500a}
.b-run{background:#d8eef8;color:#125a75}.b-err{background:#fcdcdc;color:#7f1d1d}
.edge{border:1px solid #e5e9f0;border-radius:10px;padding:12px;margin:10px 0;background:#fafbfd}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#5f6b7a}
input[type=text]{width:60%;padding:7px;border:1px solid #cbd5e1;border-radius:8px}
audio{vertical-align:middle;height:34px}
.small{font-size:13px;color:#64748b}
a{color:#125a75}
</style></head><body>
<header><h1>Audio PII Redaction</h1>
<div class="tab active" id="t-up" onclick="show('up')">Upload &amp; Redact</div>
<div class="tab" id="t-q" onclick="show('q')">Review quarantine</div>
</header><main>
<div id="v-up">
 <div class="card"><h3>Redact an audio file</h3>
  <p class="small">The full production workflow runs locally: transcribe &rarr; detect &rarr;
  word-aware beeps &rarr; QC &rarr; beep-edge listen check &rarr; judge audit. Files that fail a
  gate are held for review, never shipped.</p>
  <input type="file" id="file" accept="audio/*">
  <button onclick="upload()" id="upbtn">Redact</button></div>
 <div class="card"><h3>Jobs</h3><div id="jobs" class="small">none yet</div></div>
</div>
<div id="v-q" style="display:none"></div>
</main><script>
function show(v){for(const x of ['up','q']){document.getElementById('v-'+x).style.display=x===v?'':'none';
document.getElementById('t-'+x).classList.toggle('active',x===v);}if(v==='q')loadQ();}
async function upload(){const f=document.getElementById('file').files[0];if(!f)return alert('pick a file');
document.getElementById('upbtn').disabled=true;
const r=await fetch('/api/upload?name='+encodeURIComponent(f.name),{method:'POST',body:f});
document.getElementById('upbtn').disabled=false;
if(!r.ok){alert('upload failed');return;}poll();}
async function poll(){const r=await fetch('/api/jobs');const js=await r.json();
const el=document.getElementById('jobs');
if(!js.length){el.textContent='none yet';return;}
el.innerHTML=js.map(j=>{
 let b={running:'b-run',delivered:'b-del',quarantined:'b-q',error:'b-err'}[j.state]||'b-run';
 let extra='';
 if(j.state==='delivered')extra=' &nbsp; <a href="/download/'+j.fid+'/'+j.fid+'_beeped.__BEEP_EXT__">beeped audio</a>'
  +' &middot; <a href="/download/'+j.fid+'/'+j.fid+'_redacted.txt">redacted transcript</a>'
  +' &middot; <a href="/download/'+j.fid+'/qc_report.md">QC</a>'
  +' &middot; <a href="/download/'+j.fid+'/manifest.json">manifest</a>';
 if(j.state==='quarantined')extra=' &nbsp; held for review — open the Review quarantine tab';
 return '<div style="margin:6px 0"><span class="badge '+b+'">'+j.state+'</span> '+j.fid+extra+'</div>';
}).join('');
if(js.some(j=>j.state==='running'))setTimeout(poll,4000);}
async function loadQ(){const r=await fetch('/api/quarantine');const qs=await r.json();
const v=document.getElementById('v-q');
if(!qs.length){v.innerHTML='<div class="card">Quarantine is empty.</div>';return;}
v.innerHTML=qs.map(q=>{
 const edges=q.unresolved.map((u,i)=>{
  const s=u.beep[0],e=u.beep[1];
  const a=Math.max(0,(u.side==='start'?s-2500:e-500)),b=(u.side==='start'?s+500:e+2500);
  return '<div class="edge"><b>beep '+s+'&ndash;'+e+' ms</b> &middot; flagged side: '+u.side
   +' &middot; heard: <span class="mono">'+u.heard.join(', ')+'</span><br>'
   +'<audio controls preload="none" src="/api/snippet?fid='+q.fid+'&a='+a+'&b='+b+'"></audio> '
   +'<button class="warn" onclick="extend(\\''+q.fid+'\\','+s+','+e+',\\''+u.side+'\\')">Extend beep 400 ms</button>'
   +'</div>';}).join('');
 const acts=q.actions.length?'<p class="small">history: '+q.actions.map(a=>a.action+' @'+a.at).join(' &middot; ')+'</p>':'';
 return '<div class="card"><h3>'+q.fid+'</h3>'
  +'<p class="small">reason: '+q.reason+' &middot; QC '+q.qc+' &middot; audit leaks '+q.audit_leaks+'</p>'
  +edges+acts
  +'<p><input type="text" id="note-'+q.fid+'" placeholder="reviewer note (required)"> '
  +'<button onclick="decide(\\''+q.fid+'\\',\\'release\\')">Approve &amp; release</button> '
  +'<button class="ghost" onclick="decide(\\''+q.fid+'\\',\\'keep\\')">Keep held</button></p></div>';
}).join('');}
async function extend(fid,s,e,side){
 const r=await fetch('/api/extend',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({fid:fid,beep:[s,e],side:side,ms:400})});
 alert(r.ok?'beep extended and audio re-rendered — re-listen to confirm':'extend failed');loadQ();}
async function decide(fid,action){
 const note=document.getElementById('note-'+fid).value.trim();
 if(!note)return alert('a reviewer note is required');
 const r=await fetch('/api/'+action,{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({fid:fid,note:note})});
 alert(r.ok?(action==='release'?'released to deliveries/':'kept; decision recorded'):'action failed');loadQ();}
poll();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            return self._send(200,
                               PAGE.replace("__BEEP_EXT__", common.BEEP_FORMAT)
                               .encode(), "text/html; charset=utf-8")
        if u.path == "/api/jobs":
            with JOBS_LOCK:
                js = [dict(job_id=k, **{kk: v[kk] for kk in
                                        ("fid", "state")})
                      for k, v in sorted(JOBS.items())]
            return self._send(200, js)
        if u.path == "/api/quarantine":
            return self._send(200, quarantine_summary())
        if u.path == "/api/snippet":
            fid = re.sub(r"[^A-Za-z0-9._-]", "", q["fid"][0])
            try:
                wav = snippet_wav(fid, int(q["a"][0]), int(q["b"][0]))
            except Exception as e:
                return self._send(500, {"error": str(e)[:200]})
            return self._send(200, wav, "audio/wav")
        if u.path.startswith("/download/"):
            parts = u.path.split("/")
            if len(parts) == 4:
                fid = re.sub(r"[^A-Za-z0-9._-]", "", parts[2])
                name = re.sub(r"[^A-Za-z0-9._-]", "", parts[3])
                p = ROOT / "deliveries" / fid / name
                if p.exists():
                    ctype = ("audio/flac" if name.endswith(".flac")
                             else "audio/mp4" if name.endswith(".m4a")
                             else "text/plain; charset=utf-8")
                    return self._send(200, p.read_bytes(), ctype)
            return self._send(404, {"error": "not found"})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if u.path == "/api/upload":
            q = urllib.parse.parse_qs(u.query)
            name = re.sub(r"[^A-Za-z0-9 ._-]", "",
                          (q.get("name") or ["upload.mp3"])[0]) or "upload.mp3"
            UPLOADS.mkdir(parents=True, exist_ok=True)
            dest = UPLOADS / name
            k = 1
            while dest.exists():
                dest = UPLOADS / f"{Path(name).stem}_{k}{Path(name).suffix}"
                k += 1
            dest.write_bytes(raw)
            job_id = f"j{int(time.time())}_{dest.stem[:16]}"
            with JOBS_LOCK:
                JOBS[job_id] = {"file": str(dest), "fid": fid_for(dest),
                                "state": "queued", "log_tail": "",
                                "started": time.time()}
            threading.Thread(target=run_job, args=(job_id, dest),
                             daemon=True).start()
            return self._send(200, {"job_id": job_id})
        try:
            body = json.loads(raw.decode() or "{}")
        except ValueError:
            return self._send(400, {"error": "bad json"})
        fid = re.sub(r"[^A-Za-z0-9._-]", "", str(body.get("fid", "")))
        try:
            if u.path == "/api/extend":
                nb = rel.extend(fid, int(body["beep"][0]),
                                int(body["beep"][1]), body["side"],
                                int(body.get("ms", 400)))
                return self._send(200, {"new_beep": list(nb[:2])})
            if u.path == "/api/release":
                rel.release(fid, str(body.get("note", ""))[:500])
                return self._send(200, {"ok": True})
            if u.path == "/api/keep":
                rel.keep(fid, str(body.get("note", ""))[:500])
                return self._send(200, {"ok": True})
        except SystemExit as e:
            return self._send(400, {"error": str(e)[:200]})
        except Exception as e:
            return self._send(500, {"error": str(e)[:200]})
        return self._send(404, {"error": "not found"})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"Audio PII Redaction web app -> http://127.0.0.1:{a.port} "
          "(localhost only; Ctrl-C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
