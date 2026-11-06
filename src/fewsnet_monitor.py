"""
A small local page with a Refresh button that asks the FEWS NET API whether a month's white maize
wholesale prices (Nigeria) are out yet.

    python3 src/fewsnet_monitor.py            # then open http://localhost:8765
    python3 src/fewsnet_monitor.py --port 9000

Why a local server: the FEWS NET API sends no cross-origin header, so a web page cannot call it
directly; this script fetches on the page's behalf. Standard library only, read-only, nothing saved.

Each check reads https://fdw.fews.net/api/marketpricefacts.json for Nigeria over the chosen month and the
three before it, keeps product "Maize Grain (White)", price type "Wholesale", monthly series, and converts
the value to NGN/kg from the record's unit (100_kg -> divide by 100), the same conversion as the gap-filled
panel. "Published on" is the record's status_changed timestamp (docs/PUBLICATION_LAGS.md).

Weekly check (added 2026-11-04): FEWS NET's weekly quotes reach the API about 4 days after each week, weeks before the
monthly value (Sep 2026: every week in by 4 Oct, monthly still absent on 4 Nov). The page also asks each market's weekly
series (schedule=Weekly, one request per dataseries, run in parallel) and shows weeks in / weeks expected (Wednesdays in
the month), the mean so far, and when the latest week entered FEWS NET's database (`created`). Same rules as
src/fewsnet_weekly_feeder.py, which feeds the panel; this page only reports.
"""
from __future__ import annotations

import argparse
import json
import time
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

API = "https://fdw.fews.net/api/marketpricefacts.json"
PRODUCT, PRICE_TYPE = "Maize Grain (White)", "Wholesale"
WEEKLY_API = "https://fdw.fews.net/api/marketpricefacts/?format=json&schedule=Weekly&dataseries={}"
# white maize wholesale weekly dataseries, as SERIES in src/fetch_fewsnet_weekly.py (copied: this page is standard library only)
WEEKLY_SERIES = {"Maiduguri": 15129, "Gombe": 15081, "Gujungu": 15093, "Giwa": 15069, "Saminaka": 15141, "Kano, Dawanau": 15045,
                 "Dandume": 15033, "Gwandu, Dodoru": 15057, "Lagos, Mile 12": 15117, "Ibadan, Bodija": 15021,
                 "Kaura Namoda": 15105, "Biu": 25467, "Damaturu": 25453, "Mubi": 25481, "Potiskum": 25439}


def month_start(ym: str, back: int = 0) -> date:
    y, m = map(int, ym.split("-"))
    m -= back
    while m <= 0:
        m, y = m + 12, y - 1
    return date(y, m, 1)


def kg_per_unit(unit: str) -> float | None:
    if unit == "kg":
        return 1.0
    hit = re.fullmatch(r"(\d+)_kg", unit or "")
    if hit:
        return float(hit.group(1))
    return {"mt": 1000.0, "tonne": 1000.0}.get(unit)


def wednesdays(month: str) -> list[date]:
    d, nxt = month_start(month), month_start(month, -1)
    out = []
    while d < nxt:
        if d.weekday() == 2:
            out.append(d)
        d += timedelta(days=1)
    return out


def weekly_one(market: str, sid: int, month: str) -> dict:
    req = urllib.request.Request(WEEKLY_API.format(sid), headers={"User-Agent": "fewsnet-monitor"})
    with urllib.request.urlopen(req, timeout=120) as r:
        rows = json.load(r)
    rows = rows if isinstance(rows, list) else rows.get("results", [])
    weeks = {}
    for x in rows:
        if x.get("value") is None or x.get("product") != PRODUCT or x.get("collection_schedule") != "Weekly":
            continue
        d = date.fromisoformat((x.get("period_date") or "")[:10])
        wk = d + timedelta(days=(2 - d.weekday()) % 7)          # a Tuesday quote belongs to that week's Wednesday
        k = kg_per_unit(x.get("unit"))
        if k:
            weeks[wk] = (x["value"] / k, (x.get("created") or "")[:16].replace("T", " "))
    exp = wednesdays(month)
    got = {w: weeks[w] for w in exp if w in weeks}
    latest = max(weeks) if weeks else None
    status = ("complete" if len(got) == len(exp) else "closed with gaps" if got and latest and latest >= month_start(month, -1)
              else "partial" if got else "none")
    return {"weeks_in": len(got), "weeks_expected": len(exp), "status": status,
            "mean": round(sum(v[0] for v in got.values()) / len(got), 2) if got else None,
            "last_week": max(got).isoformat() if got else None, "entered": max(v[1] for v in got.values()) if got else None,
            "latest_week_any": latest.isoformat() if latest else None}


def weekly_check(month: str) -> dict:
    def one(item):
        try:
            return item[0], weekly_one(item[0], item[1], month)
        except Exception as e:  # one market failing is shown, not hidden
            return item[0], {"error": str(e)}
    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(sorted(ex.map(one, WEEKLY_SERIES.items())))


def check(month: str) -> dict:
    # One request per month. On 4 Nov 2026 a single request spanning Jun-Sep 2026 returned Jun, Jul and Aug only (891
    # records, Sep missing) although FEWS NET had published Sep on 4 Oct; on 5 Nov the same request returned all four
    # months (1,188). The cause is not confirmed (no row cap; the API sends Cache-Control max-age=900, private), so each
    # month is requested on its own with a cache-busting parameter, and the old multi-month request is repeated as a
    # cross-check: if its row count differs from the per-month total the page says so.
    months = [month_start(month, k).strftime("%Y-%m") for k in (3, 2, 1, 0)]
    rows, urls, per_month = [], [], {}

    def get(start: date, end: date) -> list:
        q = {"country_code": "NG", "start_date": start.isoformat(), "end_date": end.isoformat(), "nocache": str(time.time_ns())}
        urls.append(API + "?" + urllib.parse.urlencode(q))
        with urllib.request.urlopen(urllib.request.Request(urls[-1], headers={"User-Agent": "fewsnet-monitor"}), timeout=120) as r:
            got = json.load(r)
        return got if isinstance(got, list) else got.get("results", [])

    for ym in months:
        got = get(month_start(ym), month_start(ym, -1) - timedelta(days=1))
        per_month[ym] = len(got)
        rows += got
    span = len(get(month_start(months[0]), month_start(month, -1) - timedelta(days=1)))
    cross = {"per_month_rows": per_month, "multi_month_rows": span, "agree": span == sum(per_month.values())}
    url = urls[-2]  # the latest month on its own (urls[-1] is the cross-check request)
    out: dict = {}
    for x in rows:
        if x.get("product") != PRODUCT or x.get("price_type") != PRICE_TYPE or x.get("collection_schedule") != "Monthly":
            continue
        ym = (x.get("period_date") or "")[:7]
        if ym not in months:
            continue
        k = kg_per_unit(x.get("unit"))
        out.setdefault(x["market"], {})[ym] = {
            "price": round(x["value"] / k, 2) if k and x.get("value") is not None else None,
            "raw": f'{x.get("value")} per {x.get("unit")}',
            "published": (x.get("status_changed") or "")[:10],
            "status": x.get("collection_status") or x.get("status"),
        }
    return {"month": month, "months": months, "markets": dict(sorted(out.items())), "url": url, "weekly": weekly_check(month),
            "records_read": len(rows), "request_crosscheck": cross, "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FEWS NET release check</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mute:#6b7280;--line:#e5e7eb;--navy:#0E355C;--teal:#01ABA8;--tint:#e9f7f6;--warn:#fff7e6}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--fg:#e8eaed;--mute:#9aa0a6;--line:#2a2f36;--navy:#8fb8e6;--teal:#3fd0cc;--tint:#13302f;--warn:#33280f}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,Avenir,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;color:var(--navy);margin:0 0 4px}
h2{font-size:16px;color:var(--navy);margin:22px 0 8px}
.sub{color:var(--mute);margin:0 0 20px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:18px}
button{background:var(--navy);color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:15px;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}
input{font-size:15px;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
.status{border-radius:10px;padding:16px 18px;margin-bottom:18px;font-size:18px;font-weight:600}
.yes{background:var(--tint);color:var(--teal)} .no{background:var(--warn)} .err{background:var(--warn);color:#b45309}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:560px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--navy);font-size:13px} td small{display:block;color:var(--mute);font-size:12px}
.target{background:var(--tint)} .missing{color:var(--mute)}
.foot{color:var(--mute);font-size:12px;margin-top:14px;word-break:break-all}
</style></head><body><main>
<h1>FEWS NET release check</h1>
<p class="sub">White maize, wholesale, Nigeria. NGN/kg. Monthly values and the weekly quotes inside the month, each requested month by month.</p>
<div class="bar"><label>Month <input id="m" type="month"></label><button id="go">Refresh</button><span id="when" class="sub" style="margin:0"></span></div>
<div id="status" class="status no">Press Refresh to check.</div>
<div id="wstatus" class="status no" style="display:none"></div>
<h2>Weekly quotes in the month</h2>
<div class="scroll"><table id="w"></table></div>
<h2>Monthly values</h2>
<div class="scroll"><table id="t"></table></div>
<p class="foot" id="foot"></p>
</main><script>
const m=document.getElementById('m'), go=document.getElementById('go');
m.value=new URLSearchParams(location.search).get('month')||'__DEFAULT__';
async function run(){
  go.disabled=true; go.textContent='Checking...';
  const st=document.getElementById('status'), t=document.getElementById('t');
  try{
    const r=await fetch('/api/check?month='+encodeURIComponent(m.value)); const d=await r.json();
    if(d.error) throw new Error(d.error);
    const names=Object.keys(d.markets), n=names.filter(k=>d.markets[k][d.month]).length;
    const label=new Date(d.month+'-01T00:00').toLocaleString('en-GB',{month:'long',year:'numeric'});
    st.className='status '+(n?'yes':'no');
    st.textContent=n?`${label}, monthly: published for ${n} of ${names.length} markets`:`${label}, monthly: not published yet (0 of ${names.length} markets)`;
    let h='<tr><th>Market</th>'+d.months.map(x=>`<th${x===d.month?' class="target"':''}>${x}</th>`).join('')+'</tr>';
    for(const k of names){ h+='<tr><td>'+k+'</td>'+d.months.map(x=>{const v=d.markets[k][x], c=x===d.month?' class="target"':'';
      return v?`<td${c}>${v.price??v.raw}<small>published ${v.published||'?'}</small></td>`:`<td${c}><span class="missing">not yet</span></td>`}).join('')+'</tr>'; }
    t.innerHTML=h;
    const W=d.weekly||{}, wn=Object.keys(W), done=wn.filter(k=>W[k].status==='complete').length, some=wn.filter(k=>W[k].weeks_in>0).length;
    const ws=document.getElementById('wstatus'); ws.style.display='block';
    ws.className='status '+(some?'yes':'no');
    ws.textContent=`${label}, weekly: complete for ${done} of ${wn.length} markets, at least one week for ${some}`;
    const odd=wn.filter(k=>W[k].status==='complete' && !(d.markets[k]||{})[d.month]);   // guard against the 4 Nov 2026 query bug
    if(odd.length){ ws.className='status err'; ws.textContent+=` | CHECK: weekly complete but no monthly value for ${odd.join('; ')}. FEWS NET normally enters both together; verify the monthly query before concluding it is unpublished.`; }
    let g='<tr><th>Market</th><th>Weeks in</th><th>Mean so far</th><th>Latest week</th><th>Entered</th><th>Status</th></tr>';
    for(const k of wn){ const v=W[k];
      g+= v.error?`<tr><td>${k}</td><td colspan="5" class="missing">error: ${v.error}</td></tr>`:
        `<tr><td>${k}</td><td>${v.weeks_in} of ${v.weeks_expected}</td><td>${v.mean??'<span class="missing">none</span>'}</td><td>${v.last_week??'<small>last any: '+(v.latest_week_any||'?')+'</small>'}</td><td>${v.entered??''}</td><td${v.status==='complete'?' class="target"':''}>${v.status}</td></tr>`; }
    document.getElementById('w').innerHTML=g;
    document.getElementById('when').textContent='Last checked '+d.checked_at;
    document.getElementById('foot').textContent=`${d.records_read} records read from ${d.url}`;
    const c=d.request_crosscheck;   // the 4 Nov 2026 multi-month request returned one month short; flag any repeat
    if(c && !c.agree){ ws.className='status err'; ws.textContent+=` | CHECK: one request over ${d.months[0]} to ${d.month} returned ${c.multi_month_rows} rows, the four single-month requests ${Object.values(c.per_month_rows).reduce((a,b)=>a+b,0)} (${JSON.stringify(c.per_month_rows)}). FEWS NET's API is answering inconsistently; trust neither until they agree.`; }
  }catch(e){ st.className='status err'; st.textContent='Could not reach FEWS NET: '+e.message; }
  go.disabled=false; go.textContent='Refresh';
}
go.onclick=run; run();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    default_month = "2026-09"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        if u.path == "/":
            self._send(200, PAGE.replace("__DEFAULT__", self.default_month).encode(), "text/html; charset=utf-8")
        elif u.path == "/api/check":
            month = urllib.parse.parse_qs(u.query).get("month", [self.default_month])[0]
            try:
                if not re.fullmatch(r"\d{4}-\d{2}", month):
                    raise ValueError("month must look like 2026-09")
                body = check(month)
            except Exception as e:  # shown on the page, not hidden
                body = {"error": str(e)}
            self._send(200, json.dumps(body).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--month", default="2026-09", help="month the page opens on (YYYY-MM)")
    a = ap.parse_args()
    Handler.default_month = a.month
    print(f"FEWS NET release check: http://localhost:{a.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
