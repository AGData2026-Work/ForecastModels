"""
Maize quotes from the AFEX price-update WhatsApp export (`_chat.txt`), for the gap-filled panel's AFEX source.

The AFEX weekly panel (20260923_AFEX_MultiCommodity_Weekly_Panel.xlsx) applies a narrow new-crop rule: a
"new" maize quote is dropped only in weeks where the same market also reports an untagged or old quote. At
the harvest turn that still mixes new crop into some months, e.g. Dandume Sep 2026: panel 279 NGN/kg, old
crop only 345 (the reporter sends "White maize hybrid Old" and "White maize New" in the same message). This
module rebuilds monthly old-crop prices from the reporters' own messages so those months can be corrected.

Parsing rules (applied to messages whose first lines name one of MARKETS):
- a maize block is a blank-line-separated block whose first line contains "maize" and not "yellow";
- tag: "new" if the first line says new, "old" if it says old, otherwise "untagged";
- price: the first "#/₦/NGN <amount>[-<amount>]" in the block, midpoint of the range; typos such as
  "31,00p" fall back to the other end of the range; amounts under 10,000 per bag are dropped;
- bag weight: the first "<40-200> kg" in the block, else 100 kg (counted in `weight_stated`).
Message dates are the WhatsApp timestamps, day/month/year.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MARKETS = {"Dandume": r"Dandume Market", "Saminaka": r"Saminaka", "Bakori": r"Bakori", "Danja": r"Danja",
           "Pambegua": r"Pambegua", "Tundun Saibu": r"Tundun\s*saibu", "Anchau": r"Anchau"}


def read_messages(chat_txt: str | Path) -> list[tuple[pd.Timestamp, str]]:
    t = Path(chat_txt).read_text(encoding="utf-8")
    for ch in ("‎", "\r"):
        t = t.replace(ch, "")
    t = t.replace(" ", " ").replace(" ", " ")
    out = []
    for m in re.split(r"\n(?=\[\d{1,2}/\d{1,2}/\d{2}, )", t):
        h = re.match(r"\[(\d{1,2})/(\d{1,2})/(\d{2}), [^\]]*\] [^:]*: (.*)", m, re.S)
        if h:
            out.append((pd.Timestamp(2000 + int(h.group(3)), int(h.group(2)), int(h.group(1))), h.group(4)))
    return out


def _num(x: str | None) -> int | None:
    return int(x.replace(",", "")) if x and re.fullmatch(r"[\d,]+", x) else None


def maize_quotes(chat_txt: str | Path) -> pd.DataFrame:
    rows = []
    for date, body in read_messages(chat_txt):
        head = " ".join(body.strip().split("\n")[:3])
        mk = next((k for k, pat in MARKETS.items() if re.search(pat, head, re.I)), None)
        if not mk:
            continue
        for block in re.split(r"\n\s*\n", body):
            lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
            if not lines or not re.search(r"maize", lines[0], re.I) or re.search(r"yellow", lines[0], re.I):
                continue
            txt = " ".join(lines)
            pr = re.search(r"(?:#|₦|NGN)\s*([\d,]{4,}\w?)(?:\s*-\s*(?:#|₦|NGN)?\s*([\d,]{4,}))?", txt)
            if not pr:
                continue
            lo, hi = _num(pr.group(1)), _num(pr.group(2))
            lo, hi = lo or hi, hi or lo
            if not lo:
                continue
            if lo < 10000 or hi > 3 * lo:
                lo = hi = max(lo, hi)
            if lo < 10000:
                continue
            w = re.search(r"(\d{2,3})\s*kg", txt, re.I)
            kg = int(w.group(1)) if w and 40 <= int(w.group(1)) <= 200 else 100
            tag = "new" if re.search(r"\bnew\b", lines[0], re.I) else ("old" if re.search(r"\bold\b", lines[0], re.I) else "untagged")
            rows.append(dict(date=date, market=mk, label=lines[0][:40], tag=tag, price_NGN_per_kg=(lo + hi) / 2 / kg,
                             weight_stated=bool(w)))
    return pd.DataFrame(rows)


def old_crop_overrides(chat_txt: str | Path) -> pd.DataFrame:
    """Monthly mean of old and untagged quotes, for the (market, month) cells where the reporter also sent a
    new-crop quote that month. Months without a new-crop quote are left to the AFEX panel (the two agree to
    a median 1.2% there). Returns months down, markets across."""
    q = maize_quotes(chat_txt)
    q["month"] = q["date"].dt.strftime("%Y-%m")
    keep = []
    for (mk, mo), g in q.groupby(["market", "month"]):
        if "new" in set(g.tag) and (g.tag != "new").any():
            keep.append(dict(market=mk, month=mo, price=g.loc[g.tag != "new", "price_NGN_per_kg"].mean()))
    return pd.DataFrame(keep).pivot(index="month", columns="market", values="price") if keep else pd.DataFrame()
