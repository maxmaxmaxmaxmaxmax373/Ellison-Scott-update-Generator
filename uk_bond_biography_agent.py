#!/usr/bin/env python3
"""
UK Gilt Biography Agent
=======================
Generates "bond biography" chapters (Jupyter notebooks) for British government
securities (gilts) from the Ellison-Scott UK gilt database, UK_Gilts_Bond_Database.xlsx:
502 stocks, month-end amounts outstanding 1946-2023 and month-end prices 1975-2023.

It is the UK counterpart of the Hall-Payne-Sargent US bond biography generator, with
gilt-specific additions:
  * yields computed the way the gilt market quotes them: gross redemption yield (to the
    earliest date for a double-dated stock priced at or above par), flat yield for undated
    stocks, real yield for index-linked stocks (old style with an 8-month lag, new style
    with a 3-month lag);
  * relative value against gilts of similar remaining life, and the stock's position on
    the yield curve (with a fitted Nelson-Siegel curve) at three dates;
  * total returns to holders in nominal and real (RPI-deflated) terms, risk and drawdown;
  * taps, tranche amalgamations, conversions and buy-backs detected from the amounts;
  * a British historical knowledge base (wars, sterling crises, Bank Rate, debt
    management reforms, prime ministers) and explanations of UK instrument features:
    double-dated, undated, partly paid, tranches, convertible, index-linked, variable and
    floating rate, nationalisation compensation stock, death-duty stock, green gilt.

Every chapter is self-contained: the notebook carries the helper functions and reads the
Excel database from its own folder or a parent folder (or from $UK_GILTS_DB).

Usage:
    python uk_bond_biography_agent.py --bond-id 32400
    python uk_bond_biography_agent.py --bond-id 32400 20100 55500 --execute
    python uk_bond_biography_agent.py --search "War Loan"
    python uk_bond_biography_agent.py --search "Green" --generate-first
    python uk_bond_biography_agent.py --list-categories
    python uk_bond_biography_agent.py --list-candidates 30
    python uk_bond_biography_agent.py --bond-id 32400 --print-prompt
    python uk_bond_biography_agent.py --all --min-months 120
    python uk_bond_biography_agent.py --interactive

Requirements:
    pip install pandas numpy matplotlib openpyxl nbformat nbconvert ipykernel
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


# ─── Shared helpers (also written into every chapter notebook) ─────────────────

HELPERS_SRC = r'''
# ── UK gilt helpers (shared by the generator and every chapter notebook) ──────
import os
import re
import calendar
from pathlib import Path
import numpy as np
import pandas as pd

DB_FILENAME = "UK_Gilts_Bond_Database.xlsx"


def locate_database(start=None):
    """Find the database: $UK_GILTS_DB, else the current folder or up to three parents."""
    env = os.environ.get("UK_GILTS_DB")
    if env and Path(env).exists():
        return Path(env)
    here = Path(start or Path.cwd()).resolve()
    for d in [here, *here.parents][:4]:
        if (d / DB_FILENAME).exists():
            return d / DB_FILENAME
    raise FileNotFoundError(f"{DB_FILENAME} not found; set the UK_GILTS_DB environment variable")


def _wide(df):
    out = df.set_index(["L1 ID", "Series"]).T
    out.index = pd.to_datetime(out.index)
    return out.apply(pd.to_numeric, errors="coerce")


def load_database(path=None):
    """Load BondList, prices, quantities, RPI and the correction log from the Excel database."""
    path = Path(path) if path else locate_database()
    sh = pd.read_excel(path, sheet_name=["BondList", "BondPrice", "BondQuant", "RPI", "Corrections"])
    rpi = sh["RPI"]
    rpi = pd.Series(rpi.iloc[:, 1].values, index=pd.to_datetime(rpi.iloc[:, 0]) + pd.offsets.MonthEnd(0))
    return {"list": sh["BondList"].set_index("L1 ID"), "price": _wide(sh["BondPrice"]),
            "quant": _wide(sh["BondQuant"]), "rpi": rpi, "corrections": sh["Corrections"], "path": path}


def series(db, kind, bond_id, name):
    """kind = 'price' or 'quant'; name = 'Average', 'Partly Paid', 'Total Outstanding', 'Indexed Outstanding'."""
    df = db[kind]
    if (bond_id, name) not in df.columns:
        return pd.Series(dtype=float)
    return df[(bond_id, name)].dropna()


# Before 28 February 1986 (when the Accrued Income Scheme began) stocks with more than five years to final
# maturity, and undated stocks, were quoted with accrued interest included ("dirty"): their prices drop by about
# half a coupon when they go ex-dividend. Shorter stocks were already quoted clean, as all stocks are since.
DIRTY_BEFORE = pd.Timestamp("1986-02-28")
DIRTY_MIN_YEARS = 5.0
EX_DIV_DAYS = 35          # ex-dividend about five weeks before each coupon (35 days before the quote date fits best)
_MONTHS = {m: k for k, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                                       "Nov", "Dec"], 1)}


def coupon_schedule(row, start, end):
    """Coupon dates around [start, end]: from 'Payment Dates' (e.g. '1 Mar / 1 Sep'), else back from the final date."""
    dates = []
    txt = row.get("Payment Dates")
    if isinstance(txt, str):
        for part in txt.split("/"):
            bits = part.split()
            if len(bits) == 2 and bits[0].isdigit() and bits[1][:3] in _MONTHS:
                m = _MONTHS[bits[1][:3]]
                for y in range(start.year - 1, end.year + 2):
                    dates.append(pd.Timestamp(y, m, min(int(bits[0]), calendar.monthrange(y, m)[1])))
    if not dates and pd.notna(row.get("Payable Date")):
        step = 12 // (int(row["Coupons Per Year"]) if pd.notna(row.get("Coupons Per Year")) else 2)
        mat, k = pd.Timestamp(row["Payable Date"]), 0
        while mat - pd.DateOffset(months=step * k) >= start - pd.DateOffset(years=1):
            dates.append(mat - pd.DateOffset(months=step * k))
            k += 1
    return sorted(set(dates))


def clean_price(db, bond_id):
    """Fully-paid month-end price on a clean basis. Dirty source prices (before 28 Feb 1986, more than five years
    to final maturity or undated) are converted: clean = dirty - accrued interest when cum-dividend, and
    dirty + rebate interest when ex-dividend. Index-linked coupons are uplifted by the index ratio."""
    row = db["list"].loc[bond_id]
    p = series(db, "price", bond_id, "Average")
    c = row["Coupon Rate"]
    if p.empty or pd.isna(c):
        return p
    old = np.asarray(p.index < DIRTY_BEFORE)
    if pd.notna(row.get("Payable Date")):
        old &= np.asarray((pd.Timestamp(row["Payable Date"]) - p.index).days / 365.25 > DIRTY_MIN_YEARS)
    if not old.any():
        return p
    cds = coupon_schedule(row, p.index[old].min(), DIRTY_BEFORE)
    if len(cds) < 2:
        return p
    cdn = np.array([x.value for x in cds], dtype=np.int64)
    quote = p.index[old].map(pd.offsets.BMonthEnd().rollback)   # prices are taken on the last business day
    d = pd.DatetimeIndex(quote).values.astype("datetime64[ns]").astype(np.int64)
    nx = np.clip(np.searchsorted(cdn, d, side="right"), 1, len(cdn) - 1)
    prev, nxt = cdn[nx - 1], cdn[nx]
    a = (d - prev) / (nxt - prev)
    freq = int(row["Coupons Per Year"]) if pd.notna(row.get("Coupons Per Year")) else 2
    cpn = np.full(len(d), c / freq)
    if row["Category L1"] == "Index-linked":
        cpn = cpn * index_ratio(db, bond_id, p.index[old]).values
    ex = d > nxt - EX_DIV_DAYS * 86400 * 10**9
    adj = np.where(ex, cpn * (1 - a), -cpn * a)
    out = p.copy()
    out.iloc[np.where(old)[0]] = p.values[old] + np.nan_to_num(adj)
    return out


def is_old_style_il(row, bond_id):
    return row["Category L1"] == "Index-linked" and bond_id < 55000


def index_ratio(db, bond_id, dates):
    """RPI uplift since the base date: RPI(m-8)/base for old-style, reference RPI/base for new-style."""
    row = db["list"].loc[bond_id]
    base, rpi = row["Base RPI"], db["rpi"]
    dates = pd.DatetimeIndex(dates)
    if is_old_style_il(row, bond_id):
        ref = rpi.reindex(dates - pd.offsets.MonthEnd(8)).values
    else:
        a = rpi.reindex(dates - pd.offsets.MonthEnd(3)).values
        b = rpi.reindex(dates - pd.offsets.MonthEnd(2)).values
        frac = (dates.day - 1) / dates.days_in_month
        ref = a + frac * (b - a)
    return pd.Series(np.asarray(ref, float) / base, index=dates)


def _yield_to(dates, prices, coupon, freq, maturity):
    """Gross redemption yield (% a year, compounded `freq` times) for clean prices on `dates`,
    with accrued interest; returns (yield %, modified duration in years)."""
    dates = pd.DatetimeIndex(dates)
    step = 12 // freq
    cds = [pd.Timestamp(maturity)]
    while cds[-1] > dates.min():
        cds.append(pd.Timestamp(maturity) - pd.DateOffset(months=step * len(cds)))
    cdn = np.array(sorted(c.value for c in cds), float) / 86400e9
    d = dates.values.astype("datetime64[ns]").astype(np.int64) / 86400e9
    nxt = np.searchsorted(cdn, d, side="right")
    ok = (nxt > 0) & (nxt < len(cdn))
    y_out = np.full(len(d), np.nan); dur_out = np.full(len(d), np.nan)
    if not ok.any():
        return y_out, dur_out
    d, p, nx = d[ok], np.asarray(prices, float)[ok], nxt[ok]
    prev, nextc = cdn[nx - 1], cdn[nx]
    w = (nextc - d) / (nextc - prev)
    n = len(cdn) - nx
    dirty = p + coupon / freq * (1 - w)
    K = int(n.max()); k = np.arange(K)
    t = w[:, None] + k[None, :]
    cf = np.where(k[None, :] < n[:, None], coupon / freq, 0.0)
    cf[np.arange(len(n)), n - 1] += 100.0
    y = np.full(len(d), max(coupon / 100.0, 0.03))
    for _ in range(60):
        v = 1.0 / (1.0 + y / freq)
        disc = v[:, None] ** t
        pv = (cf * disc).sum(1)
        dpv = -(cf * disc * t).sum(1) * v / freq
        stepv = (pv - dirty) / dpv
        y = np.clip(y - stepv, -0.2, 3.0)
        if np.nanmax(np.abs(stepv)) < 1e-11:
            break
    v = 1.0 / (1.0 + y / freq); disc = v[:, None] ** t
    mac = (cf * disc * t).sum(1) / (cf * disc).sum(1) / freq
    y_out[ok] = y * 100; dur_out[ok] = mac / (1 + y / freq)
    return y_out, dur_out


IL_INFLATION = 3.0      # % a year: assumed future RPI inflation for old-style index-linked real yields


def old_style_il_yields(db, bond_id, px, infl=IL_INFLATION):
    """Real yields (%) and modified durations of an old-style index-linked gilt from clean nominal prices `px`.
    Each cash flow is indexed to the RPI eight months before it is paid; RPI values not yet published at the
    price date are projected at `infl`% a year from the latest one. The nominal redemption yield is then
    converted to a real yield: (1 + r/f) = (1 + y/f) / (1 + infl)^(1/f)."""
    row = db["list"].loc[bond_id]
    rpi, base, c = db["rpi"], row["Base RPI"], row["Coupon Rate"]
    freq = int(row["Coupons Per Year"]) if pd.notna(row["Coupons Per Year"]) else 2
    mat = pd.Timestamp(row["Payable Date"])
    cds = [mat]
    while cds[-1] > px.index.min():
        cds.append(mat - pd.DateOffset(months=12 // freq * len(cds)))
    cds = sorted(cds)
    g = (1 + infl / 100) ** (1 / 12)
    ys, durs = np.full(len(px), np.nan), np.full(len(px), np.nan)
    for k, (t, P) in enumerate(px.items()):
        known = rpi.loc[:t - pd.offsets.MonthEnd(1)].dropna()     # RPI for month m is published in month m+1
        fut = [d for d in cds if d > t]
        past = [d for d in cds if d <= t]
        if known.empty or not fut or not past or pd.isna(P):
            continue
        last_m, last_v = known.index[-1], known.iloc[-1]

        def ref(d):
            m = d + pd.offsets.MonthEnd(0) - pd.offsets.MonthEnd(8)
            if m <= last_m:
                return rpi.get(m, np.nan)
            return last_v * g ** ((m.year - last_m.year) * 12 + m.month - last_m.month)
        idx = np.array([ref(d) for d in fut]) / base
        if np.isnan(idx).any():
            continue
        cf = c / freq * idx
        cf[-1] += 100 * idx[-1]
        w = (fut[0] - t).days / (fut[0] - past[-1]).days
        dirty = P + c / freq * idx[0] * (1 - w)
        tt = w + np.arange(len(fut))
        y = 0.05
        for _ in range(60):
            v = 1 / (1 + y / freq)
            pv = (cf * v ** tt).sum()
            dpv = -(cf * v ** tt * tt).sum() * v / freq
            step = (pv - dirty) / dpv
            y = min(max(y - step, -0.5), 3.0)
            if abs(step) < 1e-11:
                break
        v = 1 / (1 + y / freq)
        durs[k] = (cf * v ** tt * tt).sum() / (cf * v ** tt).sum() / freq / (1 + y / freq)
        ys[k] = ((1 + y / freq) / (1 + infl / 100) ** (1 / freq) - 1) * freq * 100
    return ys, durs


def bond_yields(db, bond_id, min_years=0.25):
    """Monthly yields for one bond from clean prices (see clean_price). Columns: price, yield (%), kind,
    maturity_used, remaining (years), duration.
    Dated conventional: gross redemption yield; a double-dated stock is assumed redeemed at the earliest date when
    priced at or above par and at the final date otherwise. Undated: flat (running) yield. Index-linked: real yield
    (new style: from the real price; old style: see old_style_il_yields, which assumes 3% future inflation).
    Variable/floating-rate stocks: no yield. Yields within `min_years` of redemption are dropped (too noisy)."""
    row = db["list"].loc[bond_id]
    p = clean_price(db, bond_id)
    out = pd.DataFrame({"price": p})
    c = row["Coupon Rate"]
    if p.empty or row["Category L1"] == "Variable/Floating Rate" or pd.isna(c):
        out["yield"] = np.nan; out["kind"] = "none"
        return out
    freq = int(row["Coupons Per Year"]) if pd.notna(row["Coupons Per Year"]) else 2
    if row["Category L1"] == "Undated" or pd.isna(row["Payable Date"]):
        out["yield"] = c / p * 100; out["kind"] = "flat"
        out["remaining"] = np.nan; out["duration"] = 100 / out["yield"]
        m = re.search(r"announced (\d+)/(\d+)/(\d{4})", str(row.get("Special Features", "")))
        if m:   # once redemption was announced the stock stopped behaving like a perpetuity
            announced = pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
            out.loc[out.index >= announced, ["yield", "duration"]] = np.nan
        return out
    px = p.copy(); kind = "gross redemption"
    final, early = pd.Timestamp(row["Payable Date"]), row["Redeemable After Date"]
    if row["Category L1"] == "Index-linked":
        kind = "real"
        if is_old_style_il(row, bond_id):
            y, dur = old_style_il_yields(db, bond_id, p)
            out["yield"], out["kind"], out["maturity_used"] = y, kind, final
            out["remaining"] = (final - out.index).days / 365.25
            out["duration"] = dur
            out.loc[out["remaining"] < min_years, ["yield", "duration"]] = np.nan
            return out
    y, dur = _yield_to(p.index, px.values, c, freq, final)
    mat = np.full(len(p), final.value, dtype=np.int64)
    if pd.notna(early):
        early = pd.Timestamp(early)
        ye, de = _yield_to(p.index, px.values, c, freq, early)
        use = (px.values >= 100) & (p.index < early)
        y = np.where(use, ye, y); dur = np.where(use, de, dur)
        mat = np.where(use, early.value, mat)
    out["yield"] = y; out["kind"] = kind
    out["maturity_used"] = pd.to_datetime(mat)
    out["remaining"] = (out["maturity_used"] - out.index).dt.days / 365.25
    out["duration"] = dur
    out.loc[out["remaining"] < min_years, ["yield", "duration"]] = np.nan
    return out


def yield_panel(db, categories=("Conventional",)):
    """Yields for every bond in the given Category L1 groups: {bond_id: bond_yields DataFrame}."""
    bl = db["list"]; panel = {}
    for i in bl.index[bl["Category L1"].isin(categories)]:
        y = bond_yields(db, i)
        if "yield" in y and y["yield"].notna().any():
            panel[i] = y
    return panel


def peer_stats(panel, bond_id, own, band=0.15, min_band=1.0, min_peers=3, long_only=None):
    """Each month: median and inter-quartile range of peer yields. Peers have remaining life within
    max(min_band, band x own remaining life) years of this bond; for undated stocks (long_only=years)
    peers are all bonds with at least that remaining life."""
    frames = [d[["yield", "remaining"]].assign(id=i) for i, d in panel.items() if i != bond_id]
    if not frames:
        return pd.DataFrame(columns=["median", "q25", "q75", "n"])
    stack = pd.concat(frames).dropna()
    groups = dict(tuple(stack.groupby(level=0)))
    rows = {}
    for date, r in own.dropna(subset=["yield"]).iterrows():
        g = groups.get(date)
        if g is None:
            continue
        if long_only is not None:
            g = g[g["remaining"] >= long_only]
        else:
            T = r["remaining"]
            if pd.isna(T):
                continue
            g = g[(g["remaining"] - T).abs() <= max(min_band, band * T)]
        if len(g) >= min_peers:
            rows[date] = {"median": g["yield"].median(), "q25": g["yield"].quantile(.25),
                          "q75": g["yield"].quantile(.75), "n": len(g)}
    return pd.DataFrame.from_dict(rows, orient="index")


def total_return_index(db, bond_id, start=100.0):
    """Approximate total-return index (clean price change plus accrued coupon) from the first fully-paid price.
    Index-linked coupons (and new-style prices) are uplifted by the index ratio. Returns DataFrame
    with columns nominal and, where RPI is available, real (deflated by RPI)."""
    row = db["list"].loc[bond_id]
    p = clean_price(db, bond_id)
    if len(p) < 2:
        return pd.DataFrame()
    c = row["Coupon Rate"] if pd.notna(row["Coupon Rate"]) else 0.0
    irc = pd.Series(1.0, index=p.index); irp = pd.Series(1.0, index=p.index)
    if row["Category L1"] == "Index-linked":
        irc = index_ratio(db, bond_id, p.index)
        if not is_old_style_il(row, bond_id):
            irp = irc
    value = p * irp
    months = (p.index.year * 12 + p.index.month).to_series(index=p.index).diff()
    r = (value.diff() + c / 12 * irc * months) / value.shift(1)
    idx = start * (1 + r.fillna(0)).cumprod()
    out = pd.DataFrame({"nominal": idx})
    rpi = db["rpi"].reindex(p.index)
    if rpi.notna().any():
        first = rpi.first_valid_index()
        out["real"] = out["nominal"] / out.loc[first, "nominal"] * start / (rpi / rpi.loc[first])
    return out


def gilt_stock_total(db):
    """Total nominal outstanding of all bonds in the database, by month (£)."""
    q = db["quant"]
    cols = [c for c in q.columns if c[1] == "Total Outstanding"]
    return q[cols].sum(axis=1, min_count=1)


def nelson_siegel(T, y, taus=np.linspace(0.5, 15, 60)):
    """Fit y(T) = b0 + b1*f1(T/tau) + b2*(f1(T/tau) - exp(-T/tau)), f1(x) = (1 - exp(-x))/x, by least squares
    over a grid of tau. Returns a function of T, or None with fewer than 5 points."""
    T = np.asarray(T, float); y = np.asarray(y, float)
    if len(T) < 5:
        return None
    best = None
    for tau in taus:
        x = T / tau
        f1 = (1 - np.exp(-x)) / x
        X = np.column_stack([np.ones_like(T), f1, f1 - np.exp(-x)])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        sse = ((X @ b - y) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, tau, b)
    _, tau, b = best

    def curve(t):
        x = np.maximum(np.asarray(t, float), 1e-6) / tau
        f1 = (1 - np.exp(-x)) / x
        return b[0] + b[1] * f1 + b[2] * (f1 - np.exp(-x))
    return curve
'''

exec(HELPERS_SRC, globals())

DATA_END = pd.Timestamp("2023-12-31")   # last month in the database
NAME_COL = "Treasury's Name Of Issue"


# ─── Historical Knowledge Base ─────────────────────────────────────────────────

def _e(start, end, name, category, key=False, month=False):
    """One event. `month=True` means only the month is certain; `key` events are drawn on charts."""
    return {"start": start, "end": end or start, "name": name, "category": category,
            "key": key, "month": month}


UK_EVENTS = [
    # Wars
    _e("1899-10-11", "1902-05-31", "Second Boer War", "war", key=True),
    _e("1914-08-04", "1918-11-11", "First World War", "war", key=True),
    _e("1939-09-03", "1945-08-15", "Second World War", "war", key=True),
    _e("1950-06-25", "1953-07-27", "Korean War and British rearmament", "war"),
    _e("1956-10-29", "1956-11-07", "Suez crisis and run on sterling", "crisis", key=True),
    _e("1982-04-02", "1982-06-14", "Falklands War", "war", key=True),
    _e("1990-08-02", "1991-02-28", "Iraqi invasion of Kuwait and Gulf War", "war"),
    _e("2022-02-24", None, "Russian invasion of Ukraine; energy-price shock", "war", key=True),
    # Gold standard, sterling and the interwar years
    _e("1888-03-01", None, "Goschen's conversion of the 3% Consols", "debt management", month=True),
    _e("1914-07-31", "1915-01-04", "London Stock Exchange closed", "crisis"),
    _e("1914-11-01", None, "First War Loan (3½%)", "debt management", month=True),
    _e("1915-06-01", None, "Second War Loan (4½%)", "debt management", month=True),
    _e("1917-01-01", "1917-02-28", "Great War Loan: 5% War Loan 1929-47", "debt management", key=True, month=True),
    _e("1919-06-01", "1919-07-31", "Victory Loan: 4% Victory Bonds and 4% Funding Loan", "debt management", month=True),
    _e("1925-04-28", None, "Return to gold at the pre-war parity", "monetary", key=True),
    _e("1926-05-04", "1926-05-12", "General Strike", "crisis"),
    _e("1929-10-24", "1929-10-29", "Wall Street Crash", "crisis"),
    _e("1931-07-01", "1931-09-20", "Sterling crisis: May Report and National Government", "crisis", month=True),
    _e("1931-09-21", None, "Britain leaves the gold standard", "monetary", key=True),
    _e("1932-06-30", None, "Bank Rate cut to 2%; 5% War Loan conversion announced", "debt management", key=True),
    # War and post-war finance
    _e("1944-07-01", "1944-07-22", "Bretton Woods conference", "monetary"),
    _e("1945-12-06", None, "Anglo-American Loan agreement signed", "fiscal"),
    _e("1946-03-01", None, "Bank of England nationalised", "monetary", key=True),
    _e("1946-10-01", "1947-01-31", "Dalton's 2½% undated issue: the peak of cheap money", "debt management", month=True),
    _e("1947-07-15", "1947-08-20", "Sterling convertibility crisis", "crisis", key=True),
    _e("1948-01-01", None, "Inland transport nationalised (British Transport Stock)", "nationalisation"),
    _e("1948-04-01", None, "Electricity nationalised (British Electricity Stock)", "nationalisation"),
    _e("1949-05-01", None, "Gas nationalised (Gas Stock)", "nationalisation"),
    _e("1949-09-18", None, "Sterling devalued from $4.03 to $2.80", "crisis", key=True),
    _e("1951-11-01", None, "End of cheap money: Bank Rate raised; Treasury bills funded", "monetary", key=True, month=True),
    # Bretton Woods and stop-go
    _e("1957-09-19", None, "Bank Rate raised to 7%", "monetary", key=True),
    _e("1959-08-01", None, "Radcliffe Report on the monetary system", "monetary", month=True),
    _e("1964-11-23", None, "Sterling crisis: Bank Rate raised to 7%", "crisis"),
    _e("1967-11-18", None, "Sterling devalued from $2.80 to $2.40", "crisis", key=True),
    _e("1971-08-15", None, "US closes the gold window", "monetary"),
    _e("1971-09-16", None, "Competition and Credit Control", "monetary", key=True),
    _e("1972-06-23", None, "Sterling floated", "monetary", key=True),
    _e("1972-10-13", None, "Bank Rate replaced by Minimum Lending Rate", "monetary"),
    # The Great Inflation
    _e("1973-10-17", "1974-03-31", "First oil shock", "crisis", key=True),
    _e("1973-12-01", "1975-12-31", "Secondary banking crisis ('Lifeboat')", "crisis", month=True),
    _e("1974-01-01", "1974-03-07", "Three-Day Week", "crisis"),
    _e("1975-08-01", None, "RPI inflation peaks near 27%", "crisis", key=True, month=True),
    _e("1976-09-28", "1976-12-15", "IMF crisis", "crisis", key=True),
    _e("1977-01-01", "1977-10-31", "Minimum Lending Rate cut from 15% to 5%", "monetary", month=True),
    _e("1977-05-27", None, "First variable-rate gilt issued", "debt management"),
    _e("1978-11-01", "1979-02-28", "Winter of Discontent", "crisis", key=True, month=True),
    # Monetarism and deregulation
    _e("1979-10-23", None, "Exchange controls abolished", "monetary", key=True),
    _e("1979-11-15", None, "Minimum Lending Rate raised to 17%", "monetary", key=True),
    _e("1980-03-26", None, "Medium-Term Financial Strategy", "fiscal"),
    _e("1981-03-10", None, "Contractionary 1981 Budget", "fiscal"),
    _e("1981-03-27", None, "First index-linked gilt issued", "debt management", key=True),
    _e("1982-03-01", None, "Index-linked gilts opened to all investors", "debt management", month=True),
    _e("1985-01-01", None, "Sterling crisis: interest rates raised sharply", "crisis", month=True),
    _e("1985-10-17", None, "Mansion House speech: overfunding ended", "monetary"),
    _e("1986-01-01", "1986-04-30", "Oil-price collapse", "crisis", month=True),
    _e("1986-10-27", None, "Big Bang: gilt-edged market makers replace jobbers", "market", key=True),
    _e("1987-04-01", "1990-03-31", "Public sector debt repayment", "fiscal", month=True),
    _e("1987-05-01", None, "First gilt auction", "debt management", month=True),
    _e("1987-10-19", None, "Black Monday stock-market crash", "crisis", key=True),
    _e("1989-10-05", None, "Base rates raised to 15%", "monetary"),
    _e("1990-10-08", None, "Sterling joins the ERM", "monetary", key=True),
    _e("1992-09-16", None, "Black Wednesday: sterling leaves the ERM", "crisis", key=True),
    # Inflation targeting
    _e("1992-10-08", None, "Inflation targeting adopted", "monetary", key=True),
    _e("1994-02-04", "1994-12-31", "Global bond-market sell-off", "crisis", key=True),
    _e("1995-07-01", None, "Debt Management Review report", "debt management", month=True),
    _e("1996-01-02", None, "Open gilt repo market", "market"),
    _e("1997-04-06", None, "Minimum Funding Requirement for pension funds", "market"),
    _e("1997-05-06", None, "Bank of England operational independence", "monetary", key=True),
    _e("1997-12-08", None, "Gilt strips market opens", "market"),
    _e("1998-04-01", None, "Debt Management Office takes over gilt issuance", "debt management", key=True),
    _e("1998-08-17", "1998-10-15", "Russian default and LTCM", "crisis", key=True),
    _e("1999-01-01", None, "Euro launched", "monetary"),
    _e("2001-09-11", None, "11 September attacks", "crisis"),
    _e("2003-03-20", None, "Iraq War begins", "war"),
    _e("2003-12-10", None, "Inflation target switched from RPIX to CPI", "monetary"),
    _e("2005-05-01", None, "First 50-year gilt (4¼% Treasury Gilt 2055)", "debt management", month=True),
    _e("2005-09-01", None, "First syndicated gilt; first new-style index-linked gilt", "debt management", month=True),
    # Crisis, QE and after
    _e("2007-08-09", None, "Interbank markets freeze", "crisis"),
    _e("2007-09-14", None, "Run on Northern Rock", "crisis", key=True),
    _e("2008-09-15", None, "Lehman Brothers collapses", "crisis", key=True),
    _e("2008-10-08", None, "UK bank rescue package; coordinated rate cuts", "crisis"),
    _e("2008-11-06", None, "Bank Rate cut from 4.5% to 3%", "monetary"),
    _e("2009-01-19", None, "Second UK bank rescue package", "crisis"),
    _e("2009-03-05", None, "Bank Rate 0.5%; quantitative easing begins", "monetary", key=True),
    _e("2009-08-06", None, "QE extended to £175bn", "monetary"),
    _e("2010-05-01", "2012-09-06", "Euro-area sovereign debt crisis", "crisis", month=True),
    _e("2010-06-22", None, "Emergency Budget: fiscal consolidation", "fiscal"),
    _e("2011-10-06", None, "Second round of quantitative easing", "monetary"),
    _e("2013-08-07", None, "Forward guidance introduced", "monetary"),
    _e("2014-10-31", "2015-07-05", "Redemption of the undated gilts", "debt management", key=True),
    _e("2016-06-23", None, "EU referendum", "political", key=True),
    _e("2016-08-04", None, "Bank Rate cut to 0.25%; QE restarted", "monetary"),
    _e("2017-11-02", None, "First Bank Rate rise in a decade", "monetary"),
    _e("2020-01-31", None, "UK leaves the European Union", "political"),
    _e("2020-03-11", "2020-03-19", "COVID-19: Bank Rate to 0.1%, £200bn QE", "crisis", key=True),
    _e("2020-11-05", None, "Asset purchases raised to £895bn", "monetary"),
    _e("2021-09-21", None, "First green gilt syndicated", "debt management"),
    _e("2021-12-16", None, "Bank Rate starts to rise", "monetary", key=True),
    _e("2022-02-03", None, "Quantitative tightening begins", "monetary"),
    _e("2022-09-23", "2022-10-14", "Mini-budget and LDI crisis", "crisis", key=True),
    _e("2022-10-01", None, "CPI inflation peaks at 11.1%", "crisis", month=True),
    _e("2022-11-01", None, "Active gilt sales (QT) begin", "monetary"),
    _e("2023-08-03", None, "Bank Rate peaks at 5.25%", "monetary", key=True),
]

# Prime ministers: (start, name, party). Each term ends when the next begins.
GOVERNMENTS = [
    ("1895-06-25", "Salisbury", "Conservative"), ("1902-07-11", "Balfour", "Conservative"),
    ("1905-12-05", "Campbell-Bannerman", "Liberal"), ("1908-04-05", "Asquith", "Liberal"),
    ("1915-05-25", "Asquith", "Coalition"), ("1916-12-06", "Lloyd George", "Coalition"),
    ("1922-10-23", "Bonar Law", "Conservative"), ("1923-05-22", "Baldwin", "Conservative"),
    ("1924-01-22", "MacDonald", "Labour"), ("1924-11-04", "Baldwin", "Conservative"),
    ("1929-06-05", "MacDonald", "Labour"), ("1931-08-24", "MacDonald", "National"),
    ("1935-06-07", "Baldwin", "National"), ("1937-05-28", "Chamberlain", "National"),
    ("1940-05-10", "Churchill", "Coalition"), ("1945-05-23", "Churchill", "Conservative"),
    ("1945-07-26", "Attlee", "Labour"), ("1951-10-26", "Churchill", "Conservative"),
    ("1955-04-06", "Eden", "Conservative"), ("1957-01-10", "Macmillan", "Conservative"),
    ("1963-10-19", "Douglas-Home", "Conservative"), ("1964-10-16", "Wilson", "Labour"),
    ("1970-06-19", "Heath", "Conservative"), ("1974-03-04", "Wilson", "Labour"),
    ("1976-04-05", "Callaghan", "Labour"), ("1979-05-04", "Thatcher", "Conservative"),
    ("1990-11-28", "Major", "Conservative"), ("1997-05-02", "Blair", "Labour"),
    ("2007-06-27", "Brown", "Labour"), ("2010-05-11", "Cameron", "Con-LD coalition"),
    ("2015-05-08", "Cameron", "Conservative"), ("2016-07-13", "May", "Conservative"),
    ("2019-07-24", "Johnson", "Conservative"), ("2022-09-06", "Truss", "Conservative"),
    ("2022-10-25", "Sunak", "Conservative"),
]
GOV_END = "2024-07-05"

ERAS = [
    ("1888-01-01", "1914-07-31", "Late-Victorian and Edwardian calm",
     "Consols yields were near historic lows under the gold standard. Goschen's 1888 conversion cut the "
     "coupon on the bulk of the debt, and the Boer War (1899-1902) was financed partly by new borrowing."),
    ("1914-08-01", "1918-12-31", "First World War finance",
     "The national debt rose more than tenfold, financed by successive War Loans (1914, 1915 and the great "
     "5% War Loan of 1917), Exchequer Bonds, National War Bonds and Treasury bills."),
    ("1919-01-01", "1932-06-29", "Post-war debt overhang and the gold standard",
     "The Victory Loan of 1919 began the long effort to fund the floating debt. Returning to gold at the "
     "pre-war parity in 1925 required tight money while debt stood well above 150% of GDP; the 1931 crisis "
     "forced sterling off gold."),
    ("1932-06-30", "1939-08-31", "Cheap money and the War Loan conversion",
     "Bank Rate was cut to 2% on 30 June 1932, the day the conversion of the 5% War Loan into a 3½% stock was "
     "announced. Lower debt interest and cheap money supported the recovery of the 1930s."),
    ("1939-09-01", "1945-08-31", "Second World War: the 'three per cent war'",
     "The Treasury held long-term borrowing costs at about 3% through continuous tap issues of National War "
     "Bonds, Savings Bonds and Defence Bonds, backed by exchange control, capital-issues control and rationing."),
    ("1945-09-01", "1951-10-31", "Post-war cheap money and nationalisation",
     "Dalton pushed long yields towards 2½% before the policy collapsed in 1947. Owners of nationalised "
     "industries were compensated with government-guaranteed stocks, and sterling was devalued in 1949."),
    ("1951-11-01", "1971-09-15", "Bretton Woods, Bank Rate and stop-go",
     "Monetary policy was revived in November 1951. Bank Rate and gilt yields moved with repeated sterling "
     "crises, culminating in the 1967 devaluation; the Bank of England sold gilts through its tap system and "
     "'leaned into the wind' in the market."),
    ("1971-09-16", "1979-05-03", "Competition and Credit Control and the Great Inflation",
     "Competition and Credit Control liberalised bank lending, sterling floated in 1972, and inflation peaked "
     "near 27% in 1975. Gilt yields reached the mid-teens and the 1976 IMF crisis followed; the authorities "
     "experimented with partly-paid, convertible and variable-rate stocks to sell debt."),
    ("1979-05-04", "1992-10-07", "Monetarism, deregulation and the ERM",
     "The Medium-Term Financial Strategy, the abolition of exchange controls, index-linked gilts (1981), the end "
     "of overfunding (1985), Big Bang (1986) and gilt auctions (1987) transformed the market; the government "
     "repaid debt in 1987-90 and sterling was in the ERM from 1990 to 1992."),
    ("1992-10-08", "2007-08-08", "Inflation targeting and the 'NICE' decade",
     "Inflation targeting (1992), Bank of England independence (1997), the Debt Management Office (1998) and "
     "the repo and strips markets modernised gilts. Pension regulation raised demand for long and index-linked "
     "gilts; 50-year and new-style index-linked gilts appeared in 2005."),
    ("2007-08-09", "2020-02-29", "Financial crisis, quantitative easing and ultra-low rates",
     "Bank Rate fell to 0.5% in March 2009 and the Bank of England bought gilts under quantitative easing while "
     "issuance surged with the deficit. The undated gilts were redeemed in 2015, and the Brexit vote brought "
     "a further cut in 2016."),
    ("2020-03-01", "2023-12-31", "Pandemic, inflation and quantitative tightening",
     "Pandemic borrowing was matched by asset purchases that reached £895bn. CPI inflation peaked at 11.1% in "
     "October 2022 and Bank Rate rose from 0.1% to 5.25%; the September 2022 mini-budget triggered the LDI "
     "crisis, and active gilt sales began that November."),
]

# Real, standard works. (citation, (from_year, to_year) coverage or None, feature tags that also select it)
LITERATURE = [
    ("Ellison, Martin and Andrew Scott (2020). \"Managing the UK National Debt 1694-2018.\" *American Economic "
     "Journal: Macroeconomics* 12(3): 227-257.", None, {"core"}),
    ("Kynaston, David (2017). *Till Time's Last Sand: A History of the Bank of England 1694-2013*. London: "
     "Bloomsbury.", None, {"core"}),
    ("Dimson, Elroy, Paul Marsh and Mike Staunton (2002). *Triumph of the Optimists: 101 Years of Global "
     "Investment Returns*. Princeton: Princeton University Press.", None, {"core"}),
    ("Hall, George J., Jonathan Payne and Thomas J. Sargent (2018). \"US Federal Debt 1776-1960: Quantities "
     "and Prices.\" Working paper (the US database behind the companion bond biographies).", None, {"core"}),
    ("UK Debt Management Office. *Formulae for Calculating Gilt Prices from Yields* (editions from 1998).",
     None, {"core"}),
    ("Pember & Boyle (1950). *British Government Securities in the Twentieth Century*, 2nd ed.; supplement "
     "1950-1976 (1976). London: privately printed.", (1900, 1976), set()),
    ("Sayers, R. S. (1976). *The Bank of England 1891-1944*. Cambridge: Cambridge University Press.",
     (1891, 1944), set()),
    ("Wormell, Jeremy (2000). *The Management of the National Debt of the United Kingdom, 1900-1932*. "
     "London: Routledge.", (1900, 1932), set()),
    ("Committee on National Debt and Taxation (Colwyn Committee) (1927). *Report*. Cmd 2800. London: HMSO.",
     (1919, 1932), set()),
    ("Howson, Susan (1975). *Domestic Monetary Management in Britain 1919-38*. Cambridge: Cambridge "
     "University Press.", (1919, 1938), set()),
    ("Cairncross, Alec and Barry Eichengreen (1983). *Sterling in Decline: The Devaluations of 1931, 1949 "
     "and 1967*. Oxford: Blackwell.", (1931, 1967), set()),
    ("Howson, Susan (1993). *British Monetary Policy 1945-51*. Oxford: Clarendon Press.", (1945, 1951), set()),
    ("Cairncross, Alec (1985). *Years of Recovery: British Economic Policy 1945-51*. London: Methuen.",
     (1945, 1951), set()),
    ("Chester, Norman (1975). *The Nationalisation of British Industry 1945-51*. London: HMSO.",
     None, {"nationalisation", "airline"}),
    ("Allen, William A. (2019). *The Bank of England and the Government Debt: Operations in the Gilt-Edged "
     "Market, 1928-1972*. Cambridge: Cambridge University Press.", (1928, 1972), set()),
    ("Allen, William A. (2014). *Monetary Policy and Financial Repression in Britain, 1951-59*. Basingstoke: "
     "Palgrave Macmillan.", (1951, 1959), {"serial-funding"}),
    ("Committee on the Working of the Monetary System (Radcliffe Committee) (1959). *Report*. Cmnd 827. "
     "London: HMSO.", (1951, 1960), set()),
    ("Reinhart, Carmen M. and M. Belen Sbrancia (2015). \"The Liquidation of Government Debt.\" *Economic "
     "Policy* 30(82): 291-333.", (1945, 1980), set()),
    ("Capie, Forrest (2010). *The Bank of England: 1950s to 1979*. Cambridge: Cambridge University Press.",
     (1950, 1979), set()),
    ("Needham, Duncan (2014). *UK Monetary Policy from Devaluation to Thatcher, 1967-82*. Basingstoke: "
     "Palgrave Macmillan.", (1967, 1982), set()),
    ("Schaefer, Stephen M. (1981). \"Measuring a Tax-Specific Term Structure of Interest Rates in the Market "
     "for British Government Securities.\" *Economic Journal* 91(362): 415-438.", None, {"low-coupon"}),
    ("Campbell, John Y. and Robert J. Shiller (1996). \"A Scorecard for Indexed Government Debt.\" *NBER "
     "Macroeconomics Annual* 11: 155-197.", None, {"il-old", "il-new"}),
    ("Barr, David G. and John Y. Campbell (1997). \"Inflation, Real Interest Rates, and the Bond Market: A "
     "Study of UK Nominal and Index-Linked Government Bond Prices.\" *Journal of Monetary Economics* 39(3): "
     "361-383.", None, {"il-old", "il-new"}),
    ("Deacon, Mark, Andrew Derry and Dariush Mirfendereski (2004). *Inflation-Indexed Securities: Bonds, "
     "Swaps and Other Derivatives*, 2nd ed. Chichester: Wiley.", None, {"il-old", "il-new"}),
    ("HM Treasury and Bank of England (1995). *Report of the Debt Management Review*. London: HM Treasury.",
     (1993, 2023), set()),
    ("UK Debt Management Office. *Gilt Market: Annual Review* (annual, from 1998-99).", (1998, 2023), set()),
    ("Greenwood, Robin and Dimitri Vayanos (2010). \"Price Pressure in the Government Bond Market.\" "
     "*American Economic Review* 100(2): 585-590.", None, {"long-2004"}),
    ("Joyce, Michael, Ana Lasaosa, Ibrahim Stevens and Matthew Tong (2011). \"The Financial Market Impact of "
     "Quantitative Easing in the United Kingdom.\" *International Journal of Central Banking* 7(3): 113-161.",
     (2009, 2023), set()),
    ("HM Treasury (2021). *UK Government Green Financing Framework*. London: HM Treasury.", None, {"green"}),
    ("Bank of England (2022). *Financial Stability Report*, December 2022.", (2022, 2023), set()),
]

# Background on individual stocks with a well-known history.
BOND_NOTES = {
    32400: "The 3½% War Loan was created in 1932, when holders of the 5% War Loan 1929-47 - the largest single "
           "stock left by the First World War - were invited to convert into a 3½% undated stock, and the great "
           "majority did. Its redemption was announced in December 2014 and completed in March 2015.",
    32900: "The 2½% Consols descend from the eighteenth-century consolidated annuities by way of Goschen's 1888 "
           "conversion of the 3% Consols into a new stock whose coupon fell to 2½% in 1903.",
    33000: "The 2½% Treasury Stock 1975 or after - the 'Daltons' - was issued in 1946-47, when Chancellor Hugh "
           "Dalton tried to push long-term interest rates down to 2½%. The stock soon fell well below its issue "
           "price as cheap money broke down in 1947.",
    50400: "The 2% Index-linked Treasury 1996 was the first index-linked gilt, issued in March 1981. At first "
           "index-linked stocks could be held only by pension funds and similar institutions; the restriction "
           "was lifted in 1982.",
    32270: "The 4¼% Treasury Gilt 2055, first issued in May 2005, was the first 50-year conventional gilt, "
           "introduced as pension funds sought long-dated assets to match their liabilities.",
    55500: "The 1¼% Index-linked Treasury Gilt 2055, launched in September 2005, was the first new-style "
           "index-linked gilt (three-month indexation lag) and the first gilt sold by syndication.",
    32202: "The 0⅞% Green Gilt 2033 was the UK's first green gilt, launched by syndication in September 2021 "
           "under the Green Financing Framework.",
    32266: "The 1½% Green Gilt 2053 was the UK's second green gilt, first issued in October 2021.",
    1800: "The 4% Victory Bonds were issued in the Victory Loan of 1919, the first large post-war attempt to fund "
          "the floating debt left by the First World War.",
    15100: "The 4% Funding Loan 1960-90 was issued in 1919, alongside the Victory Bonds, to fund the floating "
           "debt left by the First World War.",
}

INDUSTRY = {"British Transport": "inland transport (railways, canals and road haulage), from 1 January 1948",
            "British Electric": "the electricity supply industry, from 1 April 1948",
            "Exchequer Gas": "the gas industry, from 1 May 1949"}


# ─── Formatting ────────────────────────────────────────────────────────────────

FRAC = {"1/8": "⅛", "1/4": "¼", "3/8": "⅜", "1/2": "½", "5/8": "⅝", "3/4": "¾", "7/8": "⅞"}
FRAC_DEC = {0.125: "⅛", 0.25: "¼", 0.375: "⅜", 0.5: "½", 0.625: "⅝", 0.75: "¾", 0.875: "⅞"}


def _v(x):
    """NaN/NaT -> None."""
    try:
        return None if pd.isna(x) else x
    except (TypeError, ValueError):
        return x


def pretty_name(name):
    """'8 3/4% Treasury 1997' -> '8¾% Treasury 1997'."""
    return re.sub(r"(\d+) ([1357]/[248])%", lambda m: m.group(1) + FRAC[m.group(2)] + "%", str(name))


def coupon_str(c):
    if c is None or pd.isna(c):
        return "variable"
    whole, frac = int(c), round(c - int(c), 3)
    if frac == 0:
        return f"{whole}%"
    if frac in FRAC_DEC:
        return f"{whole}{FRAC_DEC[frac]}%"
    return f"{c:g}%"


def fmt_date(d):
    if d is None or pd.isna(d):
        return "n/a"
    d = pd.Timestamp(d)
    return f"{d.day} {d:%B %Y}"


def fmt_month(d):
    if d is None or pd.isna(d):
        return "n/a"
    return f"{pd.Timestamp(d):%b %Y}"


def fmt_event_date(e):
    s, t = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
    f = fmt_month if e["month"] else fmt_date
    return f(s) if s == t else f"{f(s)} - {f(t)}"


def fmt_gbp(x):
    """£ amount -> '£1,911m' / '£12.4bn'."""
    if x is None or pd.isna(x):
        return "n/a"
    a = abs(x)
    if a >= 1e10:
        return f"£{x / 1e9:,.1f}bn"
    if a >= 1e9:
        return f"£{x / 1e9:,.2f}bn"
    if a >= 1e8:
        return f"£{x / 1e6:,.0f}m"
    return f"£{x / 1e6:,.1f}m"


def fmt_px(p):
    return "n/a" if p is None or pd.isna(p) else f"£{p:,.2f}"


def fmt_pct(x, digits=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}f}%"


def md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(c).replace("|", "/") for c in r) + " |\n"
    return out


def plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def indefinite(word):
    """'a' or 'an' before a coupon such as '8¾%' or '11%' (eight, eleven, eighteen)."""
    lead = re.match(r"\d+", word)
    n = lead.group(0) if lead else ""
    return "an" if (n.startswith("8") or n in ("11", "18") or word[:1].lower() in "aeiou") else "a"


def years_between(a, b):
    return (pd.Timestamp(b) - pd.Timestamp(a)).days / 365.25


def safe_filename(name):
    s = re.sub(r"(\d+) (\d)/(\d)", r"\1_\2-\3", str(name)).replace("%", "pct")
    return re.sub(r"[^A-Za-z0-9\-]+", "_", s).strip("_")


# ─── Knowledge-base queries ────────────────────────────────────────────────────

def get_overlapping_events(start, end):
    s, t = pd.Timestamp(start), pd.Timestamp(end)
    return [e for e in UK_EVENTS if pd.Timestamp(e["start"]) <= t and pd.Timestamp(e["end"]) >= s]


def get_eras(start, end):
    s, t = pd.Timestamp(start), pd.Timestamp(end)
    return [e for e in ERAS if pd.Timestamp(e[0]) <= t and pd.Timestamp(e[1]) >= s]


def governments_between(start, end):
    s, t = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for k, (d, pm, party) in enumerate(GOVERNMENTS):
        a = pd.Timestamp(d)
        b = pd.Timestamp(GOVERNMENTS[k + 1][0] if k + 1 < len(GOVERNMENTS) else GOV_END)
        if a <= t and b >= s:
            out.append((a, b, pm, party))
    return out


def prime_minister_on(date):
    g = governments_between(date, date)
    return g[-1][2] if g else None


def nearest_event(date, max_days=62):
    """The knowledge-base event most likely behind a price move in the month ending `date`: an event starting in
    that month, else one under way during it, else one that ended within `max_days` before it."""
    d = pd.Timestamp(date)
    w0 = d - pd.offsets.MonthEnd(1)
    best, best_score = None, None
    for e in UK_EVENTS:
        s, t = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
        if s > d or t < w0 - pd.Timedelta(days=max_days):
            continue
        if s > w0:
            score = 0
        elif t > w0:
            score = (w0 - s).days / 10
        else:
            score = (w0 - t).days
        if (t - s).days > 180:
            score += 120        # prefer specific events to wars and long crises
        if best_score is None or score < best_score:
            best, best_score = e, score
    return best


def suggest_references(info, flags):
    s = pd.Timestamp(info["start"]).year
    t = pd.Timestamp(info["end_for_ranges"]).year
    refs = []
    for cite, cover, tags in LITERATURE:
        if "core" in tags or (tags & flags) or (cover and cover[0] <= t and cover[1] >= s):
            refs.append(cite)
    return refs


# ─── Database access ───────────────────────────────────────────────────────────

def get_bond_info(db, bond_id):
    bl = db["list"]
    if bond_id not in bl.index:
        return None
    r = bl.loc[bond_id]
    g = lambda c: _v(r.get(c))
    ts = lambda c: None if g(c) is None else pd.Timestamp(g(c))
    info = {
        "id": int(bond_id), "name": r["Treasury's Name Of Issue"], "pretty": pretty_name(r["Treasury's Name Of Issue"]),
        "l1": g("Category L1"), "l2": g("Category L2"), "l3": g("Category L3"), "term": g("Term Of Loan"),
        "issue": ts("First Issue Date"), "early": ts("Redeemable After Date"), "payable": ts("Payable Date"),
        "final_red": ts("Final Redemption Date"), "coupon": g("Coupon Rate"), "freq_text": g("Coupon Frequency"),
        "freq": int(g("Coupons Per Year")) if g("Coupons Per Year") else 2, "callable": g("Callable"),
        "inst": g("Inst Code"), "isin": g("ISIN"), "sedol": g("SEDOL"), "suffix": g("Tranche Suffix"),
        "parent": int(g("Parent ID")) if g("Parent ID") else None, "amalg": ts("Amalgamated Date"),
        "tranches": g("Tranches"), "first_coupon": ts("First Coupon Date"), "pay_dates": g("Payment Dates"),
        "calls": g("Partly Paid Calls"), "special": g("Special Features") or "", "lag": g("Indexation Lag (months)"),
        "base_rpi": g("Base RPI"), "price_basis": g("Price Basis"), "price_from": ts("Price From"),
        "price_to": ts("Price To"), "quant_from": ts("Quantity From"), "quant_to": ts("Quantity To"),
        "source": g("Source") or "", "notes": g("Notes"),
    }
    last_q = db["quant"].index.max()
    if info["parent"] and info["amalg"] is not None:
        info["status"], info["end"] = "amalgamated", info["amalg"]
    elif info["final_red"] is not None:
        info["status"], info["end"] = "redeemed", info["final_red"]
    elif info["quant_to"] is not None and info["quant_to"] >= last_q:
        info["status"], info["end"] = "outstanding", None
    else:
        info["status"] = "ended"
        info["end"] = max([d for d in (info["quant_to"], info["price_to"]) if d is not None], default=None)
    starts = [d for d in (info["issue"], info["quant_from"], info["price_from"]) if d is not None]
    info["start"] = info["issue"] or (min(starts) if starts else pd.Timestamp("1946-01-31"))
    info["end_for_ranges"] = info["end"] or DATA_END
    info["parent_name"] = pretty_name(db["list"].at[info["parent"], "Treasury's Name Of Issue"]) \
        if info["parent"] in db["list"].index else None
    return info


def detect_flags(info):
    f = set()
    l1, l2, sp = info["l1"], info["l2"] or "", info["special"]
    if l1 == "Undated":
        f.add("undated")
    elif info["early"] is not None and info["payable"] is not None and info["early"] < info["payable"]:
        f.add("double-dated")
    if info["calls"]:
        f.add("partly-paid")
    if info["parent"]:
        f.add("tranche")
    if info["tranches"]:
        f.add("has-tranches")
    if "Convertible" in sp or "Convertible" in l2:
        f.add("convertible")
    if l1 == "Index-linked":
        f.add("il-old" if info["id"] < 55000 else "il-new")
    if l1 == "Variable/Floating Rate":
        f.add("variable")
    if l2 in INDUSTRY:
        f.add("nationalisation")
    if l2 in ("BOAC", "BEA"):
        f.add("airline")
    if "Death duties" in sp:
        f.add("death-duties")
    if "Sinking fund" in sp:
        f.add("sinking-fund")
    if "Special tax" in sp:
        f.add("special-tax")
    if l2 == "Green Gilt":
        f.add("green")
    if "Redeemed early" in sp:
        f.add("redeemed-early")
    if sp == "Small":
        f.add("small")
    if "Not in DMO" in sp or "Not always a BGS" in sp:
        f.add("not-dmo")
    if "Bought in" in sp:
        f.add("bought-in")
    if "Serial Funding" in l2:
        f.add("serial-funding")
    if l2 in ("National War Bonds", "National Defence Bonds", "Savings Bonds", "Victory Bonds"):
        f.add("savings")
    if "Specification_RE" in info["source"]:
        f.add("limited-metadata")
    if info["issue"] is not None:
        for e in UK_EVENTS:
            if e["category"] == "war" and e["name"] in ("First World War", "Second World War") and \
                    pd.Timestamp(e["start"]) <= info["issue"] <= pd.Timestamp(e["end"]):
                f.add("wartime")
    if l1 in ("Conventional", "Undated") and info["coupon"] is not None and info["coupon"] <= 5 and \
            info["price_from"] is not None and info["price_from"] < pd.Timestamp("1996-01-01"):
        f.add("low-coupon")
    mat = info["payable"] or info["final_red"]
    if l1 == "Undated" or (info["issue"] is not None and mat is not None and years_between(info["issue"], mat) >= 25):
        f.add("long")
        if info["end_for_ranges"] >= pd.Timestamp("2004-01-01"):
            f.add("long-2004")
    return f


def maturity_class(years):
    """DMO convention: shorts up to 7 years, mediums 7-15, longs over 15."""
    return "short" if years <= 7 else ("medium" if years <= 15 else "long")


def describe_kind(info, flags):
    if "tranche" in flags:
        return f"a tranche of the {info['parent_name'] or 'parent stock'}"
    if "undated" in flags:
        return "an undated (perpetual) gilt"
    if "il-old" in flags:
        return "an old-style index-linked gilt (eight-month indexation lag)"
    if "il-new" in flags:
        return "a new-style index-linked gilt (three-month indexation lag)"
    if "variable" in flags:
        return "a floating-rate gilt" if "Floating" in (info["l2"] or "") else "a variable-rate gilt"
    if "nationalisation" in flags:
        return "a nationalisation compensation stock guaranteed by the Treasury"
    if "airline" in flags:
        return "a Treasury-guaranteed stock of a state airline"
    base = "green gilt" if "green" in flags else ("convertible gilt" if "convertible" in flags else
                                                  ("double-dated conventional gilt" if "double-dated" in flags
                                                   else "conventional gilt"))
    mat = info["payable"] or info["final_red"]
    if info["issue"] is not None and mat is not None:
        yrs = years_between(info["issue"], mat)
        cls = maturity_class(yrs)
        return f"a {cls}-dated {base} ({yrs:.0f} years to {'final ' if 'double-dated' in flags else ''}maturity at issue)"
    art = "an" if base[0] in "aeiou" else "a"
    return f"{art} {base}"


# ─── Analysis ──────────────────────────────────────────────────────────────────

class Cache:
    """Things shared across chapters in one run: yield panels and market-wide monthly price changes."""

    def __init__(self, db):
        self.db, self._panels, self._mkt = db, {}, {}

    def panel(self, group):
        if group not in self._panels:
            self._panels[group] = yield_panel(self.db, group)
        return self._panels[group]

    def market_return(self, group):
        if group not in self._mkt:
            bl = self.db["list"]
            ids = [i for i in bl.index[bl["Category L1"].isin(group)] if (i, "Average") in self.db["price"].columns]
            px = pd.DataFrame({i: clean_price(self.db, i) for i in ids}).reindex(self.db["price"].index)
            self._mkt[group] = px.pct_change(fill_method=None).median(axis=1) * 100
        return self._mkt[group]


def peer_setup(info, flags):
    """(yield panel group, peer_stats keyword arguments) for relative value; (None, {}) when yields are not
    meaningful. Index-linked gilts are few, so their maturity band is wider."""
    if "variable" in flags or info["coupon"] is None:
        return None, {}
    if info["l1"] == "Index-linked":
        return ("Index-linked",), {"band": 0.35, "min_band": 3.0, "min_peers": 2}
    if "undated" in flags:
        return ("Conventional",), {"long_only": 15}
    return ("Conventional",), {}


def price_statistics(p):
    if len(p) == 0:
        return {}
    return {"n": len(p), "first": p.index[0], "last": p.index[-1], "first_px": p.iloc[0], "last_px": p.iloc[-1],
            "mean": p.mean(), "std": p.std(), "min": p.min(), "min_date": p.idxmin(), "max": p.max(),
            "max_date": p.idxmax(), "above_par": (p >= 100).mean() * 100}


def era_statistics(p, y):
    rows = []
    for s, t, name, _ in ERAS:
        seg = p[(p.index >= s) & (p.index <= t)]
        if len(seg) == 0:
            continue
        ys = y[(y.index >= s) & (y.index <= t)] if y is not None else pd.Series(dtype=float)
        rows.append({"era": name, "from": seg.index[0], "to": seg.index[-1], "n": len(seg), "mean": seg.mean(),
                     "min": seg.min(), "max": seg.max(),
                     "yield": ys.mean() if len(ys.dropna()) else np.nan})
    return rows


def quantity_changes(q, info, db):
    """Large month-on-month changes in nominal outstanding: |change| >= max(£50m, 5% of previous)."""
    if len(q) < 2:
        return []
    bl = db["list"]
    tr = bl[bl["Parent ID"] == info["id"]]
    amalg = {pd.Timestamp(r["Amalgamated Date"]): (i, r["Tranche Suffix"]) for i, r in tr.iterrows()
             if pd.notna(r["Amalgamated Date"])}
    out = []
    full = q.reindex(pd.date_range(q.index[0], q.index[-1], freq=pd.offsets.MonthEnd()))
    dq = full.diff()
    for d, ch in dq.dropna().items():
        prev = full.shift(1).loc[d]
        if abs(ch) < max(50e6, 0.05 * prev):
            continue
        kind = "Further issue (tap, auction or syndication)" if ch > 0 else "Reduction (buy-back, conversion or purchase)"
        for ad, (tid, suf) in amalg.items():
            if ch > 0 and abs((ad - d).days) <= 45:
                kind = f"Tranche {suf} (ID {tid}) amalgamated"
        if ch > 0 and info["issue"] is not None and abs((info["issue"] - d).days) <= 45:
            kind = "Initial issue"
        if "convertible" in detect_flags(info) and ch < 0:
            kind = "Reduction (probably conversion)"
        out.append({"date": d, "change": ch, "before": prev, "after": full.loc[d], "pct": ch / prev * 100, "kind": kind})
    return out


def quantity_lifecycle(q, qidx):
    if len(q) == 0:
        return {}
    pos = q.diff()[q.diff() > 0]
    neg = q.diff()[q.diff() < 0]
    out = {"first_date": q.index[0], "first": q.iloc[0], "peak": q.max(), "peak_date": q.idxmax(),
           "last_date": q.index[-1], "last": q.iloc[-1], "n_up": len(pos), "n_down": len(neg),
           "issued_after_first": pos.sum(), "reduced": -neg.sum()}
    if len(qidx):
        out.update({"idx_last": qidx.iloc[-1], "idx_peak": qidx.max(), "idx_peak_date": qidx.idxmax()})
    return out


def yield_statistics(yf):
    y = yf["yield"].dropna() if "yield" in yf else pd.Series(dtype=float)
    if len(y) == 0:
        return {}
    return {"kind": yf["kind"].iloc[0], "n": len(y), "first": y.iloc[0], "first_date": y.index[0], "last": y.iloc[-1],
            "last_date": y.index[-1], "mean": y.mean(), "min": y.min(), "min_date": y.idxmin(), "max": y.max(),
            "max_date": y.idxmax()}


def relative_value(yf, peers):
    if peers is None or len(peers) == 0:
        return {}
    s = ((yf["yield"] - peers["median"]) * 100).dropna()
    if len(s) == 0:
        return {}
    out = {"n": len(s), "mean_bp": s.mean(), "median_bp": s.median(), "rich_share": (s < 0).mean() * 100,
           "first_bp": s.iloc[0], "first_date": s.index[0], "last_bp": s.iloc[-1], "last_date": s.index[-1],
           "max_bp": s.max(), "max_date": s.idxmax(), "min_bp": s.min(), "min_date": s.idxmin(),
           "peer_n": peers["n"].median()}
    out["by_era"] = []
    for a, b, name, _ in ERAS:
        seg = s[(s.index >= a) & (s.index <= b)]
        if len(seg) >= 6:
            out["by_era"].append((name, len(seg), seg.mean()))
    return out


def returns_summary(tr):
    if tr is None or len(tr) < 13:
        return {}
    yrs = years_between(tr.index[0], tr.index[-1])
    out = {"from": tr.index[0], "to": tr.index[-1], "years": yrs, "mult": tr["nominal"].iloc[-1] / 100}
    out["ann"] = (out["mult"] ** (1 / yrs) - 1) * 100
    r = tr["nominal"].pct_change().dropna()
    out["vol"] = r.std() * np.sqrt(12) * 100
    dd = tr["nominal"] / tr["nominal"].cummax() - 1
    out["max_dd"] = dd.min() * 100
    out["dd_date"] = dd.idxmin()
    out["dd_peak"] = tr["nominal"].loc[:dd.idxmin()].idxmax()
    if "real" in tr and tr["real"].notna().sum() >= 13:
        rr = tr["real"].dropna()
        ry = years_between(rr.index[0], rr.index[-1])
        out.update({"real_from": rr.index[0], "real_mult": rr.iloc[-1] / rr.iloc[0],
                    "real_ann": ((rr.iloc[-1] / rr.iloc[0]) ** (1 / ry) - 1) * 100})
    roll = r.rolling(12, min_periods=10).std() * np.sqrt(12) * 100
    if roll.notna().any():
        out["roll_max"], out["roll_max_date"] = roll.max(), roll.idxmax()
    return out


def detect_price_events(p, market, yf=None, peers=None, max_events=12):
    """Months with a large price move (>= 5%) or a large move relative to comparable gilts. With yields and peers,
    'relative' means the change in the yield spread over the peer median (>= 40 bp), which compares like with like
    whatever the maturity; otherwise it is the price change minus the median gilt's (>= 3 points)."""
    if len(p) < 3:
        return []
    idx = pd.date_range(p.index[0], p.index[-1], freq=pd.offsets.MonthEnd())
    r = p.reindex(idx).pct_change(fill_method=None) * 100
    m = market.reindex(idx)
    use_yield = yf is not None and peers is not None and len(peers) > 12
    if use_yield:
        y = yf["yield"].reindex(idx)
        pm = peers["median"].reindex(idx)
        dy, dpm = y.diff() * 100, pm.diff() * 100
        rel = dy - dpm
        score = pd.concat([r.abs() / 5, rel.abs() / 40], axis=1).max(axis=1)
    else:
        rel = r - m
        score = pd.concat([r.abs() / 5, rel.abs() / 3], axis=1).max(axis=1)
    hits = score[score >= 1].sort_values(ascending=False).index[:max_events]
    out = []
    for d in sorted(hits):
        e = nearest_event(d)
        row = {"date": d, "ret": r.loc[d], "market": m.loc[d], "rel": rel.loc[d], "use_yield": use_yield,
               "event": f"{e['name']} ({fmt_event_date(e)})" if e else None}
        if use_yield:
            row.update({"dy": dy.loc[d], "dpeer": dpm.loc[d]})
        out.append(row)
    return out


def find_related_bonds(db, info, max_each=6):
    bl = db["list"].copy()
    me = info["id"]
    root = info["parent"] or me
    fam = [i for i in bl.index[(bl["Parent ID"] == root) | (bl.index == root)] if i != me]
    main = bl[bl["Parent ID"].isna() & (bl.index != me) & ~bl.index.isin(fam)]
    mat = lambda r: r["Payable Date"] if pd.notna(r["Payable Date"]) else r["Final Redemption Date"]
    my_mat = info["payable"] or info["final_red"]
    npx = lambda i: int(db["price"][(i, "Average")].notna().sum()) if (i, "Average") in db["price"].columns else 0

    def item(i, rel):
        r = bl.loc[i]
        return {"id": int(i), "name": pretty_name(r["Treasury's Name Of Issue"]), "rel": rel,
                "issue": _v(r["First Issue Date"]), "maturity": _v(mat(r)), "coupon": _v(r["Coupon Rate"]),
                "n_prices": npx(i)}

    out = {"Tranche family": [item(i, "Parent" if i == root else f"Tranche {bl.at[i, 'Tranche Suffix']}")
                              for i in fam]}
    same = main[main["Category L3"] == info["l3"]]
    if my_mat is not None and len(same):
        key = same.apply(lambda r: abs((pd.Timestamp(mat(r)) - my_mat).days) if pd.notna(mat(r)) else 1e9, axis=1)
        same = same.loc[key.sort_values().index]
    used = set(fam)
    out["Same category and decade"] = [item(i, "same maturity decade") for i in same.index[:max_each]]
    used |= set(same.index[:max_each])
    if info["issue"] is not None:
        c = main[main["First Issue Date"].notna() & ~main.index.isin(used)]
        gap = (c["First Issue Date"] - info["issue"]).abs()
        c = c[gap <= pd.Timedelta(days=366)]
        c = c.loc[gap.loc[c.index].sort_values().index]
        out["Issued within a year"] = [item(i, "contemporary") for i in c.index[:max_each]]
        used |= set(c.index[:max_each])
    if info["coupon"] is not None:
        c = main[(main["Coupon Rate"] == info["coupon"]) & (main["Category L1"] == info["l1"]) & ~main.index.isin(used)]
        if info["issue"] is not None and len(c):
            c = c.loc[(c["First Issue Date"] - info["issue"]).abs().sort_values().index]
        out["Same coupon"] = [item(i, "same coupon") for i in c.index[:max_each]]
    return {k: v for k, v in out.items() if v}


def pick_related_for_chart(related, db, own_yields, k=4):
    own = own_yields["yield"].dropna().index if "yield" in own_yields else pd.Index([])
    if len(own) == 0:
        return []
    cands = []
    for key in ("Same category and decade", "Tranche family", "Issued within a year", "Same coupon"):
        for r in related.get(key, []):
            if len(cands) >= k or r["id"] in [c[0] for c in cands] or r["n_prices"] < 12:
                continue
            yk = bond_yields(db, r["id"])
            if "yield" in yk and len(yk["yield"].dropna().index.intersection(own)) >= 12:
                cands.append((r["id"], r["name"]))
    return cands


def pick_snapshots(yf, panel):
    y = yf["yield"].dropna() if "yield" in yf else pd.Series(dtype=float)
    if len(y) == 0 or not panel:
        return []
    ok = []
    for d in y.index:
        n = sum(1 for v in panel.values() if d in v.index and pd.notna(v.at[d, "yield"]))
        if n >= 8:
            ok.append(d)
    if not ok:
        return []
    ys = y.loc[ok]
    picks = [(ys.index[0], "first priced month"), (ys.idxmax(), "highest yield"), (ys.index[-1], "last priced month")]
    out = []
    for d, why in sorted(picks):
        if all(abs(years_between(d, o[0])) >= 3 for o in out):
            out.append((d, why))
        elif why == "highest yield":
            out = [(o[0], o[1] + " and highest yield") if o[0] == d or abs(years_between(d, o[0])) < 3 else o
                   for o in out]
    return out


def chart_events(info, max_n=12):
    s, t = info["start"], info["end_for_ranges"]
    ev = [e for e in UK_EVENTS if e["key"] and s <= pd.Timestamp(e["start"]) <= t]
    if len(ev) <= max_n:
        chosen = ev
    else:
        span = (t - s).days
        chosen, last = [], None
        for e in ev:
            d = pd.Timestamp(e["start"])
            if last is None or (d - last).days >= span / (max_n + 2):
                chosen.append(e)
                last = d
    return {e["start"]: (e["name"] if len(e["name"]) <= 40 else e["name"][:38] + "…") for e in chosen}


def event_responses(events, price, yf):
    """Price and yield change from the month before each short event to the month it ends."""
    out = {}
    if len(price) < 2:
        return out
    y = yf["yield"] if "yield" in yf else pd.Series(dtype=float)
    for e in events:
        s, t = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
        if (t - s).days > 200:
            continue
        a = (s - pd.offsets.MonthEnd(1)).normalize()
        b = (t + pd.offsets.MonthEnd(0)).normalize()
        if a in price.index and b in price.index:
            dp = (price.loc[b] / price.loc[a] - 1) * 100
            dy = (y.get(b, np.nan) - y.get(a, np.nan)) * 100
            out[e["name"]] = (dp, dy)
    return out


def analyse(db, bond_id, cache):
    info = get_bond_info(db, bond_id)
    if info is None:
        return None
    flags = detect_flags(info)
    price = clean_price(db, bond_id)
    raw = series(db, "price", bond_id, "Average")
    if len(raw) and ((raw - price.reindex(raw.index)).abs() > 1e-9).any():
        flags.add("dirty-converted")
    pp = series(db, "price", bond_id, "Partly Paid")
    quant = series(db, "quant", bond_id, "Total Outstanding")
    qidx = series(db, "quant", bond_id, "Indexed Outstanding")
    yf = bond_yields(db, bond_id)
    group, peer_kw = peer_setup(info, flags)
    has_yield = "yield" in yf and yf["yield"].notna().sum() >= 3 and group is not None
    peers = peer_stats(cache.panel(group), bond_id, yf, **peer_kw) if has_yield else None
    tr = total_return_index(db, bond_id) if len(price) >= 13 and "variable" not in flags else None
    mgroup = ("Index-linked",) if info["l1"] == "Index-linked" else ("Conventional", "Undated")
    events = get_overlapping_events(info["start"], info["end_for_ranges"])
    related = find_related_bonds(db, info)
    ctx = {
        "info": info, "flags": flags, "price": price, "pp": pp, "quant": quant, "qidx": qidx, "yields": yf,
        "group": group if has_yield else None, "peer_kw": peer_kw, "long_only": peer_kw.get("long_only"),
        "peers": peers,
        "has_price": len(price) >= 2, "has_quant": len(quant) >= 1, "has_yield": has_yield,
        "pstats": price_statistics(price), "ystats": yield_statistics(yf) if has_yield else {},
        "era_stats": era_statistics(price, yf["yield"] if has_yield else None) if len(price) else [],
        "qlife": quantity_lifecycle(quant, qidx), "qchanges": quantity_changes(quant, info, db),
        "rv": relative_value(yf, peers) if has_yield else {}, "tr": tr, "ret": returns_summary(tr),
        "pevents": detect_price_events(price, cache.market_return(mgroup), yf if has_yield else None,
                                       peers) if len(price) >= 3 else [],
        "events": events, "eras": get_eras(info["start"], info["end_for_ranges"]),
        "govs": governments_between(info["start"], info["end_for_ranges"]),
        "responses": event_responses(events, price, yf), "related": related,
        "related_chart": pick_related_for_chart(related, db, yf) if has_yield else [],
        "snapshots": pick_snapshots(yf, cache.panel(group)) if has_yield else [],
        "chart_events": chart_events(info), "refs": suggest_references(info, flags),
        "corrections": corrections_for(db, bond_id, info),
    }
    return ctx


def corrections_for(db, bond_id, info):
    c = db["corrections"]
    ids = c["L1 ID"].astype(str).str.strip()
    mask = ids == str(bond_id)
    has_price_oct22 = info["price_from"] is not None and info["price_from"] <= pd.Timestamp("2022-10-31") and \
        (info["price_to"] is None or info["price_to"] >= pd.Timestamp("2022-10-31"))
    if has_price_oct22:
        mask |= ids == "all"
    if info["l1"] == "Index-linked":
        mask |= ids == "all index-linked"
    return c[mask]


# ─── Notebook Generation (Cell Builders) ───────────────────────────────────────

ENRICH = "\n\n<!-- ENRICH: {} -->"


def build_title_cells(ctx):
    i = ctx["info"]
    ids = [f"**L1 ID:** {i['id']}"]
    if i["inst"]:
        ids.append(f"**Instrument code:** {i['inst']}")
    if i["isin"] and str(i["isin"]).startswith("GB"):
        ids.append(f"**ISIN:** {i['isin']}")
    toc = ["At a Glance", "Overview", "The Instrument", "Historical Context", "Lifecycle Timeline",
           "Issuance and Amount Outstanding"]
    if ctx["has_price"]:
        toc += ["Market Price"]
    if ctx["has_yield"]:
        toc += ["Yield and Relative Value", "On the Yield Curve"]
    if ctx["ret"]:
        toc += ["Returns and Risk to Holders"]
    if ctx["has_price"]:
        toc += ["Market Events"]
    toc += ["Related Stocks", "Issuance, Distribution and Redemption", "Implications and Legacy",
            "Data Notes", "References"]
    return [new_markdown_cell(
        f"# Gilt Biography: {i['pretty']}\n\n"
        "*Generated from the Ellison-Scott UK government bond (gilt) database: 502 stocks, amounts outstanding "
        "1946-2023, month-end prices 1975-2023.*\n\n"
        + "  \n".join(ids) + "  \n"
        f"**Category:** {i['l1']} > {i['l2']} > {i['l3']}\n\n"
        "**Contents:** " + " · ".join(toc) + "\n\n---"
    )]


def build_setup_cells(ctx):
    i, f = ctx["info"], ctx["flags"]
    lo = ctx["long_only"]
    group = ctx["group"]
    bars = []
    life_end = i["end_for_ranges"]
    q = ctx["quant"]
    if i["issue"] is not None or len(q):
        s = i["issue"] if i["issue"] is not None else q.index[0]
        bars.append(("In issue" if i["issue"] is not None else "Recorded outstanding", s, life_end, "#1f4e79"))
    if len(ctx["pp"]):
        bars.append(("Partly paid", ctx["pp"].index[0], ctx["pp"].index[-1], "#c0504d"))
    if ctx["has_price"]:
        bars.append(("Month-end prices", ctx["price"].index[0], ctx["price"].index[-1], "#2e7d32"))
    if "double-dated" in f:
        bars.append(("Call window", i["early"], i["payable"], "#e0a030"))
    if "undated" in f and i["early"] is not None:
        bars.append(("Callable at par", i["early"], life_end, "#e0a030"))
    bars = [(a, pd.Timestamp(b).strftime("%Y-%m-%d"), pd.Timestamp(c).strftime("%Y-%m-%d"), d) for a, b, c, d in bars]
    govs = [(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"), pm, party) for a, b, pm, party in ctx["govs"]]
    kind = ctx["ystats"].get("kind")
    ylabel = {"gross redemption": "Gross redemption yield", "flat": "Flat (running) yield",
              "real": "Real yield"}.get(kind, "Yield")
    if "il-old" in f:
        plabel = "Clean price incl. RPI uplift (old style)"
    elif "il-new" in f:
        plabel = "Real clean price (new style)"
    else:
        plabel = "Clean price, fully paid"
    if "dirty-converted" in f:
        plabel += " (pre-1986 dirty prices converted)"
    lines = [
        "# ── Load the database and this gilt's series ─────────────────────────────",
        "import matplotlib.pyplot as plt",
        "import matplotlib.dates as mdates",
        "import warnings",
        "warnings.filterwarnings('ignore')",
        "plt.style.use('seaborn-v0_8-whitegrid')",
        "plt.rcParams.update({'figure.figsize': (12, 5.5), 'font.size': 11, 'axes.titlesize': 13})",
        "C_BOND, C_PEER, C_ACC, C_REAL = '#1f4e79', '#8c8c8c', '#c0504d', '#2e7d32'",
        "",
        "db = load_database()   # UK_Gilts_Bond_Database.xlsx in this folder or a parent; $UK_GILTS_DB overrides",
        f"BOND_ID = {i['id']}",
        "info = db['list'].loc[BOND_ID]",
        f"NAME = {i['pretty']!r}",
        "price    = clean_price(db, BOND_ID)                             # clean £ per £100 nominal, fully paid",
        "price_pp = series(db, 'price', BOND_ID, 'Partly Paid')          # while instalments were still due",
        "quant    = series(db, 'quant', BOND_ID, 'Total Outstanding')    # £ nominal",
        "quant_ix = series(db, 'quant', BOND_ID, 'Indexed Outstanding')  # £ incl. RPI uplift (index-linked only)",
        "yields   = bond_yields(db, BOND_ID)",
        f"PRICE_LABEL = {plabel!r}",
        f"YIELD_LABEL = {ylabel!r}",
    ]
    if group:
        peer_desc = ("long-dated conventional gilts (15+ years to redemption)" if lo else
                     ("index-linked gilts of similar remaining life" if group == ("Index-linked",) else
                      "conventional gilts of similar remaining life"))
        lines += [
            f"panel = yield_panel(db, {group!r})   # yields of every gilt in the peer group",
            f"peers = peer_stats(panel, BOND_ID, yields{''.join(f', {k}={v!r}' for k, v in ctx['peer_kw'].items())})",
            f"PEER_LABEL = {peer_desc!r}",
        ]
    else:
        lines += ["panel, peers = {}, pd.DataFrame()"]
    lines += [
        "",
        f"EVENTS = {ctx['chart_events']!r}",
        f"LIFE_BARS = {bars!r}",
        f"GOVERNMENTS = {govs!r}",
        "PARTY_COLOURS = {'Conservative': '#0087dc', 'Labour': '#e4003b', 'Liberal': '#faa61a',",
        "                 'Coalition': '#9e9e9e', 'National': '#7f7f7f', 'Con-LD coalition': '#8e7cc3'}",
        f"SNAPSHOTS = {[d.strftime('%Y-%m-%d') for d, _ in ctx['snapshots']]!r}",
        f"RELATED = {dict(ctx['related_chart'])!r}",
        "",
        "def mark_events(ax, events=None, fontsize=7.5):",
        "    # dashed vertical lines and labels for the key UK events inside the current x-range",
        "    events = EVENTS if events is None else events",
        "    lo, hi = ax.get_xlim()",
        "    top = ax.get_ylim()[1]",
        "    for d, label in events.items():",
        "        x = mdates.date2num(pd.Timestamp(d))",
        "        if lo <= x <= hi:",
        "            ax.axvline(x, color='grey', ls='--', lw=0.8, alpha=0.5)",
        "            ax.text(x, top, ' ' + label, rotation=90, va='top', ha='right', fontsize=fontsize, color='dimgrey')",
        "",
        "print(f'{NAME} (ID {BOND_ID}): {len(price)} month-end prices, {len(quant)} months of amounts outstanding')",
    ]
    return [
        new_markdown_cell(
            "### Setup\n\nThe first code cell holds the helper functions shared by every chapter (loading the "
            "database, yields, index ratios, peer statistics, total returns). The second loads this gilt's series. "
            "The database file `UK_Gilts_Bond_Database.xlsx` must be in the notebook's folder or a parent folder, "
            "or its path set in the `UK_GILTS_DB` environment variable."),
        new_code_cell(HELPERS_SRC.strip()),
        new_code_cell("\n".join(lines)),
    ]


def build_glance_cells(ctx):
    i, f, ps, ys, ql, ret = ctx["info"], ctx["flags"], ctx["pstats"], ctx["ystats"], ctx["qlife"], ctx["ret"]
    rows = [("Stock", i["pretty"]), ("Type", f"{i['l1']} / {i['l2']}")]
    cp = coupon_str(i["coupon"])
    freq = i["freq_text"] or "semi-annually"
    rows.append(("Coupon", f"{cp} paid {freq}" + (f" ({i['pay_dates']})" if i["pay_dates"] else "")))
    rows.append(("First issued", fmt_date(i["issue"]) if i["issue"] is not None else
                 f"before the data begin (first recorded {fmt_month(i['quant_from'])})"))
    if "undated" in f:
        terms = "Undated" + (f"; redeemable at par on or after {fmt_date(i['early'])}" if i["early"] is not None else "")
    elif "double-dated" in f:
        terms = f"Double-dated: redeemable between {fmt_date(i['early'])} and {fmt_date(i['payable'])}"
    elif i["payable"] is not None:
        terms = f"Repayable {fmt_date(i['payable'])}"
    else:
        terms = "n/a"
    rows.append(("Redemption terms", terms))
    if i["status"] == "redeemed":
        rows.append(("Redeemed", fmt_date(i["final_red"])))
    elif i["status"] == "amalgamated":
        rows.append(("Amalgamated", f"with {i['parent_name']} (ID {i['parent']}) on {fmt_date(i['amalg'])}"))
    elif i["status"] == "outstanding":
        rows.append(("Status", f"Outstanding at the end of the data ({fmt_month(ctx['quant'].index[-1])})"))
    else:
        rows.append(("Last recorded", fmt_month(i["end"])))
    if i["l1"] == "Index-linked":
        rows.append(("Indexation", f"RPI, {int(i['lag'])}-month lag; base RPI {i['base_rpi']:.4g}"
                     if i["lag"] and i["base_rpi"] else "RPI"))
    if i["calls"]:
        rows.append(("Partly paid", i["calls"]))
    if ql:
        rows.append(("Peak nominal outstanding", f"{fmt_gbp(ql['peak'])} ({fmt_month(ql['peak_date'])})"))
    if ps:
        rows.append(("Month-end prices", f"{ps['n']} months, {fmt_month(ps['first'])} to {fmt_month(ps['last'])}"))
        rows.append(("Price range", f"{fmt_px(ps['min'])} ({fmt_month(ps['min_date'])}) to "
                                    f"{fmt_px(ps['max'])} ({fmt_month(ps['max_date'])})"))
    if ys:
        rows.append((f"{ys['kind'].capitalize()} yield range",
                     f"{fmt_pct(ys['min'])} ({fmt_month(ys['min_date'])}) to {fmt_pct(ys['max'])} "
                     f"({fmt_month(ys['max_date'])})"))
    if ret:
        s = f"{ret['ann']:.1f}% a year nominal over {ret['years']:.1f} years"
        if "real_ann" in ret:
            s += f"; {ret['real_ann']:.1f}% a year real (from {fmt_month(ret['real_from'])})"
        rows.append(("Total return to holders", s))
    if i["special"]:
        rows.append(("Special features (source)", i["special"]))
    return [new_markdown_cell("## At a Glance\n\n" + md_table(["", ""], rows))]


def build_overview_cells(ctx):
    i, f, ps, ys, ql, ret, rv = (ctx[k] for k in ("info", "flags", "pstats", "ystats", "qlife", "ret", "rv"))
    kind = describe_kind(i, f)
    cp = coupon_str(i["coupon"])
    p1 = f"The **{i['pretty']}** is {kind}"
    if i["coupon"] is not None:
        p1 += f", paying {indefinite(cp)} {cp} coupon {i['freq_text'] or 'semi-annually'}"
    p1 += "."
    if i["issue"] is not None:
        pm = prime_minister_on(i["issue"])
        p1 += f" It was first issued on {fmt_date(i['issue'])}" + (f", when {pm} was Prime Minister" if pm else "") + "."
    else:
        p1 += (f" It was issued before the amounts series begin; it is first recorded in "
               f"{fmt_month(i['quant_from'])}.")
    if "undated" in f:
        p1 += (" It had no final maturity: the Treasury could repay it at par"
               + (f" at any time after {fmt_date(i['early'])}" if i["early"] is not None else "") +
               ", but was never obliged to.")
    elif "double-dated" in f:
        p1 += (f" The Treasury could repay it at par at any time between {fmt_date(i['early'])} and "
               f"{fmt_date(i['payable'])}.")
    elif i["payable"] is not None and "tranche" not in f and not (
            i["status"] == "redeemed" and abs((i["final_red"] - i["payable"]).days) <= 5):
        p1 += f" It was repayable on {fmt_date(i['payable'])}."
    if i["status"] == "redeemed" and i["payable"] is not None and abs((i["final_red"] - i["payable"]).days) <= 5 \
            and "double-dated" not in f:
        yrs = years_between(i["start"], i["final_red"])
        p1 += f" It was repaid at maturity on {fmt_date(i['final_red'])}, {yrs:.0f} years after it " \
              f"{'was first issued' if i['issue'] is not None else 'first appears in the data'}."
    elif i["status"] == "redeemed":
        yrs = years_between(i["start"], i["final_red"])
        p1 += f" It was redeemed on {fmt_date(i['final_red'])}, {yrs:.0f} years after it " \
              f"{'was first issued' if i['issue'] is not None else 'first appears in the data'}."
    elif i["status"] == "amalgamated":
        p1 += f" It was amalgamated with the parent stock on {fmt_date(i['amalg'])}."
    elif i["status"] == "outstanding":
        p1 += f" It was still outstanding at the end of the data ({fmt_month(ctx['quant'].index[-1])})."
    eras = ctx["eras"]
    if len(eras) == 1:
        p1 += f" Its life fell within one era of British public finance: *{eras[0][2]}*."
    elif eras:
        names = [e[2] for e in eras]
        p1 += (f" Its life spanned {len(eras)} eras of British public finance, from "
               f"*{names[0]}* to *{names[-1]}*.")
    if i["id"] in BOND_NOTES:
        p1 += "\n\n" + BOND_NOTES[i["id"]]

    p2 = []
    if ql:
        s = (f"The database records its nominal amount outstanding from {fmt_month(ql['first_date'])} "
             f"({fmt_gbp(ql['first'])}) to {fmt_month(ql['last_date'])} ({fmt_gbp(ql['last'])}), with a peak of "
             f"{fmt_gbp(ql['peak'])} in {fmt_month(ql['peak_date'])}.")
        big_up = [c for c in ctx["qchanges"] if c["change"] > 0 and c["kind"] != "Initial issue"]
        if big_up:
            s += f" It was enlarged {plural(len(big_up), 'time')} by large further issues or tranches."
        p2.append(s)
    if ps:
        p2.append(f"Month-end prices run from {fmt_month(ps['first'])} to {fmt_month(ps['last'])} ({ps['n']} "
                  f"observations): the price averaged {fmt_px(ps['mean'])} per £100 nominal, ranging from "
                  f"{fmt_px(ps['min'])} ({fmt_month(ps['min_date'])}) to {fmt_px(ps['max'])} "
                  f"({fmt_month(ps['max_date'])}).")
    elif i["start"] < pd.Timestamp("1975-11-30"):
        p2.append("The database has no prices for this stock: the price series begin in November 1975.")
    if ys:
        p2.append(f"Its {ys['kind']} yield ranged from {fmt_pct(ys['min'])} to {fmt_pct(ys['max'])}.")
    if rv:
        rich = "below" if rv["mean_bp"] < 0 else "above"
        p2.append(f"On average it yielded {abs(rv['mean_bp']):.0f} basis points {rich} the median of its peers.")
    if ret:
        s = (f"A holder who bought at the first recorded price and reinvested the coupons earned "
             f"{ret['ann']:.1f}% a year in nominal terms")
        if "real_ann" in ret:
            s += f" and {ret['real_ann']:.1f}% a year after inflation (from {fmt_month(ret['real_from'])})"
        p2.append(s + ".")
    text = "## Overview\n\n" + p1 + ("\n\n" + " ".join(p2) if p2 else "")
    text += ENRICH.format("Add 1-2 paragraphs on why this stock was issued (the fiscal and monetary situation, "
                          "the Chancellor and the funding policy of the day), who bought it, and what makes its "
                          "story distinctive in the history of the British national debt.")
    return [new_markdown_cell(text)]


def feature_paragraphs(ctx):
    i, f = ctx["info"], ctx["flags"]
    out = []
    if "double-dated" in f:
        s = (f"**Double-dated (callable).** The Treasury could redeem the stock at par at any time between "
             f"{fmt_date(i['early'])} and {fmt_date(i['payable'])}, after giving notice. This is a call option held "
             "by the government: when market yields fell below the coupon it could refinance more cheaply, and when "
             "yields were above the coupon it would wait for the final date. Market convention, followed in this "
             "chapter, measures the yield to the earliest date when the stock trades at or above par and to the "
             "final date otherwise.")
        fr = i["final_red"]
        if fr is not None:
            if fr <= i["early"] + pd.Timedelta(days=45):
                s += f" It was repaid at the first opportunity ({fmt_date(fr)}), consistent with market yields below its coupon."
            elif fr >= i["payable"] - pd.Timedelta(days=45):
                s += f" It ran to its final date ({fmt_date(fr)}), consistent with market yields above its coupon."
            else:
                s += f" It was repaid on {fmt_date(fr)}, inside the call window."
        out.append(s)
    if "undated" in f:
        s = ("**Undated.** The stock had no final maturity. The Treasury could repay it at par at its option "
             + (f"(on or after {fmt_date(i['early'])}) " if i["early"] is not None else "") +
             "but was never obliged to; holders could get their money back only by selling in the market. Its price "
             "therefore behaved like that of a perpetuity - roughly the coupon divided by the long-term interest "
             "rate - and the natural yield measure is the flat (running) yield, coupon ÷ price. Because their "
             "coupons were low, the undated stocks became worth repaying only when long yields fell below those "
             "coupons: between February and July 2015 the government redeemed all eight remaining undated gilts.")
        if i["special"]:
            s += f" For this stock the source records: *{i['special']}*."
        out.append(s)
    if "partly-paid" in f:
        out.append(f"**Partly paid.** Buyers paid for the stock in instalments: {i['calls']}. Until the last "
                   "instalment the stock traded partly paid; those prices are kept in the database's separate "
                   "'Partly Paid' series and are not comparable with fully-paid prices. Partly-paid issues were common "
                   "in the 1970s and 1980s: they let the authorities sell large amounts of stock while demand was "
                   "strong and spread the cash payments over several weeks, and they gave buyers a geared exposure to "
                   "gilt prices.")
    if "tranche" in f:
        out.append(f"**Tranche.** This record is tranche {i['suffix']} of the {i['parent_name']} (ID {i['parent']}). "
                   "A tranche was a further issue of an existing stock, often created for the Bank of England, which "
                   "sold it into the market as demand allowed. It usually differed from the parent only in its first "
                   "interest payment and was amalgamated with it once the terms became identical"
                   + (f" - here on {fmt_date(i['amalg'])}" if i["amalg"] is not None else "") +
                   ". After amalgamation the amount is part of the parent's series.")
    if "has-tranches" in f:
        out.append(f"**Tranches.** The stock was enlarged by later tranches: {i['tranches']}. Each was a further "
                   "issue of the same stock, amalgamated with it once its terms became identical.")
    if "convertible" in f:
        out.append("**Convertible.** Holders had the option to convert the stock, on specified dates and terms, into "
                   "another (usually longer-dated) gilt. Convertibles let the Treasury sell shorter stock at a lower "
                   "yield in exchange for an option that, if exercised, lengthened the maturity of the debt. The "
                   "conversion terms are not recorded in the database; falls in the amount outstanding around the "
                   "conversion dates probably reflect conversions.")
    if "il-old" in f:
        s = ("**Index-linked (old style).** Coupons and principal are uplifted by the Retail Prices Index with an "
             f"eight-month lag: each payment is scaled by RPI eight months earlier divided by the base RPI "
             f"({i['base_rpi']:.4g} on the January 1987 = 100 scale). The lag lets the next coupon be known in "
             "advance, but leaves holders unprotected against inflation in the last eight months. Prices are quoted "
             "in nominal terms including the uplift. Because part of the uplift in each future payment depends on "
             "RPI values not yet published, a real yield needs an inflation assumption: this chapter projects the "
             "RPI at 3% a year, a conventional choice.")
        if i["issue"] is not None and i["issue"] < pd.Timestamp("1982-03-31"):
            s += (" When this stock was first issued, index-linked gilts could be held only by pension funds and "
                  "similar institutions; the restriction was lifted in 1982.")
        out.append(s)
    if "il-new" in f:
        out.append("**Index-linked (new style).** Introduced in 2005, the new design follows the Canadian model: "
                   "a three-month lag, with the reference RPI for each day interpolated between the RPI three and "
                   "two months earlier. Prices are quoted in real terms, so the price chart shows the real clean "
                   "price; the cash price is price × index ratio"
                   + (f" (base reference RPI {i['base_rpi']:.5g})." if i["base_rpi"] else "."))
    if "variable" in f:
        if "Floating" in (i["l2"] or ""):
            out.append("**Floating rate.** The coupon was reset every quarter in line with three-month sterling "
                       "interbank rates (LIBID) and paid quarterly. Because the coupon tracked money-market rates the "
                       "price stayed close to par, and there is no meaningful redemption yield.")
        else:
            out.append("**Variable rate.** The coupon was not fixed but reset every six months in line with the "
                       "Treasury bill rate. Variable-rate stocks were issued in 1977-79, when investors were wary of "
                       "fixed coupons after the inflation of the mid-1970s; because the coupon tracked short-term "
                       "rates there is no meaningful redemption yield.")
    if "nationalisation" in f:
        out.append(f"**Nationalisation compensation stock.** Issued to the former owners when the Attlee government "
                   f"nationalised {INDUSTRY[i['l2']]}. It was the obligation of the new public corporation, "
                   "guaranteed by the Treasury, and traded in the gilt-edged market like a government stock. Paying "
                   "compensation in low-coupon, long-dated stock at the cheap-money yields of the late 1940s was "
                   "itself a policy choice - holders lost heavily in real terms as yields rose in the following "
                   "decades.")
    if "airline" in f:
        out.append("**State airline stock.** Stock of the state airline (BOAC or BEA), guaranteed by the Treasury "
                   "and counted with government debt. Such issues were small.")
    if "death-duties" in f:
        out.append("**Accepted for death duties.** The stock could be surrendered at a fixed value in payment of "
                   "estate duty. When it traded below that value it was worth more to holders facing estate duty than "
                   "to other investors, which distorts its price and yield; the source excludes such stocks from "
                   "yield indices.")
    if "sinking-fund" in f:
        out.append("**Sinking fund.** The stock carried a sinking fund: money was set aside to buy stock in the "
                   "market or redeem it, gradually reducing the amount outstanding.")
    if "special-tax" in f:
        out.append("**Special tax status.** The source flags this stock as having special tax treatment (for "
                   "example, exemption from UK tax for holders not ordinarily resident, 'FOTRA' status). Compare its "
                   "yield with ordinary gilts with care.")
    if "green" in f:
        out.append("**Green gilt.** Proceeds are matched to eligible green expenditure under the UK Government "
                   "Green Financing Framework (June 2021). Otherwise it is an ordinary conventional gilt, ranking "
                   "equally with the rest of the debt.")
    if "redeemed-early" in f:
        out.append("**Redeemed early.** The source records that the stock was repaid before its final date - the "
                   "Treasury exercised its call option.")
    if "serial-funding" in f:
        out.append("**Serial funding stock.** Short stocks maturing in successive years, used in the early 1950s "
                   "to fund part of the Treasury bills held by the banks as cheap money ended (see Allen 2014).")
    if "savings" in f:
        out.append(f"**Savings-campaign stock.** {i['l2']} were sold continuously to the public and institutions "
                   "through the savings movement, at fixed terms, to absorb savings during and after the wars.")
    if "wartime" in f:
        war = "First World War" if i["issue"] < pd.Timestamp("1919-01-01") else "Second World War"
        out.append(f"**Wartime issue.** First issued during the {war}"
                   + ("; war loans were sold through patriotic public campaigns." if war == "First World War" else
                      ", when the Treasury held borrowing costs at about 3% behind exchange and capital-issues "
                      "controls."))
    if "low-coupon" in f:
        out.append("**Low coupon (tax effect).** Gilt coupons are taxed as income, but capital gains on gilts have "
                   "been exempt from capital gains tax (for stock held more than a year from 1969, and entirely from "
                   "1986). A low-coupon stock priced below par returns part of its yield as a tax-free gain, so "
                   "high-rate taxpayers bid such stocks up and their gross redemption yields were typically below "
                   "those of high-coupon gilts of similar maturity. Bear this in mind when reading the peer spread "
                   "below.")
    if "bought-in" in f:
        out.append("**Bought in.** The source notes that stock was bought in by the government, so the amount in "
                   "market hands was smaller than the recorded total.")
    if "not-dmo" in f:
        out.append(f"**Status.** The source notes: *{i['special']}*. The stock may be missing from some official "
                   "lists of British government securities.")
    if "small" in f:
        out.append("**Small issue.** Flagged as small in the source.")
    if "limited-metadata" in f:
        out.append("**Limited metadata.** This early stock comes from the older Ellison-Scott files, which record "
                   "little beyond the name, coupon, redemption date and amounts outstanding.")
    return out


def build_instrument_cells(ctx):
    i = ctx["info"]
    labels = [("Treasury name", i["name"]), ("Category (L1 / L2 / L3)", f"{i['l1']} / {i['l2']} / {i['l3']}"),
              ("Term of loan", i["term"]), ("Coupon rate", coupon_str(i["coupon"]) if i["coupon"] is not None else None),
              ("Coupon frequency", i["freq_text"]), ("Payment dates", i["pay_dates"]),
              ("First coupon", fmt_date(i["first_coupon"]) if i["first_coupon"] is not None else None),
              ("First issue date", fmt_date(i["issue"]) if i["issue"] is not None else None),
              ("Redeemable after", fmt_date(i["early"]) if i["early"] is not None else None),
              ("Payable (final) date", fmt_date(i["payable"]) if i["payable"] is not None else None),
              ("Final redemption", fmt_date(i["final_red"]) if i["final_red"] is not None else None),
              ("Callable", {1: "Yes", 0: "No"}.get(i["callable"])), ("Partly-paid calls", i["calls"]),
              ("Tranches", i["tranches"]),
              ("Parent stock", f"{i['parent_name']} (ID {i['parent']})" if i["parent"] else None),
              ("Amalgamated", fmt_date(i["amalg"]) if i["amalg"] is not None else None),
              ("Indexation lag", f"{int(i['lag'])} months" if i["lag"] else None),
              ("Base RPI (Jan 1987 = 100)", f"{i['base_rpi']:.5g}" if i["base_rpi"] else None),
              ("Price basis", i["price_basis"]), ("Instrument code", i["inst"]),
              ("ISIN", i["isin"] if i["isin"] and str(i["isin"]).startswith("GB") else None), ("SEDOL", i["sedol"]),
              ("Special features", i["special"] or None)]
    rows = [(a, b) for a, b in labels if b is not None and b != ""]
    paras = feature_paragraphs(ctx)
    text = "## The Instrument\n\n" + md_table(["Feature", "Value"], rows)
    if paras:
        text += "\n### What the features mean\n\n" + "\n\n".join(paras)
    text += ENRICH.format("Describe the prospectus terms (how the stock was offered, the issue price, any "
                          "conversion or tax terms) and how these features shaped who held the stock.")
    return [new_markdown_cell(text)]


def build_context_cells(ctx):
    i, events, eras = ctx["info"], ctx["events"], ctx["eras"]
    text = "## Historical Context\n\n"
    if eras:
        text += "### Eras\n\n"
        for s, t, name, desc in eras:
            text += f"**{name} ({s[:4]}-{t[:4]}).** {desc}\n\n"
    if ctx["govs"]:
        g = [f"{pm} ({party}, {max(a, i['start']).year}-{min(b, i['end_for_ranges']).year})"
             for a, b, pm, party in ctx["govs"]]
        text += "**Prime ministers during the stock's life:** " + "; ".join(g) + ".\n\n"
    if events:
        resp = ctx["responses"]
        show_resp = bool(resp)
        head = ["Date", "Event", "Category"] + (["Price change", "Yield change"] if show_resp else [])
        rows = []
        for e in events:
            r = [fmt_event_date(e), e["name"], e["category"].title()]
            if show_resp:
                if e["name"] in resp:
                    dp, dy = resp[e["name"]]
                    r += [f"{dp:+.1f}%", "n/a" if pd.isna(dy) else f"{dy:+.0f} bp"]
                else:
                    r += ["", ""]
            rows.append(r)
        text += "### Events during the stock's life\n\n" + md_table(head, rows)
        if show_resp:
            text += ("\n*Price and yield changes run from the month-end before a short event (six months or less) "
                     "to the month-end in which it ended.*\n")
    qs = [f"- How did the {e['name']} affect this stock's price, yield and the demand for it?"
          for e in events if e["category"] in ("crisis", "war", "monetary")][:6]
    text += ENRICH.format("Expand on the historical context. Specifically:\n" + ("\n".join(qs) if qs else
                          "- What was the fiscal and political backdrop to this stock?"))
    return [new_markdown_cell(text)]


TIMELINE_CODE = """\
# Lifecycle timeline: life of the stock, price coverage, call window, prime ministers and key events
fig, ax = plt.subplots(figsize=(12, 2.6 + 0.6 * (len(LIFE_BARS) + 1)))
lo = min(mdates.date2num(pd.Timestamp(b[1])) for b in LIFE_BARS)
hi = max(mdates.date2num(pd.Timestamp(b[2])) for b in LIFE_BARS)
for k, (label, start, end, colour) in enumerate(LIFE_BARS):
    s, e = mdates.date2num(pd.Timestamp(start)), mdates.date2num(pd.Timestamp(end))
    ax.barh(k, max(e - s, 20), left=s, height=0.5, color=colour, alpha=0.85)
    ax.text(s, k - 0.32, f'{pd.Timestamp(start):%b %Y}', fontsize=8, ha='left', va='bottom')
    if e - s > (hi - lo) * 0.15:
        ax.text(e, k - 0.32, f'{pd.Timestamp(end):%b %Y}', fontsize=8, ha='right', va='bottom')
gy = len(LIFE_BARS)
for start, end, pm, party in GOVERNMENTS:
    s = max(mdates.date2num(pd.Timestamp(start)), lo)
    e = min(mdates.date2num(pd.Timestamp(end)), hi)
    if e <= s:
        continue
    ax.barh(gy, e - s, left=s, height=0.5, color=PARTY_COLOURS.get(party, '#cccccc'), alpha=0.55, edgecolor='white')
    if e - s > (hi - lo) * 0.05:
        ax.text((s + e) / 2, gy, pm, ha='center', va='center', fontsize=7.5)
ax.set_yticks(range(gy + 1))
ax.set_yticklabels([b[0] for b in LIFE_BARS] + ['Prime Minister'])
pad = (hi - lo) * 0.03
ax.set_xlim(lo - pad, hi + pad)
ax.set_ylim(gy + 0.6, -4.4)          # leave room at the top for event labels
ax.xaxis_date()
for d, label in EVENTS.items():
    x = mdates.date2num(pd.Timestamp(d))
    if lo <= x <= hi:
        ax.axvline(x, color='grey', ls=':', lw=0.8)
        short = label if len(label) <= 30 else label[:29] + '…'
        ax.text(x, -0.55, short, rotation=90, ha='center', va='bottom', fontsize=7, color='dimgrey')
ax.set_title(f'{NAME}: lifecycle')
ax.grid(axis='y', visible=False)
plt.tight_layout()
plt.show()"""


def build_timeline_cells(ctx):
    return [new_markdown_cell("## Lifecycle Timeline\n\nThe bars show when the stock was in issue, when month-end "
                              "prices are available, any partly-paid period and call window, and the prime "
                              "ministers of the day; dotted lines mark key events."),
            new_code_cell(TIMELINE_CODE)]


QUANT_CODE = """\
# Nominal amount outstanding; dots mark large changes (at least £50m and 5%)
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.step(quant.index, quant / 1e6, where='post', color=C_BOND, lw=1.8, label='Nominal outstanding')
if len(quant_ix):
    ax.step(quant_ix.index, quant_ix / 1e6, where='post', color=C_REAL, lw=1.4, label='Indexed (RPI-uplifted) outstanding')
full = quant.reindex(pd.date_range(quant.index[0], quant.index[-1], freq=pd.offsets.MonthEnd()))
dq = full.diff()
big = dq[dq.abs() >= np.maximum(50e6, 0.05 * full.shift(1))].dropna()
if len(big):
    ax.scatter(big.index, full.loc[big.index] / 1e6, c=np.where(big > 0, C_BOND, C_ACC), s=30, zorder=3,
               label='Large change (blue up, red down)')
ax.set_ylabel('£ million')
top = max(quant.max(), quant_ix.max() if len(quant_ix) else 0) / 1e6
ax.set_ylim(0, top * 1.25)
share = quant / gilt_stock_total(db).reindex(quant.index) * 100
ax2 = ax.twinx()
ax2.plot(share.index, share, color=C_PEER, lw=1, ls='--', label='Share of all gilts in the database (%)')
ax2.set_ylabel('% of nominal gilt stock')
ax2.set_ylim(0, share.max() * 1.25)
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9)
ax.set_title(f'{NAME}: amount outstanding')
mark_events(ax)
plt.tight_layout()
plt.show()"""

MV_CODE = """\
# Market value of the amount outstanding = price x nominal / 100 (new-style index-linked prices are real, so uplift them)
px = price.copy()
if info['Category L1'] == 'Index-linked' and not is_old_style_il(info, BOND_ID):
    px = px * index_ratio(db, BOND_ID, px.index).values
mv = (px * quant.reindex(px.index) / 100 / 1e6).dropna()
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.fill_between(mv.index, mv, color=C_BOND, alpha=0.2)
ax.plot(mv.index, mv, color=C_BOND, lw=1.5)
ax.set_ylabel('£ million')
ax.set_ylim(bottom=0)
ax.set_title(f'{NAME}: market value of the amount outstanding')
mark_events(ax)
plt.tight_layout()
plt.show()
print(f'Market value: peak £{mv.max():,.0f}m ({mv.idxmax():%b %Y}); last £{mv.iloc[-1]:,.0f}m ({mv.index[-1]:%b %Y})')"""


def build_quantity_cells(ctx):
    i, ql, ch = ctx["info"], ctx["qlife"], ctx["qchanges"]
    cells = []
    text = "## Issuance and Amount Outstanding\n\n"
    if not ql:
        text += "The database has no amounts outstanding for this stock."
        return [new_markdown_cell(text)]
    text += (f"The nominal amount outstanding is recorded monthly from {fmt_month(ql['first_date'])} to "
             f"{fmt_month(ql['last_date'])}. It started at {fmt_gbp(ql['first'])}, peaked at {fmt_gbp(ql['peak'])} "
             f"in {fmt_month(ql['peak_date'])} and ended at {fmt_gbp(ql['last'])}.")
    moves = []
    if ql["n_up"]:
        moves.append(f"rose in {plural(ql['n_up'], 'month')} (adding {fmt_gbp(ql['issued_after_first'])} in all)")
    if ql["n_down"]:
        moves.append(f"fell in {plural(ql['n_down'], 'month')} (removing {fmt_gbp(ql['reduced'])})")
    text += (" Over the period the amount " + " and ".join(moves) + "." if moves else
             " The amount did not change while it was recorded.")
    if "idx_last" in ql:
        text += (f" Including the RPI uplift, the indexed amount peaked at {fmt_gbp(ql['idx_peak'])} "
                 f"({fmt_month(ql['idx_peak_date'])}).")
    if i["start"] < pd.Timestamp("1946-01-31") and ql["first_date"] <= pd.Timestamp("1946-01-31"):
        text += " (The amounts series begin in January 1946, so the stock's earlier history is not covered.)"
    if ch:
        rows = [(fmt_month(c["date"]), f"{c['change'] / 1e6:+,.0f}", f"{c['pct']:+.0f}%", fmt_gbp(c["after"]), c["kind"])
                for c in ch[:25]]
        text += ("\n\n### Large changes\n\nMonths in which the amount changed by at least £50m and 5%. The "
                 "classification is inferred from the data and the tranche records.\n\n"
                 + md_table(["Month", "Change (£m)", "Change", "Outstanding after", "Likely cause"], rows))
        if len(ch) > 25:
            text += f"\n*…and {len(ch) - 25} more.*\n"
    text += ENRICH.format("Explain the issuance history: how the stock was sold (tender, tap, auction or "
                          "syndication), who bought the large further issues and why, and what caused the reductions "
                          "(conversion offers, purchases by the National Debt Commissioners, buy-backs or "
                          "redemption).")
    cells.append(new_markdown_cell(text))
    cells.append(new_code_cell(QUANT_CODE))
    if ctx["has_price"]:
        cells.append(new_code_cell(MV_CODE))
    return cells


PRICE_CODE = """\
# Month-end clean price
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(price.index, price, color=C_BOND, lw=1.6, label=PRICE_LABEL)
if len(price_pp):
    ax.plot(price_pp.index, price_pp, color=C_ACC, lw=1.4, ls=':', label='Partly-paid price')
ax.axhline(100, color='black', lw=0.8, alpha=0.4, ls='--', label='Par (£100)')
ax.set_ylabel('£ per £100 nominal')
ax.set_title(f'{NAME}: month-end price')
ax.legend(loc='best', fontsize=9)
mark_events(ax)
plt.tight_layout()
plt.show()"""

IL_PRICE_CODE = """\
# Index-linked: separate the inflation uplift from the real price
ir = index_ratio(db, BOND_ID, price.index)
if is_old_style_il(info, BOND_ID):
    nominal_px, real_px = price, price / ir.values
else:
    nominal_px, real_px = price * ir.values, price
fig, (ax, axr) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={'height_ratios': [3, 1.3]})
ax.plot(nominal_px.index, nominal_px, color=C_BOND, lw=1.6, label='Nominal price (with RPI uplift)')
ax.plot(real_px.index, real_px, color=C_REAL, lw=1.6, label='Real price (base-date pounds)')
ax.axhline(100, color='black', lw=0.8, alpha=0.4, ls='--')
ax.set_ylabel('£ per £100 nominal')
ax.legend(loc='upper left', fontsize=9)
ax.set_title(f'{NAME}: nominal and real price')
axr.plot(ir.index, ir, color=C_ACC, lw=1.5)
axr.set_ylabel('Index ratio')
plt.tight_layout()
plt.show()"""


def build_price_cells(ctx):
    i, f, ps = ctx["info"], ctx["flags"], ctx["pstats"]
    if not ctx["has_price"]:
        msg = "## Market Price\n\nThe database has no month-end prices for this stock"
        if i["end_for_ranges"] < pd.Timestamp("1975-11-30"):
            msg += " - it left the market before the price series begin in November 1975."
        else:
            msg += "."
        return [new_markdown_cell(msg)]
    text = (f"## Market Price\n\n**Overall ({fmt_month(ps['first'])} to {fmt_month(ps['last'])}):** mean "
            f"{fmt_px(ps['mean'])}, standard deviation {ps['std']:.2f}, low {fmt_px(ps['min'])} "
            f"({fmt_month(ps['min_date'])}), high {fmt_px(ps['max'])} ({fmt_month(ps['max_date'])}); the stock was at "
            f"or above par in {ps['above_par']:.0f}% of months. Prices are month-end clean prices per £100 nominal"
            + (" (source prices before March 1986 converted from dirty to clean; see Data Notes)"
               if "dirty-converted" in f else "")
            + ("; the partly-paid period is shown separately." if len(ctx["pp"]) else "."))
    if "il-old" in f:
        text += " Old-style index-linked prices include the RPI uplift, so they are nominal prices."
    if "il-new" in f:
        text += " New-style index-linked prices are real prices (before the RPI uplift)."
    if ctx["era_stats"]:
        rows = [(e["era"], f"{fmt_month(e['from'])} - {fmt_month(e['to'])}", e["n"], f"{e['mean']:.2f}",
                 f"{e['min']:.2f}", f"{e['max']:.2f}", "" if pd.isna(e["yield"]) else f"{e['yield']:.2f}%")
                for e in ctx["era_stats"]]
        text += "\n\n**By era:**\n\n" + md_table(["Era", "Months", "Obs", "Mean", "Min", "Max", "Mean yield"], rows)
    text += ENRICH.format("Interpret the price path: the pull to par as redemption approached, the effect of the "
                          "coupon relative to market yields, and the episodes behind the highs and lows.")
    cells = [new_markdown_cell(text), new_code_cell(PRICE_CODE)]
    if i["l1"] == "Index-linked":
        cells.append(new_code_cell(IL_PRICE_CODE))
    return cells


YIELD_CODE = """\
# Yield against the peer group; lower panel = spread over the peer median (positive = cheaper than peers)
y = yields['yield'].dropna()
has_peers = len(peers) > 0
if has_peers:
    fig, (ax, axs) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1.3]})
else:
    fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(y.index, y, color=C_BOND, lw=1.8, label=f'{NAME}: {YIELD_LABEL.lower()}')
if has_peers:
    ax.fill_between(peers.index, peers['q25'], peers['q75'], color=C_PEER, alpha=0.25, label='Peers: inter-quartile range')
    ax.plot(peers.index, peers['median'], color=C_PEER, lw=1.2, label=f'Peers: median ({PEER_LABEL})')
    spread = ((y - peers['median']) * 100).dropna()
    axs.bar(spread.index, spread, width=25, color=np.where(spread > 0, C_ACC, C_BOND))
    axs.axhline(0, color='black', lw=0.8)
    axs.set_ylabel('Spread (bp)')
ax.set_ylabel('% a year')
ax.set_title(f'{NAME}: {YIELD_LABEL.lower()} and peers')
ax.legend(loc='best', fontsize=9)
mark_events(ax)
plt.tight_layout()
plt.show()"""


def build_yield_cells(ctx):
    i, f, ys, rv = ctx["info"], ctx["flags"], ctx["ystats"], ctx["rv"]
    if not ctx["has_yield"]:
        if "variable" in f:
            return [new_markdown_cell("## Yield\n\nThe coupon of a variable- or floating-rate stock reset with "
                                      "short-term interest rates, so it has no fixed-rate redemption yield to compare "
                                      "with other gilts.")]
        return []
    how = {"gross redemption": "the gross redemption yield: the discount rate that equates the price plus accrued "
                               "interest to the remaining coupons and principal, compounded "
                               + ("semi-annually" if i["freq"] == 2 else f"{i['freq']} times a year")
                               + (". For a double-dated stock it is measured to the earliest date when the price is "
                                  "at or above par and to the final date otherwise" if "double-dated" in f else ""),
           "flat": "the flat (running) yield, coupon ÷ price, the standard measure for an undated stock",
           "real": ("the real yield. For this old-style stock, RPI values not yet published are projected at 3% a "
                    "year into the eight-month-lagged cash flows, the nominal redemption yield is solved from the "
                    "price, and it is converted to a real yield with the same 3%" if "il-old" in f else
                    "the real yield: the redemption yield on the real (unindexed) price and cash flows")}[ys["kind"]]
    if ctx["long_only"]:
        peer = "long-dated conventional gilts with at least 15 years to redemption"
    elif ctx["group"] == ("Index-linked",):
        peer = ("index-linked gilts whose remaining life is within 35% (at least three years) of this stock's "
                "(at least two of them)")
    else:
        peer = "conventional gilts whose remaining life is within 15% (at least one year) of this stock's"
    text = (f"## Yield and Relative Value\n\nThe yield shown is {how}. Peers are {peer}, measured the same way; "
            "yields within three months of redemption are dropped.\n\n")
    text += (f"The {ys['kind']} yield was {fmt_pct(ys['first'])} in {fmt_month(ys['first_date'])} and "
             f"{fmt_pct(ys['last'])} in {fmt_month(ys['last_date'])}; it peaked at {fmt_pct(ys['max'])} "
             f"({fmt_month(ys['max_date'])}) and was lowest at {fmt_pct(ys['min'])} ({fmt_month(ys['min_date'])}).")
    if rv:
        text += (f" Against the peer median (typically {rv['peer_n']:.0f} stocks) the spread averaged "
                 f"{rv['mean_bp']:+.0f} bp (median {rv['median_bp']:+.0f} bp); the stock yielded less than its peers "
                 f"- traded 'rich' - in {rv['rich_share']:.0f}% of months. The widest spreads were "
                 f"{rv['max_bp']:+.0f} bp ({fmt_month(rv['max_date'])}) and {rv['min_bp']:+.0f} bp "
                 f"({fmt_month(rv['min_date'])}).")
        if len(rv["by_era"]) > 1:
            text += "\n\n" + md_table(["Era", "Months", "Mean spread (bp)"],
                                      [(n, k, f"{m:+.0f}") for n, k, m in rv["by_era"]])
        if "low-coupon" in f:
            text += ("\n*A persistently negative spread for a low-coupon stock is the tax effect described above, "
                     "not necessarily mispricing.*\n")
    text += ENRICH.format("Explain the level of the yield relative to peers: coupon and tax effects, liquidity "
                          "(size, benchmark status), call features, index-linked demand from pension funds, or "
                          "episodes of official buying and selling.")
    return [new_markdown_cell(text), new_code_cell(YIELD_CODE)]


CURVE_CODE = """\
# The stock on the yield curve: every peer-group gilt on each date, with a fitted Nelson-Siegel curve
fig, axes = plt.subplots(1, len(SNAPSHOTS), figsize=(5.2 * len(SNAPSHOTS), 4.6), squeeze=False)
for ax, d in zip(axes[0], SNAPSHOTS):
    d = pd.Timestamp(d)
    pts = pd.DataFrame([(v.at[d, 'remaining'], v.at[d, 'yield']) for k, v in panel.items()
                        if k != BOND_ID and d in v.index], columns=['T', 'y']).dropna()
    ax.scatter(pts['T'], pts['y'], s=14, color=C_PEER, alpha=0.8, label='Other gilts')
    curve = nelson_siegel(pts['T'], pts['y'])
    if curve is not None:
        tt = np.linspace(max(0.25, pts['T'].min()), pts['T'].max(), 200)
        ax.plot(tt, curve(tt), color=C_PEER, lw=1.2, label='Nelson-Siegel fit')
    own = yields.loc[d]
    T_own = own['remaining'] if pd.notna(own.get('remaining', np.nan)) else pts['T'].max() + 3
    ax.scatter([T_own], [own['yield']], s=120, marker='*', color=C_ACC, zorder=3, label=NAME)
    if curve is not None and pd.notna(own.get('remaining', np.nan)):
        print(f"{d:%b %Y}: yield {own['yield']:.2f}%, fitted curve {float(curve(T_own)):.2f}%, "
              f"residual {100 * (own['yield'] - float(curve(T_own))):+.0f} bp")
    ax.set_title(f'{d:%b %Y}')
    ax.set_xlabel('Years to redemption' + (' (undated shown at right)' if pd.isna(own.get('remaining', np.nan)) else ''))
    ax.set_ylabel('Yield (% a year)')
axes[0][0].legend(fontsize=8, loc='best')
plt.tight_layout()
plt.show()"""


def build_curve_cells(ctx):
    if not ctx["has_yield"] or not ctx["snapshots"]:
        return []
    dates = "; ".join(f"{fmt_month(d)} ({why})" for d, why in ctx["snapshots"])
    text = (f"## On the Yield Curve\n\nSnapshots of every gilt in the peer group on {len(ctx['snapshots'])} "
            f"date{'s' if len(ctx['snapshots']) > 1 else ''}: {dates}. The grey line is a Nelson-Siegel curve "
            "fitted to the other gilts; the distance of the star from the line shows whether the stock was cheap "
            "(above) or rich (below) relative to the curve.")
    text += ENRICH.format("Comment on the shape of the curve at each date (upward-sloping, flat or inverted) and "
                          "what it says about monetary policy and inflation expectations at the time.")
    return [new_markdown_cell(text), new_code_cell(CURVE_CODE)]


RETURN_CODE = """\
# Total return to a holder who bought at the first month-end price and reinvested coupons
tr = total_return_index(db, BOND_ID)
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(tr.index, tr['nominal'], color=C_BOND, lw=1.8, label='Nominal total return')
if 'real' in tr and tr['real'].notna().any():
    ax.plot(tr.index, tr['real'], color=C_REAL, lw=1.8, label='Real total return (deflated by RPI)')
ax.axhline(100, color='black', lw=0.8, alpha=0.4)
ax.set_yscale('log')
ax.set_ylabel('Index, start = 100 (log scale)')
ax.set_title(f'{NAME}: total return to holders')
ax.legend(loc='upper left', fontsize=9)
mark_events(ax)
plt.tight_layout()
plt.show()
yrs = (tr.index[-1] - tr.index[0]).days / 365.25
print(f"Nominal: x{tr['nominal'].iloc[-1] / 100:.2f} over {yrs:.1f} years = "
      f"{100 * ((tr['nominal'].iloc[-1] / 100) ** (1 / yrs) - 1):.2f}% a year")
if 'real' in tr and tr['real'].notna().sum() > 12:
    rr = tr['real'].dropna()
    ry = (rr.index[-1] - rr.index[0]).days / 365.25
    print(f"Real (from {rr.index[0]:%b %Y}): x{rr.iloc[-1] / rr.iloc[0]:.2f} = "
          f"{100 * ((rr.iloc[-1] / rr.iloc[0]) ** (1 / ry) - 1):.2f}% a year")"""

RISK_CODE = """\
# Risk: modified duration, rolling 12-month volatility of monthly returns, and drawdown from the previous peak
ret = tr['nominal'].pct_change()
vol = ret.rolling(12, min_periods=10).std() * np.sqrt(12) * 100
dd = (tr['nominal'] / tr['nominal'].cummax() - 1) * 100
dur = yields['duration'].dropna() if 'duration' in yields else pd.Series(dtype=float)
panels = [p for p in ('dur', 'vol', 'dd') if p != 'dur' or len(dur)]
fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.9 * len(panels)), sharex=True, squeeze=False)
for ax, p in zip(axes[:, 0], panels):
    if p == 'dur':
        ax.plot(dur.index, dur, color=C_BOND, lw=1.5)
        ax.set_ylabel('Modified duration (years)')
    elif p == 'vol':
        ax.plot(vol.index, vol, color=C_ACC, lw=1.5)
        ax.set_ylabel('Volatility (% a year)')
    else:
        ax.fill_between(dd.index, dd, 0, color=C_ACC, alpha=0.3)
        ax.set_ylabel('Drawdown (%)')
axes[0, 0].set_title(f'{NAME}: interest-rate risk and realised risk')
plt.tight_layout()
plt.show()"""


def build_returns_cells(ctx):
    ret, f = ctx["ret"], ctx["flags"]
    if not ret:
        if ctx["has_price"] and "variable" in f:
            return [new_markdown_cell("## Returns and Risk to Holders\n\nThe database does not record the reset "
                                      "coupons of this stock, so total returns cannot be computed.")]
        return []
    text = (f"## Returns and Risk to Holders\n\nAn investor who bought at the first month-end price "
            f"({fmt_month(ret['from'])}) and reinvested every coupon held {ret['mult']:.2f} times the starting value "
            f"by {fmt_month(ret['to'])}: {ret['ann']:.2f}% a year in nominal terms over {ret['years']:.1f} years.")
    if "real_ann" in ret:
        text += (f" In real terms (deflating by the RPI from {fmt_month(ret['real_from'])}) the return was "
                 f"{ret['real_ann']:.2f}% a year, a cumulative factor of {ret['real_mult']:.2f}.")
    else:
        text += " The RPI series in the database begins in March 1980, so real returns are not shown."
    text += (f" Returns were volatile at {ret['vol']:.1f}% a year; the worst fall from a previous peak was "
             f"{ret['max_dd']:.1f}% ({fmt_month(ret['dd_peak'])} to {fmt_month(ret['dd_date'])}).")
    if "roll_max" in ret:
        text += (f" Twelve-month volatility peaked at {ret['roll_max']:.1f}% a year around "
                 f"{fmt_month(ret['roll_max_date'])}.")
    text += ("\n\n*Method: monthly return = (change in price + coupon accrued over the month) ÷ previous price, with "
             "index-linked coupons and new-style prices uplifted by the index ratio. Taxes and dealing costs are "
             "ignored; partly-paid months are excluded.*")
    text += ENRICH.format("Put the returns in perspective: compare with inflation, Bank Rate and equities over the "
                          "same years (e.g. Dimson, Marsh and Staunton 2002), and ask who gained and who lost - "
                          "the Treasury or the holders.")
    return [new_markdown_cell(text), new_code_cell(RETURN_CODE), new_code_cell(RISK_CODE)]


def build_events_cells(ctx):
    if not ctx["has_price"]:
        return []
    ev = ctx["pevents"]
    text = "## Market Events\n\n"
    na = lambda x, f: "n/a" if pd.isna(x) else f.format(x)
    if ev and ev[0]["use_yield"]:
        text += ("Months in which the price moved by 5% or more, or the stock's yield moved at least 40 basis points "
                 "more than the median yield of its peers (the twelve largest, in date order). Comparing yield "
                 "changes with maturity-matched peers separates moves specific to this stock from market-wide "
                 "moves. The nearest event in the knowledge base is listed as a lead, not as a cause.\n\n")
        rows = [(fmt_month(e["date"]), na(e["ret"], "{:+.1f}%"), na(e["dy"], "{:+.0f}"), na(e["dpeer"], "{:+.0f}"),
                 na(e["rel"], "{:+.0f}"), e["event"] or "none in the knowledge base") for e in ev]
        text += md_table(["Month", "Price change", "Yield change (bp)", "Peer median change (bp)",
                          "Relative (bp)", "Nearest event"], rows)
    elif ev:
        text += ("Months in which the price moved by 5% or more, or by at least 3 percentage points more than the "
                 "median gilt of the same kind (the twelve largest, in date order). The nearest event in the "
                 "knowledge base is listed as a lead, not as a cause.\n\n")
        rows = [(fmt_month(e["date"]), na(e["ret"], "{:+.1f}%"), na(e["market"], "{:+.1f}%"), na(e["rel"], "{:+.1f}"),
                 e["event"] or "none in the knowledge base") for e in ev]
        text += md_table(["Month", "Price change", "Median gilt", "Excess (pts)", "Nearest event"], rows)
    if ev:
        text += ENRICH.format("Explain these moves. Separate market-wide shocks (where the peers moved too) from "
                              "moves specific to this stock (conversion offers, taps, calls, index-linking news, tax "
                              "changes, redemption announcements).")
    else:
        text += "No month-end price change of 5% or more, and no large move relative to comparable gilts, was detected."
    return [new_markdown_cell(text)]


RELATED_CODE = """\
# Yields of related stocks, measured the same way
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(yields.index, yields['yield'], color=C_BOND, lw=2.4, label=NAME)
for k, name in RELATED.items():
    yk = bond_yields(db, k)['yield'].dropna()
    ax.plot(yk.index, yk, lw=1.1, alpha=0.85, label=f'{name} (ID {k})')
ax.set_ylabel('% a year')
ax.set_title(f'{NAME} and related stocks: {YIELD_LABEL.lower()}')
ax.legend(loc='best', fontsize=8)
mark_events(ax)
plt.tight_layout()
plt.show()"""


def build_related_cells(ctx):
    rel = ctx["related"]
    text = "## Related Stocks\n\n"
    if not rel:
        return [new_markdown_cell(text + "No related stocks found.")]
    for group, items in rel.items():
        rows = [(r["id"], r["name"], fmt_date(r["issue"]) if r["issue"] is not None else "n/a",
                 fmt_date(r["maturity"]) if r["maturity"] is not None else "undated/n/a", r["rel"], r["n_prices"])
                for r in items]
        text += f"**{group}**\n\n" + md_table(["ID", "Stock", "First issued", "Maturity", "Relationship",
                                               "Price months"], rows) + "\n"
    text += ENRICH.format("Compare this stock with its closest relatives: why did the Treasury issue these stocks "
                          "side by side, and how did coupon, maturity or special terms make them trade differently?")
    cells = [new_markdown_cell(text)]
    if ctx["related_chart"] and ctx["has_yield"]:
        cells.append(new_code_cell(RELATED_CODE))
    return cells


def build_distribution_cells(ctx):
    i, f = ctx["info"], ctx["flags"]
    d = i["issue"]
    if d is None:
        how = "The database does not record how or when this stock was first sold."
    elif d < pd.Timestamp("1939-09-01"):
        how = ("Before the Second World War new stocks were generally offered for public subscription at a fixed "
               "price or created through conversion offers to holders of maturing debt.")
    elif d < pd.Timestamp("1987-05-01"):
        how = ("Stocks of this period were sold mainly through the Bank of England's tap system: the Bank took up "
               "new issues and the Government Broker sold them to the market (to the jobbers before October 1986, "
               "then to gilt-edged market makers) at prices set by the Bank, when demand allowed.")
    elif d < pd.Timestamp("2005-09-01"):
        how = ("By this time gilts were sold mainly by auction (from 1987), supplemented by taps; the Debt "
               "Management Office took over from the Bank of England in April 1998.")
    else:
        how = ("The Debt Management Office sold gilts by auction and, for large long-dated and index-linked "
               "operations, by syndication through a group of banks (from 2005).")
    if i["status"] == "redeemed":
        red = f"The stock was redeemed on {fmt_date(i['final_red'])}"
        if "double-dated" in f and i["early"] is not None:
            red += (" at the Treasury's option, before its final date" if i["final_red"] < i["payable"] - pd.Timedelta(days=45)
                    else ", at the end of its call window")
        elif "undated" in f:
            red += ", in the 2014-15 programme that repaid all the undated gilts"
        red += "."
    elif i["status"] == "amalgamated":
        red = f"As a separate line the stock ended on {fmt_date(i['amalg'])}, when it was amalgamated with the parent."
    elif i["status"] == "outstanding":
        red = "The stock was still outstanding at the end of the data."
    else:
        red = "The data do not record the final redemption."
    ch = ctx["qchanges"]
    downs = [c for c in ch if c["change"] < 0 and c["after"] > 0]
    if downs:
        red += (f" Before that, {plural(len(downs), 'large reduction')} in the amount outstanding "
                "(see the table above) point to conversion offers, purchases or buy-backs.")
    text = f"## Issuance, Distribution and Redemption\n\n{how}\n\n{red}"
    text += ENRICH.format("Describe who held the stock (banks, insurance companies, pension funds, overseas "
                          "holders, the Bank of England and, after 2009, the Asset Purchase Facility) and how its "
                          "redemption was financed.")
    return [new_markdown_cell(text)]


def build_implications_cells(ctx):
    i, f, ret = ctx["info"], ctx["flags"], ctx["ret"]
    qs = ["What does this stock's history reveal about how the British state borrowed in its era?"]
    if ret and "real_ann" in ret:
        qs.append(f"Holders earned {ret['real_ann']:.1f}% a year in real terms: who bore the cost of inflation, "
                  "and was this part of a policy of 'financial repression' or inflating away the debt?")
    if "undated" in f:
        qs.append("Why did the undated debt survive for so long, and what did its 2015 redemption signal?")
    if "double-dated" in f or "convertible" in f:
        qs.append("Did the Treasury use its options well - was the call or conversion exercised when it paid to?")
    if "il-old" in f or "il-new" in f:
        qs.append("What did index-linking teach the Treasury about inflation risk and the credibility of policy?")
    if "nationalisation" in f:
        qs.append("How did compensation in fixed-interest stock distribute the gains and losses of nationalisation?")
    text = "## Implications and Legacy\n\n" + "\n".join(f"- {q}" for q in qs)
    text += ENRICH.format("Write 2-3 paragraphs answering these questions, linking the stock to the long-run "
                          "themes of Ellison and Scott (2020): the maturity structure of the debt, the cost of "
                          "financing, and the interaction of debt management with monetary policy.")
    return [new_markdown_cell(text)]


def build_data_notes_cells(ctx):
    i, c, f = ctx["info"], ctx["corrections"], ctx["flags"]
    text = (f"## Data Notes\n\n**Source:** {i['source']} (Ellison-Scott database, as assembled in "
            f"`UK_Gilts_Bond_Database.xlsx`).")
    if i["notes"]:
        text += f"  \n**Notes on this stock:** {i['notes']}"
    if len(c):
        rows = [(r["Item"], r["Month / Date"], r["Original Value"], r["New Value"], r["Reason / Source"])
                for _, r in c.head(20).iterrows()]
        rows = [tuple("" if (x is None or (isinstance(x, float) and np.isnan(x))) else x for x in row) for row in rows]
        text += ("\n\n**Corrections applied to the source files that affect this stock:**\n\n"
                 + md_table(["Item", "Month / date", "Original", "New", "Reason"], rows))
    caveats = ["Prices are month-end clean prices per £100 nominal; amounts are month-end nominal amounts "
               "outstanding in £.",
               "Yields are computed in this chapter from the prices and terms (actual coupon dates, accrued "
               "interest); they can differ slightly from officially published yields.",
               "The October 2022 prices in the source files were copies of September 2022 and have been removed."]
    if "dirty-converted" in f:
        caveats.append("Before 28 February 1986 the source quotes stocks with more than five years to maturity (and "
                       "undated stocks) with accrued interest included - their prices drop by about half a coupon "
                       "when they go ex-dividend. This chapter converts those prices to a clean basis (subtracting "
                       "accrued interest, or adding rebate interest when ex-dividend) before computing prices, "
                       "yields and returns; `series(db, 'price', BOND_ID, 'Average')` gives the source prices.")
    if "il-old" in f:
        caveats.append("Old-style index-linked real yields assume 3% future RPI inflation. When inflation is far from "
                       "3% (as in 2022-23) they are less comparable with new-style real yields, which need no "
                       "assumption.")
    if "limited-metadata" in f:
        caveats.append("The terms of this early stock come from limited metadata and may be incomplete.")
    text += "\n\n**Caveats:**\n\n" + "\n".join(f"- {x}" for x in caveats)
    return [new_markdown_cell(text)]


def build_references_cells(ctx):
    refs = ctx["refs"]
    text = "## References\n\n" + "\n".join(f"- {r}" for r in refs)
    text += ENRICH.format("Add specific sources for this stock: the prospectus or Treasury announcement, Bank of "
                          "England Quarterly Bulletin commentary, Hansard debates, and DMO or Bank records.")
    return [new_markdown_cell(text)]


def generate_chapter_notebook(ctx):
    nb = new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"},
                   "gilt_biography": {"bond_id": ctx["info"]["id"], "name": ctx["info"]["name"],
                                      "generator": "uk_bond_biography_agent.py"}}
    builders = [build_title_cells, build_setup_cells, build_glance_cells, build_overview_cells,
                build_instrument_cells, build_context_cells, build_timeline_cells, build_quantity_cells,
                build_price_cells, build_yield_cells, build_curve_cells, build_returns_cells, build_events_cells,
                build_related_cells, build_distribution_cells, build_implications_cells, build_data_notes_cells,
                build_references_cells]
    for b in builders:
        nb.cells.extend(b(ctx))
    return nb


# ─── Enrichment prompt ─────────────────────────────────────────────────────────

def build_enrichment_prompt(ctx, chapter_path):
    i, f, ps, ql, ys = ctx["info"], ctx["flags"], ctx["pstats"], ctx["qlife"], ctx["ystats"]
    facts = [f"Type: {describe_kind(i, f)}; category {i['l1']} / {i['l2']}."]
    if i["issue"] is not None:
        facts.append(f"First issued {fmt_date(i['issue'])} (Prime Minister: {prime_minister_on(i['issue'])}).")
    if i["status"] == "redeemed":
        facts.append(f"Redeemed {fmt_date(i['final_red'])}.")
    elif i["status"] == "outstanding":
        facts.append("Still outstanding at the end of the data (2023).")
    if ql:
        facts.append(f"Peak nominal outstanding {fmt_gbp(ql['peak'])} ({fmt_month(ql['peak_date'])}).")
    if ps:
        facts.append(f"Prices {fmt_month(ps['first'])}-{fmt_month(ps['last'])}, range {fmt_px(ps['min'])}-"
                     f"{fmt_px(ps['max'])}.")
    if ys:
        facts.append(f"{ys['kind'].capitalize()} yield range {fmt_pct(ys['min'])}-{fmt_pct(ys['max'])}.")
    if i["special"]:
        facts.append(f"Special features in the source: {i['special']}.")
    if i["id"] in BOND_NOTES:
        facts.append(BOND_NOTES[i["id"]])
    topics = ["why the stock was issued (fiscal position, funding policy, the Chancellor of the day)",
              "its terms and how they shaped demand", "the events behind the largest price and yield moves",
              "how its life ended"]
    for flag, t in [("double-dated", "whether and when the Treasury's call option was worth exercising"),
                    ("undated", "the history of the undated debt and the 2014-15 redemption programme"),
                    ("partly-paid", "why the stock was sold partly paid"),
                    ("convertible", "the conversion option and whether holders used it"),
                    ("il-old", "the introduction of index-linked gilts and the eight-month lag"),
                    ("il-new", "the 2005 switch to the three-month-lag design and pension-fund (LDI) demand"),
                    ("nationalisation", "the nationalisation and the compensation terms"),
                    ("low-coupon", "the tax effect on low-coupon gilts"),
                    ("green", "the Green Financing Framework and the allocation of proceeds")]:
        if flag in f:
            topics.append(t)
    refs = "\n".join(f"  - {r}" for r in ctx["refs"])
    stem = Path(chapter_path).stem
    return f"""You are enriching an auto-generated "gilt biography" Jupyter notebook into a polished,
publication-quality chapter for a book of UK government bond biographies based on the Ellison-Scott UK
gilt database (Ellison and Scott, AEJ: Macroeconomics 2020). Work in the folder that contains
UK_Gilts_Bond_Database.xlsx.

YOUR GILT: L1 ID {i['id']} - "{i['name']}", coupon {coupon_str(i['coupon'])}.
Key facts from the database:
{chr(10).join('- ' + x for x in facts)}

STEP 1 - Read the raw draft `{chapter_path}` in full. It contains auto-generated tables, charts,
statistics and `<!-- ENRICH: ... -->` markers where hand-written narrative is needed.

STEP 2 - Write the enhanced notebook `chapters/enhanced/{stem}_enhanced.ipynb`. Rules:
- Keep ALL code cells essentially unchanged (they load the database and draw the charts).
- Replace every `<!-- ENRICH -->` marker with narrative economic history in a scholarly but readable
  voice, in British English. Cover at minimum: {'; '.join(topics)}.
- Interpret the charts in prose (price path, yield versus peers, returns, amount outstanding).
- ACCURACY IS PARAMOUNT: state only well-established facts. Do not invent quotations, figures, dates
  or archival references. Where a detail is uncertain, write in general terms.
- Cite real, standard works. Suggested:
{refs}
- Keep the notebook valid nbformat 4 JSON (build it with nbformat and validate).

STEP 3 - Verify by executing it:
    jupyter nbconvert --to notebook --execute --inplace "chapters/enhanced/{stem}_enhanced.ipynb"
The notebook finds the database in its folder or a parent folder (or via $UK_GILTS_DB). Fix any
errors and re-run until every cell executes.

Return a short report: file written, cell count, ENRICH markers replaced, confirmation that execution
succeeded, and any historical claims deliberately kept general.
"""


# ─── Search and listing ────────────────────────────────────────────────────────

def search_bonds(BondList, query):
    name = BondList["Treasury's Name Of Issue"].astype(str)
    mask = name.str.contains(query, case=False, na=False, regex=False) | \
        name.map(pretty_name).str.contains(query, case=False, regex=False) | \
        BondList["Category L2"].astype(str).str.contains(query, case=False, regex=False)
    if query.isdigit():
        mask |= BondList.index == int(query)
    return BondList[mask][["Treasury's Name Of Issue", "Category L1", "First Issue Date", "Coupon Rate",
                           "Final Redemption Date"]].copy()


def list_categories(BondList):
    return BondList.groupby(["Category L1", "Category L2"]).size()


def list_candidates(db, n):
    """Rank stocks by how much a biography can say: price history, size, special features."""
    rows = []
    for bid in db["list"].index:
        info = get_bond_info(db, bid)
        if info["parent"]:
            continue
        flags = detect_flags(info)
        npx = len(series(db, "price", bid, "Average"))
        q = series(db, "quant", bid, "Total Outstanding")
        peak = q.max() if len(q) else 0
        feats = flags & {"double-dated", "undated", "partly-paid", "has-tranches", "convertible", "il-old", "il-new",
                         "variable", "nationalisation", "death-duties", "sinking-fund", "green", "low-coupon"}
        score = min(npx, 480) / 12 + 6 * len(feats) + 4 * np.log10(max(peak, 1e6) / 1e6) + \
            (10 if bid in BOND_NOTES else 0)
        rows.append((score, bid, info["name"], npx, peak, ", ".join(sorted(feats))))
    rows.sort(reverse=True)
    print(f"{'Score':>6} {'ID':>6}  {'Stock':<42} {'Prices':>6} {'Peak':>10}  Features")
    print("-" * 110)
    for score, bid, name, npx, peak, feats in rows[:n]:
        print(f"{score:6.1f} {bid:>6}  {name:<42} {npx:>6} {fmt_gbp(peak):>10}  {feats}")


# ─── Output ────────────────────────────────────────────────────────────────────

def execute_notebook(path, db_path):
    from nbconvert.preprocessors import ExecutePreprocessor
    os.environ["UK_GILTS_DB"] = str(Path(db_path).resolve())
    nb = nbformat.read(path, as_version=4)
    ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(Path(path).parent)}})
    nbformat.write(nb, path)


def generate_one(db, cache, bond_id, out_dir, output=None, execute=False, print_prompt=False):
    ctx = analyse(db, bond_id, cache)
    if ctx is None:
        print(f"Error: bond ID {bond_id} not found.")
        return None
    i = ctx["info"]
    print(f"\nGenerating gilt biography: {i['pretty']} (ID {bond_id})")
    print(f"  Category: {i['l1']} / {i['l2']};  issued {fmt_date(i['issue'])};  status: {i['status']}")
    print(f"  Prices: {len(ctx['price'])} months;  amounts: {len(ctx['quant'])} months;  "
          f"yields: {'yes' if ctx['has_yield'] else 'no'};  features: {', '.join(sorted(ctx['flags'])) or 'none'}")
    print(f"  Events: {len(ctx['events'])};  price events: {len(ctx['pevents'])};  large amount changes: "
          f"{len(ctx['qchanges'])};  related: {sum(len(v) for v in ctx['related'].values())};  "
          f"references: {len(ctx['refs'])}")
    nb = generate_chapter_notebook(ctx)
    nbformat.validate(nb)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (output or f"chapter_{bond_id}_{safe_filename(i['name'])}.ipynb")
    nbformat.write(nb, path)
    print(f"  Saved: {path}  ({len(nb.cells)} cells, "
          f"{sum(c.source.count('<!-- ENRICH') for c in nb.cells if c.cell_type == 'markdown')} ENRICH markers)")
    if execute:
        try:
            execute_notebook(path, db["path"])
            print("  Executed successfully.")
        except Exception as exc:  # report and carry on with the other chapters
            print(f"  Execution FAILED: {str(exc).strip().splitlines()[-1] if str(exc).strip() else exc!r}")
    if print_prompt:
        print("\n" + "=" * 78 + "\n" + build_enrichment_prompt(ctx, f"{out_dir.name}/{path.name}") + "=" * 78)
    return path


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(
        description="UK Gilt Biography Agent - generate gilt biography chapters as Jupyter notebooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --bond-id 32400                      3½%% War Loan
  %(prog)s --bond-id 32400 20100 55500 --execute   several chapters, then run them
  %(prog)s --search "Consols"                   search by name or type
  %(prog)s --search "Green" --generate-first    generate the first match
  %(prog)s --list-categories                    stock types with counts
  %(prog)s --list-candidates 30                 stocks with the richest material
  %(prog)s --bond-id 32400 --print-prompt       also print an enrichment prompt
  %(prog)s --all --min-months 120               every stock with 10+ years of prices
  %(prog)s --interactive                        choose interactively
""")
    ap.add_argument("--bond-id", type=int, nargs="+", help="L1 ID(s) of the stock(s)")
    ap.add_argument("--search", type=str, help="search stocks by name or type")
    ap.add_argument("--generate-first", action="store_true", help="generate a chapter for the first search result")
    ap.add_argument("--list-categories", action="store_true", help="list stock categories with counts")
    ap.add_argument("--list-candidates", type=int, metavar="N", help="rank the N stocks with the richest material")
    ap.add_argument("--all", action="store_true", help="generate chapters for every stock (not tranches)")
    ap.add_argument("--min-months", type=int, default=0, help="with --all: minimum number of month-end prices")
    ap.add_argument("--interactive", action="store_true", help="interactive selection")
    ap.add_argument("--output", type=str, help="output file name (single stock only)")
    ap.add_argument("--output-dir", type=str, default=str(here / "chapters"), help="folder for chapters")
    ap.add_argument("--data", type=str, help="path to UK_Gilts_Bond_Database.xlsx")
    ap.add_argument("--execute", action="store_true", help="execute each notebook after writing it")
    ap.add_argument("--print-prompt", action="store_true", help="print an enrichment prompt for each chapter")
    args = ap.parse_args()

    if not any([args.bond_id, args.search, args.list_categories, args.list_candidates, args.all, args.interactive]):
        ap.print_help()
        sys.exit(0)

    data = args.data or (here / DB_FILENAME if (here / DB_FILENAME).exists() else None)
    print("Loading gilt database...")
    db = load_database(data)
    bl = db["list"]
    print(f"Loaded {len(bl)} stocks from {db['path']}\n")

    if args.list_categories:
        print("Stock categories (L1 > L2):")
        print("=" * 60)
        for (l1, l2), n in list_categories(bl).items():
            print(f"  {l1} > {l2}: {n}")
        return
    if args.list_candidates:
        list_candidates(db, args.list_candidates)
        return

    ids = []
    if args.search:
        res = search_bonds(bl, args.search)
        if len(res) == 0:
            print(f"No stocks found matching '{args.search}'")
            return
        print(f"Found {len(res)} stocks matching '{args.search}':")
        print("=" * 90)
        for idx, r in res.iterrows():
            nm = pretty_name(r[NAME_COL])
            print(f"  L1 ID {idx:>6}: {nm:<45} {r['Category L1']:<22} issued {fmt_date(r['First Issue Date'])}")
        if not args.generate_first:
            print("\nUse --bond-id <ID> to generate a chapter.")
            return
        ids = [int(res.index[0])]
    elif args.bond_id:
        ids = args.bond_id
    elif args.all:
        for bid in bl.index:
            if pd.isna(bl.at[bid, "Parent ID"]) and len(series(db, "price", bid, "Average")) >= args.min_months:
                ids.append(int(bid))
        print(f"Generating {len(ids)} chapters...")
    elif args.interactive:
        print("Interactive Gilt Biography Generator\n" + "=" * 40)
        q = input("Search for a stock (or enter L1 ID): ").strip()
        if q.isdigit():
            ids = [int(q)]
        else:
            res = search_bonds(bl, q)
            if len(res) == 0:
                print(f"No stocks found matching '{q}'")
                return
            for k, (idx, r) in enumerate(res.iterrows()):
                print(f"  [{k}] L1 ID {idx}: {pretty_name(r[NAME_COL])}")
            choice = input("\nEnter number to select (or 'q' to quit): ").strip()
            if choice.lower() == "q":
                return
            try:
                ids = [int(res.index[int(choice)])]
            except (ValueError, IndexError):
                print("Invalid selection.")
                return

    cache = Cache(db)
    written = []
    for bid in ids:
        p = generate_one(db, cache, bid, args.output_dir, args.output if len(ids) == 1 else None,
                         args.execute, args.print_prompt)
        if p:
            written.append(p)
    print(f"\n{'=' * 60}\n  {len(written)} chapter(s) written to {Path(args.output_dir).resolve()}")
    print("  Sections marked <!-- ENRICH --> need researcher input.\n" + "=" * 60)


if __name__ == "__main__":
    main()
