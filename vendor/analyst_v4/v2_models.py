"""Python mirror of every v2 workbook calculation (same definitions as the Excel formulas).
Produces: month x series calc tables, the long panel with all model forecasts, and feature-model coefficients."""
import os, sys, warnings
sys.path.insert(0, '/home/claude'); os.chdir('/home/claude'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.linear_model import HuberRegressor

VIEW = os.environ.get('VIEW', ''); SUF = f'_{VIEW}' if VIEW else ''
D = globals().get('D_OVERRIDE') or pd.read_pickle(f'v2/inputs{SUF}.pkl')
MK, MONTHS, mac = D['MK'], D['MONTHS'], D['macro']
SER = D.get('SER') or ([('White', m) for m in MK] + [('Yellow', m) for m in MK])
TYPES = [t for t in ['White', 'Yellow'] if any(s[0] == t for s in SER)]
P = pd.concat({t: D[t.lower()] for t in TYPES}, axis=1)[SER]       # month x series
FLAGS = D.get('flags')
if FLAGS is None: FLAGS = P.notna().astype(float).where(P.notna())
LP = np.log(P)
HZ = [1, 3, 6]
VINT = list(range(2015, 2026))
ORIG = pd.date_range('2016-01-01', MONTHS[-1], freq='MS')
LAST_DATA = MONTHS[-1]

# ---------------- seasonal index (workbook method, complete years only) ----------------
cpi = mac.cpi.reindex(MONTHS)
real = P.div(cpi / 100, axis=0)
yr = pd.Series(MONTHS.year, index=MONTHS)
cnt = P.notna().groupby(yr.values).transform('sum'); cnt.index = MONTHS
ysum = real.groupby(yr.values).transform('sum'); ysum.index = MONTHS
logratio = np.log(real / (ysum / 12)).where(cnt == 12)
SI = {}
for v in VINT:
    lr = logratio[yr <= v]
    raw = np.exp(lr.groupby(lr.index.month).mean())
    SI[v] = raw / raw.mean() * 100                                   # 12 x 22
def vint(t): return min(max(t.year - 1, 2015), 2025)
def si(v, m, s): return SI[v].at[m, s]

# ---------------- month x series feature tables ----------------
TM = {ty: LP[[s for s in SER if s[0] == ty]].mean(axis=1) for ty in TYPES}
REL = pd.DataFrame({s: LP[s] - TM[s[0]] for s in SER})
def prev12_mean(df): return pd.concat([df.shift(k) for k in range(1, 13)], axis=1, keys=range(12)).T.groupby(level=1).mean().T[df.columns] if False else \
    sum_mean(df)
def sum_mean(df):
    stack = np.stack([df.shift(k).values for k in range(1, 13)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); m = np.nanmean(stack, axis=0)
    return pd.DataFrame(m, index=df.index, columns=df.columns)
DEV = REL - sum_mean(REL)
D1 = LP - LP.shift(1)
ND1 = D1.mean(axis=1)
IDIO_D1 = D1.sub(ND1, axis=0)
SPRM = [m for m in MK if ('White', m) in SER and ('Yellow', m) in SER]
SPR = pd.DataFrame({m: LP[('White', m)] - LP[('Yellow', m)] for m in SPRM}, index=MONTHS)
SPR_DEV = SPR - sum_mean(SPR) if SPRM else SPR
SPRS = pd.DataFrame({s: ((SPR_DEV[s[1]] if s[0] == 'White' else -SPR_DEV[s[1]]) if s[1] in SPRM else pd.Series(np.nan, index=MONTHS)) for s in SER})
sorg = D['sorghum'].reindex(MONTHS)
RS = pd.DataFrame({s: LP[s] - (np.log(sorg[s[1]]) if s[1] in sorg.columns else np.nan) for s in SER})
RS_DEV = RS - sum_mean(RS)
# weekly
wk = D['weekly'].copy(); wk['month'] = wk.end.dt.to_period('M').dt.to_timestamp(); wk['lnp'] = np.log(wk.price_kg)
wk = wk.sort_values(['type', 'market', 'end']).reset_index(drop=True)
g = wk.groupby(['type', 'market', 'month'])
WAVG = g.lnp.mean(); WLAST = g.lnp.last(); WFIRST = g.lnp.first(); WL2 = g.lnp.apply(lambda s: s.iloc[-2:].mean())
def wtab(ser):
    t = ser.unstack(['type', 'market']).reindex(MONTHS)
    return t.reindex(columns=pd.MultiIndex.from_tuples(SER))
IN_SLOPE = wtab(WLAST - WFIRST); LW = wtab(WLAST - WAVG); L2 = wtab(WL2 - WAVG)
NAT_SLOPE = IN_SLOPE.mean(axis=1); NAT_LW = LW.mean(axis=1); NAT_L2 = L2.mean(axis=1)
par = mac.fx_parallel; off = mac.fx_official
FX3 = np.log(par / par.shift(3)); PREM = par / off - 1
# import parity gap (SI vintage = month's own year - 1)
USE_DUTY = 1
LOGI = mac['logistics'] if 'logistics' in mac else pd.Series(50.0, index=mac.index)
DUTY = mac['duty'].fillna(0) if 'duty' in mac else pd.Series(0.0, index=mac.index)
IPP = np.log((mac.world_maize + LOGI) * par / 1000 * (1 + USE_DUTY * DUTY)).reindex(MONTHS)
# upstream producing belt (lead-lag tested on data up to 2018): its move vs the national average, and its weekly momentum (last week vs 3 weeks before)
UP_BELT = [m for m in ['Dandume', 'Saminaka', 'Giwa', 'Kaura Namoda', 'Gwandu, Dodoru'] if ('White', m) in SER]
WLP = LP[[s for s in SER if s[0] == 'White']]
UPP = LP[[('White', m) for m in UP_BELT]].mean(axis=1); NATW = WLP.mean(axis=1)
UP_MOVE = UPP.diff() - NATW.diff()
_wk = D['weekly']; _wk = _wk[(_wk.type == 'White') & _wk.market.isin(UP_BELT)].copy(); _wk['lnp'] = np.log(_wk.price_kg)
_wk = _wk.groupby(['market', 'end']).lnp.mean().reset_index()
_rows = []
for _m, _s in _wk.groupby('market'):
    _s = _s.set_index('end').lnp
    for _mo, _last in _s.groupby(_s.index.to_period('M')).apply(lambda x: x.index.max()).items():
        _prev = _s[(_s.index >= _last - pd.Timedelta(days=24)) & (_s.index <= _last - pd.Timedelta(days=18))]
        _rows.append({'market': _m, 'month': _mo.to_timestamp(), 'v': _s[_last] - _prev.mean() if len(_prev) else np.nan})
UP_WK3_M = pd.DataFrame(_rows).pivot(index='month', columns='market', values='v').reindex(index=MONTHS, columns=UP_BELT) if _rows else pd.DataFrame(index=MONTHS, columns=UP_BELT, dtype=float)
UP_WK3 = UP_WK3_M.mean(axis=1)
UP_ON = {1: 0, 3: 1, 6: 0}
# import parity gap while imports are duty-free (Aug-2024 waiver onward): ln(national white price / import parity at the parallel rate)
FREE = (mac['duty'] == 0).astype(float) if 'duty' in mac else pd.Series(0.0, index=mac.index)
IPP0 = np.log((mac.world_maize + LOGI) * par / 1000)
GAP_FREE = ((NATW - IPP0.reindex(MONTHS)) * FREE.reindex(MONTHS)).where(FREE.reindex(MONTHS) == 1, 0.0)      # upstream drivers are used at 3 months only (tested: no gain at 1m, worse at 6m)
GAP = pd.DataFrame({s: [np.log(P.at[t, s] / (si(vint(t), t.month, s) / 100)) - IPP[t] for t in MONTHS] for s in SER}, index=MONTHS)
MED24 = pd.DataFrame({s: [np.nanmedian(GAP[s].loc[t - pd.DateOffset(months=23):t].values) if GAP[s].loc[t - pd.DateOffset(months=23):t].notna().any() else np.nan for t in MONTHS] for s in SER}, index=MONTHS)

# ---------------- long panel ----------------
rows = []
for t in ORIG:
    v = vint(t)
    for s in SER:
        r = dict(origin=t, type=s[0], market=s[1], zone=D['ZONE'][s[1]], vint=v, p0=P.at[t, s], mom=mac.headline_mom[t],
                 prem=PREM[t], si_o=si(v, t.month, s), si_next=si(v, (t + pd.DateOffset(months=1)).month, s))
        for h in HZ:
            tg = t + pd.DateOffset(months=h)
            r[f'tgt{h}'] = tg
            r[f'A{h}'] = P.at[tg, s] if tg <= LAST_DATA else np.nan
            r[f'obs{h}'] = FLAGS.at[tg, s] if tg <= LAST_DATA else np.nan
            r[f'si_t{h}'] = si(v, tg.month, s)
        rows.append(r)
pn = pd.DataFrame(rows)
pn['band'] = pn.zone.map(D['bands'])
for h in HZ:
    pn[f'chg{h}'] = pn[f'A{h}'] / pn.p0 - 1
    # OG models
    pn[f'V5_{h}'] = pn.p0 * (1 + pn.mom / 100) / (pn.si_next / 100) * (pn[f'si_t{h}'] / 100)
    pn[f'V7_{h}'] = pn[f'V5_{h}'] * (1 + pn.mom / 100) ** (h - 1)
    pn[f'NV_{h}'] = pn.p0
    sea = (pn[f'si_t{h}'] / pn.si_o) ** 0.5
    o6 = pn.origin - pd.DateOffset(months=6); o3 = pn.origin - pd.DateOffset(months=3)
    pn[f'FXO_{h}'] = pn.p0 * sea * np.exp(0.5 * np.log(off.reindex(pn.origin).values / off.reindex(o6).values) * (1 - 0.67 ** h))
    pn[f'FXP_{h}'] = pn.p0 * sea * np.exp(0.5 * np.log(par.reindex(pn.origin).values / par.reindex(o6).values) * (1 - 0.67 ** h))
    pn[f'FXR_{h}'] = pn.p0 * sea * np.exp(1.0 * np.log(par.reindex(pn.origin).values / par.reindex(o3).values) * (1 - 0.5 ** h))
    pn[f'LE_{h}'] = np.log(pn[f'A{h}'] / pn[f'FXR_{h}'])
    # import parity
    g_o = np.array([GAP.at[t, (ty, mk)] for t, ty, mk in zip(pn.origin, pn.type, pn.market)])
    m_o = np.array([MED24.at[t, (ty, mk)] for t, ty, mk in zip(pn.origin, pn.type, pn.market)])
    adj = np.where(np.isnan(g_o) | np.isnan(m_o), 0, (1 - 0.9 ** h) * (m_o - g_o))
    pn[f'IP_{h}'] = pn.p0 * sea * np.exp(adj)
# alert uplift (out-of-sample: past rows with prem>=40% whose target month <= this origin)
for h in HZ:
    up = np.zeros(len(pn))
    le = pn[f'LE_{h}']; ok = le.notna() & (pn.prem >= 0.4)
    for i, r in enumerate(pn.itertuples()):
        if r.prem >= 0.4:
            m = ok & (pn[f'tgt{h}'] <= r.origin)
            if m.sum() >= 22: up[i] = le[m].mean()
    pn[f'UP_{h}'] = up
    pn[f'FXA_{h}'] = pn[f'FXR_{h}'] * np.exp(up)

# ---------------- feature model ----------------
def lk(tab, t, s):
    try: return tab.at[t, s]
    except KeyError: return np.nan
for c, tab in [('f_dev', DEV), ('f_idio_d1', IDIO_D1), ('f_spr', SPRS), ('f_rsorg', RS_DEV), ('f_lw', LW), ('f_l2', L2), ('f_slope', IN_SLOPE)]:
    pn[c] = [lk(tab, t, (ty, mk)) for t, ty, mk in zip(pn.origin, pn.type, pn.market)]
pn['n_slope'] = pn.origin.map(NAT_SLOPE); pn['n_lw'] = pn.origin.map(NAT_LW); pn['n_l2'] = pn.origin.map(NAT_L2); pn['n_fx3'] = pn.origin.map(FX3)
pn['n_upmove'] = pn.origin.map(UP_MOVE); pn['n_upwk3'] = pn.origin.map(UP_WK3); pn['n_gapfree'] = pn.origin.map(GAP_FREE)
pn['f_idio_lw'] = pn.f_lw - pn.n_lw; pn['f_idio_l2'] = pn.f_l2 - pn.n_l2
for h in HZ:
    pn[f'seas{h}'] = np.where(pn.p0.notna(), np.log(pn[f'si_t{h}'] / pn.si_o), np.nan)
    pn[f'n_seas{h}'] = pn.groupby('origin')[f'seas{h}'].transform('mean')
    pn[f'f_seasrel{h}'] = pn[f'seas{h}'] - pn[f'n_seas{h}']
    pn[f'y{h}'] = np.log(pn[f'A{h}'] / pn.p0)
    pn[f'ynat{h}'] = pn.groupby('origin')[f'y{h}'].transform('mean')
NAT_F = lambda h: [f'n_seas{h}', 'n_slope', 'n_lw', 'n_fx3', 'n_upmove', 'n_upwk3', 'n_gapfree']
IDIO_F = lambda h: ['f_dev', 'f_idio_d1', 'f_spr', f'f_seasrel{h}', 'f_idio_lw', 'f_idio_l2', 'f_rsorg']

def fit(X, y):
    lo, hi = np.quantile(X, 0.02, axis=0), np.quantile(X, 0.98, axis=0)
    Xc = np.clip(X, lo, hi); mu, sd = Xc.mean(0), Xc.std(0); sd1 = np.where(sd == 0, 1, sd)
    m = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=1000).fit((Xc - mu) / sd1, y)
    b = m.coef_ / sd1; b = np.where(sd == 0, 0, b)
    b0 = m.intercept_ - np.sum(np.where(sd == 0, 0, m.coef_ * mu / sd1))
    return b0, b, lo, hi
COEF = []
for h in HZ:
    nf, idf = NAT_F(h), IDIO_F(h)
    natrows = pn.groupby('origin').agg({**{c: 'first' for c in nf}, f'ynat{h}': 'first'}).fillna({c: 0 for c in nf})
    for o in ORIG:
        cut = o - pd.DateOffset(months=h)
        ntr = natrows[(natrows.index <= cut) & natrows[f'ynat{h}'].notna()]
        itr = pn[(pn.origin <= cut) & pn[f'y{h}'].notna()]
        if len(ntr) < 12 or len(itr) < 200: continue
        use = [j for j, c in enumerate(nf) if not (c in ('n_upmove', 'n_upwk3') and not UP_ON[h])]
        nb0, nb_u, nlo_u, nhi_u = fit(ntr[[nf[j] for j in use]].values.astype(float), ntr[f'ynat{h}'].values)
        nb, nlo, nhi = np.zeros(len(nf)), np.zeros(len(nf)), np.zeros(len(nf))
        nb[use], nlo[use], nhi[use] = nb_u, nlo_u, nhi_u
        Xi = itr[idf].fillna(0).values.astype(float); yi = (itr[f'y{h}'] - itr[f'ynat{h}']).values
        ib0, ib, ilo, ihi = fit(Xi, yi)
        rec = dict(origin=o, h=h, n_train_months=len(ntr), n_train_rows=len(itr), nat_b0=nb0, idio_b0=ib0)
        for j, c in enumerate(nf): rec[f'nat_b_{j}'] = nb[j]; rec[f'nat_lo_{j}'] = nlo[j]; rec[f'nat_hi_{j}'] = nhi[j]
        for j, c in enumerate(idf): rec[f'idio_b_{j}'] = ib[j]; rec[f'idio_lo_{j}'] = ilo[j]; rec[f'idio_hi_{j}'] = ihi[j]
        COEF.append(rec)
COEF = pd.DataFrame(COEF)
for h in HZ:
    nf, idf = NAT_F(h), IDIO_F(h)
    C = COEF[COEF.h == h].set_index('origin')
    nat = np.full(len(pn), np.nan); idi = np.full(len(pn), np.nan)
    for i, r in enumerate(pn.itertuples()):
        if r.origin not in C.index: continue
        c = C.loc[r.origin]
        xn = np.nan_to_num(np.array([getattr(r, f) for f in nf], dtype=float))
        xi = np.nan_to_num(np.array([getattr(r, f) for f in idf], dtype=float))
        nat[i] = c.nat_b0 + sum(c[f'nat_b_{j}'] * min(max(xn[j], c[f'nat_lo_{j}']), c[f'nat_hi_{j}']) for j in range(len(nf)))
        idi[i] = c.idio_b0 + sum(c[f'idio_b_{j}'] * min(max(xi[j], c[f'idio_lo_{j}']), c[f'idio_hi_{j}']) for j in range(len(idf)))
    pn[f'nat_hat{h}'] = nat; pn[f'idio_hat{h}'] = idi
    pn[f'FM_{h}'] = pn.p0 * np.exp(nat + idi)
    pn[f'BL_{h}'] = np.sqrt(pn[f'FM_{h}'] * pn[f'V5_{h}'])

# ---------------- import-window regime model (M10) and regime switch (M11) ----------------
IW_THR, IW_MIN, RS_W = 0.0, 4, 0.5
GAPN = NATW - IPP0.reindex(MONTHS)                                     # national white ln price - ln import parity (before duty)
REG = ((FREE.reindex(MONTHS) == 1) & (GAPN > IW_THR)).astype(float)    # imports duty-free AND profitable
REGP = {}
for h in HZ:
    pn[f'lnchg{h}'] = np.log(pn[f'A{h}'] / pn.p0)
    nat = pn.groupby('origin').agg(y=(f'lnchg{h}', 'mean'), s=(f'seas{h}', 'mean'))
    nat['g'] = GAPN.reindex(nat.index); nat['reg'] = REG.reindex(nat.index); nat['yd'] = nat.y - nat.s
    pull = pd.Series(np.nan, index=MONTHS)
    for o in ORIG:
        if REG.get(o, 0) != 1: continue
        tr = nat[(nat.index <= o - pd.DateOffset(months=h)) & (nat.reg == 1) & nat.yd.notna() & nat.g.notna()]
        n = len(tr)
        if n < IW_MIN: continue
        Sx, Sy, Sxx, Sxy = tr.g.sum(), tr.yd.sum(), (tr.g ** 2).sum(), (tr.g * tr.yd).sum()
        b = (n * Sxy - Sx * Sy) / (n * Sxx - Sx ** 2); a = (Sy - b * Sx) / n
        pull[o] = a + b * GAPN[o]
    REGP[h] = pull
    pn[f'IW_{h}'] = pn.p0 * np.exp(pn[f'seas{h}'] + pn.origin.map(pull))
    pn[f'RS_{h}'] = np.where(pn[f'IW_{h}'].notna(), pn[f'BL_{h}'] ** (1 - RS_W) * pn[f'IW_{h}'] ** RS_W, pn[f'BL_{h}'])
pn['regime'] = pn.origin.map(REG)

MODELS = [('V5', 'OG inflation-adjusted (V5)'), ('V7', 'OG inflation-adjusted, compounded (V7)'), ('NV', 'Naive: carry last price'),
          ('FXO', 'FX pass-through, official rate'), ('FXP', 'FX pass-through, parallel rate'), ('FXA', 'Parallel FX recalibrated + premium alert'),
          ('IP', 'Import-parity anchor'), ('FM', 'Feature model (new)'), ('BL', 'Feature model + V5 blend (50/50)'),
          ('IW', 'Import-window model (regime)'), ('RS', 'Regime switch: blend + import window')]
pd.to_pickle(dict(panel=pn, COEF=COEF, SI=SI, logratio=logratio, real=real, tables=dict(GAPN=GAPN, REG=REG, REGP=REGP, GAP_FREE=GAP_FREE, UP_MOVE=UP_MOVE, UP_WK3=UP_WK3, UP_WK3_M=UP_WK3_M, UPP=UPP, NATW=NATW, REL=REL, DEV=DEV, D1=D1, IDIO_D1=IDIO_D1, SPRS=SPRS, RS_DEV=RS_DEV,
               IN_SLOPE=IN_SLOPE, LW=LW, L2=L2, GAP=GAP, MED24=MED24), nat=dict(NAT_SLOPE=NAT_SLOPE, NAT_LW=NAT_LW, NAT_L2=NAT_L2, FX3=FX3, PREM=PREM, ND1=ND1),
               MODELS=MODELS, SER=SER, TYPES=TYPES, SPRM=SPRM), f'v2/mirror{SUF}.pkl')

def metrics(pn, code, h, mask=None, obs_only=False):
    F = pn[f'{code}_{h}']; A = pn[f'A{h}']; ok = F.notna() & A.notna() & pn.p0.notna()
    if obs_only: ok &= pn[f'obs{h}'].eq(1)
    if mask is not None: ok &= mask
    ape = (F / A - 1).abs()[ok]; dirc = (((F > pn.p0) & (A > pn.p0)) | ((F < pn.p0) & (A < pn.p0)))[ok]
    ch = pn[f'chg{h}'][ok].abs()
    return dict(N=int(ok.sum()), MAPE=ape.mean() * 100, MAE=(F - A).abs()[ok].mean(), DA=dirc.mean() * 100,
                DA2=dirc[ch >= 0.02].mean() * 100, DA5=dirc[ch >= 0.05].mean() * 100, BAND=(ape <= pn.band[ok]).mean() * 100)
if __name__ == '__main__':
    orig_w = (pn.origin >= '2018-12-01') & (pn.origin <= '2024-03-01')
    for h in HZ:
        new_w = pn[f'tgt{h}'] >= '2024-10-01'
        print(f'\n=== h={h}')
        for code, name in MODELS:
            a = metrics(pn, code, h, orig_w); b = metrics(pn, code, h, new_w); c = metrics(pn, code, h)
            print(f'{name:42s} orig MAPE {a["MAPE"]:5.1f} DA {a["DA"]:4.0f} | new MAPE {b["MAPE"]:5.1f} DA {b["DA"]:4.0f} (n={b["N"]}) | all {c["MAPE"]:5.1f} n={c["N"]}')
