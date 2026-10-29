"""Monthly refit of the feature model weights for Maize_Forecast_v4_White.xlsx.

Usage:   python refit_feature_model_v4.py Maize_Forecast_v4_White.xlsx
Needs:   v2_models.py in the same folder; pandas, numpy, openpyxl, lxml, scikit-learn.

What it does: reads the input tabs (In_Price, In_Price_Yellow, In_Price_Observed, In_Macro, In_Weekly_Prices, In_Sorghum, In_Zone_Bands),
recomputes every signal exactly as the workbook does, re-learns the weights for every forecast month and horizon using only past data
(white AND yellow maize, 26 series), and writes them into the M3_FM_Weights_1m / 3m / 6m tabs. Only the numbers in those three tabs are
changed - charts, pictures, formats and every other tab are left untouched. Open the workbook in Excel afterwards: it recalculates on opening.
"""
import sys, os, re, shutil, zipfile, warnings, tempfile
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, openpyxl
from lxml import etree

path = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
R0, NM = 5, 140                                           # month rows 5..144 (Jan-2015 ...)
wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
def grid(tab, ncols):
    rows = list(wb[tab].iter_rows(min_row=4, values_only=True))
    hdr = list(rows[0][1:1 + ncols]); data = [r for r in rows[1:] if r[0] is not None]
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in data]).to_period('M').to_timestamp()
    return pd.DataFrame([[v if isinstance(v, (int, float)) else np.nan for v in r[1:1 + ncols]] for r in data], index=idx, columns=hdr)
white = grid('In_Price', 15); yellow = grid('In_Price_Yellow', 11)
MK, YMK = list(white.columns), list(yellow.columns)
MONTHS = white.index
SER = [('White', m) for m in MK] + [('Yellow', m) for m in YMK]
obs = grid('In_Price_Observed', 15)
flags = pd.concat([obs.reindex(columns=MK), yellow.notna().astype(float).where(yellow.notna())], axis=1); flags.columns = pd.MultiIndex.from_tuples(SER)
mrows = list(wb['In_Macro'].iter_rows(min_row=5, values_only=True))
cols = ['headline_mom', 'food_mom', 'cpi', 'fx_parallel', 'fx_official', 'world_maize', 'diesel', 'petrol', 'logistics', 'duty']
mac = pd.DataFrame([[v if isinstance(v, (int, float)) else np.nan for v in r[1:11]] for r in mrows if r[0] is not None],
                   index=pd.DatetimeIndex([pd.Timestamp(r[0]) for r in mrows if r[0] is not None]).to_period('M').to_timestamp(), columns=cols)
wk = [r for r in wb['In_Weekly_Prices'].iter_rows(min_row=5, values_only=True) if r[0] is not None]
weekly = pd.DataFrame({'type': [r[1] for r in wk], 'market': [r[2] for r in wk], 'end': pd.to_datetime([r[3] for r in wk]), 'price_kg': [float(r[4]) for r in wk]})
sorg = grid('In_Sorghum', 15)
zb = wb['In_Zone_Bands']
bands = {r[0]: r[1] for r in zb.iter_rows(min_row=5, max_row=9, max_col=2, values_only=True) if r[0]}
ZONE = {r[0]: r[1] for r in zb.iter_rows(min_row=5, max_row=40, min_col=4, max_col=5, values_only=True) if r[0]}
D_OVERRIDE = dict(SER=SER, MK=MK, TYPES=['White', 'Yellow'], white=white, yellow=yellow, MONTHS=MONTHS, macro=mac, weekly=weekly,
                  sorghum=sorg.reindex(MONTHS), bands=bands, ZONE=ZONE, flags=flags)
src = open(os.path.join(here, 'v2_models.py')).read().split("pd.to_pickle(dict(panel=pn")[0]
src = src.replace("sys.path.insert(0, '/home/claude'); os.chdir('/home/claude'); ", '')
g = {'D_OVERRIDE': D_OVERRIDE, '__name__': 'refit'}
exec(src, g)
COEF = g['COEF']
wb.close()

# column layout of each weights tab: B training months, C training rows, D national intercept, E..Y national (weight, low, high) x 7,
# Z market intercept, AA..AU market (weight, low, high) x 7 (out of line, market-only jump, white/yellow, seasonal vs national, own last week, own last 2 weeks, maize/sorghum)
FIELDS = ['n_train_months', 'n_train_rows', 'nat_b0'] + [f'nat_{k}_{j}' for j in range(7) for k in ('b', 'lo', 'hi')] + ['idio_b0'] + [f'idio_{k}_{j}' for j in range(7) for k in ('b', 'lo', 'hi')]
def colnum(ref): return openpyxl.utils.column_index_from_string(re.match(r'[A-Z]+', ref).group())
NS_ = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
tmp = tempfile.mkdtemp(); zin = zipfile.ZipFile(path)
wbx = etree.fromstring(zin.read('xl/workbook.xml')); rels = etree.fromstring(zin.read('xl/_rels/workbook.xml.rels'))
rid = {r.get('Id'): r.get('Target') for r in rels}
sheetfile = {s.get('name'): 'xl/' + rid[s.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')].lstrip('/').replace('xl/', '') for s in wbx.find('m:sheets', NS_)}
new = {}
for h in (1, 3, 6):
    C = COEF[COEF.h == h].set_index('origin')
    fn = sheetfile[f'M3_FM_Weights_{h}m']; root = etree.fromstring(zin.read(fn))
    for row in root.find('m:sheetData', NS_):
        r = int(row.get('r'))
        if r < R0: continue
        t = MONTHS[r - R0] if r - R0 < len(MONTHS) else None
        for c in row:
            k = colnum(c.get('r')) - 2
            if k < 0 or k >= len(FIELDS): continue
            for v in c.findall('m:v', NS_): c.remove(v)
            c.attrib.pop('t', None)
            if t is not None and t in C.index:
                v = etree.SubElement(c, '{%s}v' % NS_['m']); v.text = repr(float(C.at[t, FIELDS[k]]))
    new[fn] = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
out = path + '.tmp'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = new.get(item.filename, zin.read(item.filename))
        zout.writestr(item, data)
zin.close(); shutil.move(out, path)
print(f'Wrote weights for {COEF.origin.nunique()} forecast months ({COEF.origin.min():%b-%Y} to {COEF.origin.max():%b-%Y}). Open the workbook in Excel to recalculate.')
