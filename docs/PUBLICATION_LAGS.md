# Publication dates and lags of every input (checked 2026-10-12)

What each input is, when it actually becomes public, the lag the models assume, and whether the
assumption holds. Replaces the "assumed" release dates of D-67 and D-70 with evidence where evidence
exists, and says plainly where it does not. Monthly design (D-75): the forecast for month o is made on the
15th of o+1. Weekly design (D-68): the forecast is made on a Wednesday.

| Input | Source | Evidence of release timing | Assumed in the models | Verdict |
|---|---|---|---|---|
| Maize price, monthly (and upstream price) | FEWS NET Data Warehouse API, `marketpricefacts` | The API's own `status_changed` timestamp per value. Month o is public 1 to 5 days after month end in most recent months; share of months public by the 15th of o+1: 33% (Sep 2015 to 2019), 78% (2020-22), 90% (2023 to Sep 2024), 71% (Oct 2024 to Aug 2026). Late months since 2023: Nov 2023 (27 days), May 2024 (35), Dec 2024 (15), and the 2025 pause: Jan 2025 published Dec 2025 (310 days), Apr 2025 in May 2026 (386), Sep 2025 in Oct 2026 (369) | Monthly: month o known by the 15th of o+1. Weekly OOS (D-70): the 15th | Holds in most months from 2020; does not hold for 2015-19 as recorded by the API, nor for Jan, Apr and Sep 2025. Caveat: the timestamp is when the monthly aggregate was published; the weekly prices behind 2012-2024 months may have been public earlier, and the API holds no weekly series to check |
| Maize price, May to Aug 2025 (recovered) | FEWS NET `value_one_year_ago` on the May to Aug 2026 rows | Public only with those 2026 rows (about 12 months later) | Used as known (owner decision, D-75) | Owner-accepted look-ahead |
| Maize price, weekly (2012 to Sep 2024) | FEWS NET weekly staple food price file | No release timestamps in the file or the API | Known on the Wednesday itself (D-68) | Not confirmable from data held |
| Diesel, state monthly (to Oct 2024) | NBS Automotive Gas Oil (Diesel) Price Watch | Report dates and press coverage: Mar 2024 report about 18 Apr 2024; Sep 2024 about 18 Oct 2024; Nov 2024 about 16 Jan 2025. About 18 days after month end, up to about 47 days | Weekly: the 20th of M+1. Monthly: through o-1 at the 15th of o+1 | Weekly rule close but broken in delayed months (Nov 2024). Monthly rule holds except a delay beyond about 45 days (Nov 2024 for the Dec 2024 origin, one day late) |
| Diesel, 2024-26 (FEWS NET) | FEWS NET API | Same timestamps as maize, including the 2025 delays | As the price | Same as maize |
| Rainfall, dekadal | WFP / UCSB CHIRPS v2 via HDX `nga-rainfall-subnational` | HDX: updated every 14 days; latest update 5 Oct 2026 with data to 30 Sep (about 5 days after the dekad). Recent dekads are CHIRPS preliminary and are revised to final about three weeks after the month | Weekly: 4 weeks back. Monthly: through o-1 | Holds; our history uses final (revised) values, a small vintage difference |
| NDVI, dekadal | NASA MODIS (MOD13C1/MYD13C1) via WFP, HDX `nga-ndvi-subnational` | HDX: updated every 30 days; latest update 5 Oct 2026 with data to the dekad starting 11 Sep (about 15 days after its end; up to about 45 between updates). Whittaker-filtered: recent values change as later data arrive | Weekly: 4 weeks back. Monthly: through o-1 | Monthly holds in the usual case; weekly 4 weeks can be short by up to about 2 weeks. The filter revises recent values, so history is a later vintage than a live forecast would see |
| Naira (official rate) | CBN `GetAllExchangeRates` | Daily rate dated by business day; time of day not recorded | Day before the forecast (weekly), the 14th of o+1 (monthly) | Holds (one-day margin) |
| Food CPI (blend seasonal index only) | NBS CPI and Inflation report | Around the 15th of the next month (May 2025 on 16 Jun, Jun 2025 on 16 Jul, Jul 2025 about 15 Aug); Jan 2025 delayed by the rebasing | Only complete calendar years before the origin's year | Holds: December CPI is out by mid-January, before any January forecast date (15 Feb) |
| Fill source: Kaduna/Kano other source | Weekly reports (`Quote_Level`) | Report date minus price week: median 2 days, maximum 6 | Month known by the 15th of o+1 | Holds |
| Fill source: AFEX | AFEX weekly panel | No publication dates in the files | Month known by the 15th of o+1 | Not confirmable; needs AFEX's release record |
| Fill source: NAERLS | NAERLS wholesale maize | No publication dates in the file | Month known by the 15th of o+1 | Not confirmable; needs NAERLS's release record |

## Sources
- FEWS NET API, full fields: `https://fdw.fews.net/api/marketpricefacts.json?country_code=NG&fields=full` (pulled 2026-10-12; `created`, `status_changed`, `collection_status_changed`; `modified` is 2026-10-12 on every row and carries no information).
- HDX CKAN API `package_show` for `nga-rainfall-subnational` and `nga-ndvi-subnational` (`data_update_frequency`, `last_modified`, `dataset_date`, methodology).
- NBS e-library and microdata catalogue for the Diesel Price Watch; release days from press coverage (Nairametrics 18 Apr 2024; news of 21 Oct 2024; Pulse 16 Jan 2025).
- NBS CPI releases from press coverage (June, July, August 2025).
