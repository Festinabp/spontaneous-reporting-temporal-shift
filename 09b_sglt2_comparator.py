"""
09b_sglt2_comparator.py

Second active comparator: sodium-glucose cotransporter-2 (SGLT2) inhibitors, which treat type 2 diabetes and were
marketed and reported over the same period as the GLP-1 RAs. Run after 09_brand_adjusted_controls.py (uses its
product-name cache, work/brand_*.pkl).

  Exposed:   reports with the GLP-1 RA (class or agent) and no SGLT2 inhibitor
  Reference: reports with an SGLT2 inhibitor and no GLP-1 RA
  (DPP-4 inhibitor co-exposure is allowed in both groups.)

For pancreatitis and gallbladder/biliary events (positive control): crude RORs and
logistic-regression RORs adjusted for year only, year and reporter type, and all covariates (year, age group, sex,
reporter type, region), in all reports, in health-professional reports, and with GLP-1 RA exposure restricted to
diabetes products. Also the yearly proportion of reports including pancreatitis.

Outputs (RESULTS_DIR): SGLT2_comparator.csv, SGLT2_by_year.csv
Runtime: about 15-40 minutes for the first run (one pass over the DRUG tables), then seconds.
"""
import re
import time
import zipfile
import numpy as np
import pandas as pd
import config as C

T0 = time.time()
HCP = {"MD", "PH", "HP", "OT"}


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def find_member(zf, table):
    pat = re.compile(rf"(^|/){table}\d{{2}}q[1-4](_new)?\.txt$", re.I)
    hits = [n for n in zf.namelist() if pat.search(n)]
    if len(hits) != 1:
        raise RuntimeError(f"{zf.filename}: expected one {table} file, found {hits}")
    return hits[0]


SGLT2_PAT = re.compile(r"\b(?:" + "|".join(re.escape(n) for v in C.SGLT2_MAP.values() for n in v) + r")\b")

# ----------------------------------------------------------------------------- SGLT2 inhibitor exposure (cached)
r = pd.read_pickle(C.WORK_DIR / "reports.pkl")
zips = {}
for z in C.BASE_DIR.glob("faers_ascii_*.zip"):
    m = re.search(r"(\d{4})q([1-4])", z.name.lower())
    if m:
        zips[f"{m.group(1)}q{m.group(2)}"] = z
ids = []
for q in sorted(r["quarter"].unique()):
    cache = C.WORK_DIR / f"sglt2_{q}.pkl"
    if cache.exists():
        ids.append(pd.read_pickle(cache))
        continue
    keep = set(r.loc[r["quarter"] == q, "primaryid"].astype("int64"))
    with zipfile.ZipFile(zips[q]) as zf, zf.open(find_member(zf, "DRUG")) as fh:
        d = pd.read_csv(fh, sep="$", dtype=str, encoding="latin-1", quoting=3,
                        usecols=lambda c: c.strip().lower() in {"primaryid", "drugname", "prod_ai"},
                        on_bad_lines="skip", keep_default_na=False, na_values=[""], engine="c")
    d.columns = [c.strip().lower() for c in d.columns]
    d["primaryid"] = pd.to_numeric(d["primaryid"], errors="coerce")
    d = d[d["primaryid"].isin(keep)]
    text = (d["drugname"].fillna("") + " | " + d["prod_ai"].fillna("")).str.upper()
    u = pd.Series(text.unique())
    hit = set(u[u.str.contains(SGLT2_PAT)])          # match each distinct name once
    s = pd.Series(sorted(set(d.loc[text.isin(hit), "primaryid"].astype("int64"))), dtype="int64")
    s.to_pickle(cache)
    ids.append(s)
    log(f"{q}: SGLT2 inhibitor reports {len(s):,}")
sglt2_ids = set(pd.concat(ids).tolist()) if ids else set()

# ----------------------------------------------------------------------------- analysis dataset
brand = pd.concat([pd.read_pickle(p) for p in sorted(C.WORK_DIR.glob("brand_*.pkl"))], ignore_index=True)
r = r.merge(brand, on="primaryid", how="left")
for c in [c for c in r.columns if c.startswith(("diab_", "wt_"))]:
    r[c] = r[c].fillna(False).astype(bool)
r["sglt2"] = r["primaryid"].isin(sglt2_ids)
r["sglt2_only"] = r["sglt2"] & ~r["glp1_any"]
r["hcp"] = r["reporter_type"].isin(HCP)
r["diab_any"] = r[[f"diab_{a}" for a in C.GLP1_AGENTS]].any(axis=1)
r["wt_any"] = r[[c for c in r.columns if c.startswith("wt_")]].any(axis=1)
r["age_grp"] = pd.cut(r["age_years"], [-1, 39.999, 49.999, 59.999, 69.999, 200], labels=["<40", "40-49", "50-59", "60-69", "70+"]).astype(str).replace("nan", "missing")
r["sex_grp"] = r["sex"].fillna("missing")
r["rep_grp"] = np.select([r["hcp"], r["reporter_type"].eq("CN")], ["hcp", "consumer"], "other/unknown")
r["region"] = np.select([r["country"].eq("US"), r["country"].notna()], ["US", "non-US"], "missing")
r["year_c"] = r["report_year"].astype(int).astype(str)
LEVELS = {"year_c": [str(y) for y in range(C.YEAR_START, C.YEAR_END + 1)],
          "age_grp": ["<40", "40-49", "50-59", "60-69", "70+", "missing"],
          "sex_grp": ["F", "M", "missing"], "rep_grp": ["hcp", "consumer", "other/unknown"],
          "region": ["US", "non-US", "missing"]}
for c, lev in LEVELS.items():
    r[c] = pd.Categorical(r[c], categories=lev)
COVS = ["year_c", "age_grp", "sex_grp", "rep_grp", "region"]
EXPOSURES = C.GLP1_AGENTS + ["GLP-1 RA class"]
log(f"SGLT2 inhibitor reports without a GLP-1 RA: {int(r['sglt2_only'].sum()):,}")


def crude(exp, ev, univ):
    e, y = exp[univ].values.astype(bool), ev[univ].values.astype(bool)
    a, b, c, d = int((e & y).sum()), int((e & ~y).sum()), int((~e & y).sum()), int((~e & ~y).sum())
    if min(a, b, c, d) == 0:
        return dict(a=a, b=b, c=c, d=d, ROR=np.nan, lower=np.nan, upper=np.nan)
    est = a * d / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return dict(a=a, b=b, c=c, d=d, ROR=est, lower=est * np.exp(-1.96 * se), upper=est * np.exp(1.96 * se))


def adjusted(exp, ev, univ, covs):
    """Logistic regression (IRLS) of the event on exposure + covariates; adjusted ROR with Wald CI."""
    X = pd.get_dummies(r.loc[univ, covs], drop_first=True, dtype=float)
    X = X.loc[:, X.sum(axis=0) > 0]
    X.insert(0, "exposure", exp[univ].astype(float).values)
    X.insert(0, "const", 1.0)
    X = X.values
    y = ev[univ].astype(float).values
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ beta)))
        w = np.clip(p * (1 - p), 1e-10, None)
        H = X.T @ (X * w[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    se = np.sqrt(np.linalg.inv(H)[1, 1])
    return np.exp(beta[1]), np.exp(beta[1] - 1.96 * se), np.exp(beta[1] + 1.96 * se)


def exp_col(agent, diab=False):
    if diab:
        e = r["diab_any"] if agent == "GLP-1 RA class" else r[f"diab_{agent}"]
        return e & ~r["wt_any"]
    return r["glp1_any"] if agent == "GLP-1 RA class" else r[f"glp1_{agent}"]


MODELS = [("Year only", ["year_c"]), ("Year and reporter type", ["year_c", "rep_grp"]), ("Fully adjusted", COVS)]
SUBSETS = [("vs SGLT2i (all)", False, None), ("vs SGLT2i, health-professional reports", False, r["hcp"]),
           ("vs SGLT2i, diabetes products", True, None)]
rows = []
for event in ["pancreatitis", "gallbladder_biliary"]:
    for label, diab, sub in SUBSETS:
        for agent in EXPOSURES:
            exp = exp_col(agent, diab) & ~r["sglt2"]
            univ = exp | r["sglt2_only"]
            if sub is not None:
                univ = univ & sub
            res = crude(exp, r[event], univ)
            row = {"analysis": label, "event": event, "exposure": agent, **res}
            if res["a"] >= 10 and res["c"] >= 10:
                for name, covs in MODELS:
                    if sub is not None and covs == ["year_c", "rep_grp"]:
                        continue
                    est, lo, hi = adjusted(exp, r[event], univ, covs)
                    key = {"Year only": "year", "Year and reporter type": "yr_rep", "Fully adjusted": "adj"}[name]
                    row.update({f"{key}_ROR": est, f"{key}_lower": lo, f"{key}_upper": hi})
            rows.append(row)
    log(f"{event} done")
out = pd.DataFrame(rows)
out.to_csv(C.RESULTS_DIR / "SGLT2_comparator.csv", index=False)

g = r[r["glp1_any"] & ~r["sglt2"]]
s = r[r["sglt2_only"]]
by_year = pd.DataFrame({
    "GLP-1 RA reports (n)": g.groupby("report_year").size(),
    "GLP-1 RA with pancreatitis (%)": g.groupby("report_year")["pancreatitis"].mean() * 100,
    "SGLT2i reports (n)": s.groupby("report_year").size(),
    "SGLT2i with pancreatitis (%)": s.groupby("report_year")["pancreatitis"].mean() * 100,
    "SGLT2i reports from lawyers (%)": s.groupby("report_year")["reporter_type"].apply(lambda x: (x == "LW").mean() * 100),
})
by_year.round(2).to_csv(C.RESULTS_DIR / "SGLT2_by_year.csv")

pd.set_option("display.width", 220)
show = out[out.exposure == "GLP-1 RA class"][["analysis", "event", "a", "c", "ROR", "year_ROR", "yr_rep_ROR", "adj_ROR", "adj_lower", "adj_upper"]]
print(show.round(2).to_string(index=False))
print(by_year.round(2).to_string())
log("Done. Next: run 10_ml_extensions.py (or 11_figure_comparator.py)")
