"""
09_brand_adjusted_controls.py

Three additional robustness analyses (run after 07 and 08):

  1. Product-based indication. Indication is missing for most reports, so the marketed product is used as a
     proxy: GLP-1 RA products approved only for type 2 diabetes (Ozempic, Rybelsus, Victoza, Mounjaro,
     Trulicity, Byetta, Bydureon, Adlyxin/Lyxumia, Tanzeum/Eperzan, Soliqua/Suliqua, Xultophy, and all
     exenatide, lixisenatide, albiglutide and dulaglutide products) versus products approved for weight
     management (Wegovy, Saxenda, Zepbound). Reports naming only the generic semaglutide, liraglutide or
     tirzepatide are left unclassified.
       a. diabetes products vs DPP-4 inhibitors
       b. weight products vs diabetes products of the same agent
  2. Adjusted active-comparator RORs: logistic regression of pancreatitis on exposure (GLP-1 RA vs DPP-4
     inhibitor) adjusted for year of initial receipt, age group, sex, reporter type and reporting region.
     Also: comparisons restricted to reports with a recorded diabetes indication (INDI table), and a stepwise
     table (crude; year only; year and reporter type; fully adjusted) showing which covariates move the estimate.
  3. Positive control: the same comparisons for gallbladder/biliary events (a recognised reaction to GLP-1 RAs
     in randomized trials).

Outputs (RESULTS_DIR): Brand_indication.csv, Adjusted_comparator.csv, Adjustment_steps.csv, Positive_control.csv,
  figures/Fig_positive_control.png
Runtime: about 10-30 minutes for the first run (one more pass over the DRUG tables), then seconds.
"""
import re
import time
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

T0 = time.time()
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
HCP = {"MD", "PH", "HP", "OT"}


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def pattern(names):
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b")


# ----------------------------------------------------------------------------- product groups
DIAB_PRODUCTS = {  # products approved only for type 2 diabetes (generic names of diabetes-only agents included)
    "SEMAGLUTIDE": ["OZEMPIC", "RYBELSUS"],
    "LIRAGLUTIDE": ["VICTOZA", "XULTOPHY"],
    "TIRZEPATIDE": ["MOUNJARO"],
    "DULAGLUTIDE": ["DULAGLUTIDE", "TRULICITY"],
    "EXENATIDE": ["EXENATIDE", "BYETTA", "BYDUREON"],
    "LIXISENATIDE": ["LIXISENATIDE", "ADLYXIN", "LYXUMIA", "SOLIQUA", "SULIQUA"],
    "ALBIGLUTIDE": ["ALBIGLUTIDE", "TANZEUM", "EPERZAN"],
}
WEIGHT_PRODUCTS = {"SEMAGLUTIDE": ["WEGOVY"], "LIRAGLUTIDE": ["SAXENDA"], "TIRZEPATIDE": ["ZEPBOUND"]}
DIAB_PAT = {a: pattern(v) for a, v in DIAB_PRODUCTS.items()}
WT_PAT = {a: pattern(v) for a, v in WEIGHT_PRODUCTS.items()}


def find_member(zf, table):
    pat = re.compile(rf"(^|/){table}\d{{2}}q[1-4](_new)?\.txt$", re.I)
    hits = [n for n in zf.namelist() if pat.search(n)]
    if len(hits) != 1:
        raise RuntimeError(f"{zf.filename}: expected one {table} file, found {hits}")
    return hits[0]


def read_drug(zf):
    with zf.open(find_member(zf, "DRUG")) as fh:
        df = pd.read_csv(fh, sep="$", dtype=str, encoding="latin-1", quoting=3,
                         usecols=lambda c: c.strip().lower() in {"primaryid", "drugname", "prod_ai"},
                         on_bad_lines="skip", keep_default_na=False, na_values=[""], engine="c")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


# ----------------------------------------------------------------------------- extraction (cached)
r = pd.read_pickle(C.WORK_DIR / "reports.pkl")
zips = {}
for z in C.BASE_DIR.glob("faers_ascii_*.zip"):
    m = re.search(r"(\d{4})q([1-4])", z.name.lower())
    if m:
        zips[f"{m.group(1)}q{m.group(2)}"] = z
parts = []
for q in sorted(r["quarter"].unique()):
    cache = C.WORK_DIR / f"brand_{q}.pkl"
    if cache.exists():
        parts.append(pd.read_pickle(cache))
        continue
    target = set(r.loc[(r["quarter"] == q) & r["glp1_any"], "primaryid"].astype("int64"))
    with zipfile.ZipFile(zips[q]) as zf:
        d = read_drug(zf)
    d["primaryid"] = pd.to_numeric(d["primaryid"], errors="coerce")
    d = d[d["primaryid"].isin(target)]
    text = (d["drugname"].fillna("") + " | " + d["prod_ai"].fillna("")).str.upper()
    out = pd.DataFrame(index=pd.Index(sorted(target), name="primaryid"))
    name_only = d["drugname"].fillna("").str.upper()
    for a in C.GLP1_AGENTS:
        # diabetes-only agents: any product of the agent; others: brand name in the drug-name field
        src = text if a in ["DULAGLUTIDE", "EXENATIDE", "LIXISENATIDE", "ALBIGLUTIDE"] else name_only
        out[f"diab_{a}"] = d.loc[src.str.contains(DIAB_PAT[a])].groupby("primaryid").size().gt(0)
        if a in WT_PAT:
            out[f"wt_{a}"] = d.loc[name_only.str.contains(WT_PAT[a])].groupby("primaryid").size().gt(0)
    out = out.fillna(False).astype(bool).reset_index()
    out.to_pickle(cache)
    parts.append(out)
    log(f"{q}: product names extracted")
brand = pd.concat(parts, ignore_index=True)

# ----------------------------------------------------------------------------- analysis dataset
extra = pd.read_pickle(C.WORK_DIR / "extra_reports.pkl")
r = r.merge(brand, on="primaryid", how="left").merge(extra, on="primaryid", how="left")
for c in [c for c in r.columns if c.startswith(("diab_", "wt_")) or "_ind_" in c]:
    r[c] = r[c].fillna(False).astype(bool)
r["hcp"] = r["reporter_type"].isin(HCP)
r["dpp4_only"] = r["dpp4"] & ~r["glp1_any"]
r["diab_any"] = r[[f"diab_{a}" for a in C.GLP1_AGENTS]].any(axis=1)
r["wt_any"] = r[[f"wt_{a}" for a in WEIGHT_PRODUCTS]].any(axis=1)
# covariates for adjustment
r["age_grp"] = pd.cut(r["age_years"], [-1, 39.999, 49.999, 59.999, 69.999, 200], labels=["<40", "40-49", "50-59", "60-69", "70+"]).astype(str).replace("nan", "missing")
r["sex_grp"] = r["sex"].fillna("missing")
r["rep_grp"] = np.select([r["hcp"], r["reporter_type"].eq("CN")], ["hcp", "consumer"], "other/unknown")
r["region"] = np.select([r["country"].eq("US"), r["country"].notna()], ["US", "non-US"], "missing")
r["year_c"] = r["report_year"].astype(int).astype(str)
# reference categories (first level): year 2015, age <40, female, health professional, United States
LEVELS = {"year_c": [str(y) for y in range(C.YEAR_START, C.YEAR_END + 1)],
          "age_grp": ["<40", "40-49", "50-59", "60-69", "70+", "missing"],
          "sex_grp": ["F", "M", "missing"], "rep_grp": ["hcp", "consumer", "other/unknown"],
          "region": ["US", "non-US", "missing"]}
for c, lev in LEVELS.items():
    r[c] = pd.Categorical(r[c], categories=lev)
COVS = ["year_c", "age_grp", "sex_grp", "rep_grp", "region"]
EXPOSURES = C.GLP1_AGENTS + ["GLP-1 RA class"]
log(f"Analysis dataset ready: {len(r):,} reports")


def crude(exp, ev, univ):
    e, y = exp[univ].values.astype(bool), ev[univ].values.astype(bool)
    a, b, c, d = int((e & y).sum()), int((e & ~y).sum()), int((~e & y).sum()), int((~e & ~y).sum())
    if min(a, b, c, d) == 0:
        return dict(a=a, b=b, c=c, d=d, ROR=np.nan, lower=np.nan, upper=np.nan)
    est = a * d / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return dict(a=a, b=b, c=c, d=d, ROR=est, lower=est * np.exp(-1.96 * se), upper=est * np.exp(1.96 * se))


def adjusted(exp, ev, univ, covs=COVS):
    """Logistic regression (IRLS) of the event on exposure + covariates; returns adjusted ROR with Wald CI."""
    d = r.loc[univ, covs].copy()
    X = pd.get_dummies(d, drop_first=True, dtype=float)
    X = X.loc[:, X.sum(axis=0) > 0]   # drop indicator columns of levels absent from this subset
    X.insert(0, "exposure", exp[univ].astype(float).values)
    X.insert(0, "const", 1.0)
    X = X.values
    y = ev[univ].astype(float).values
    beta = np.zeros(X.shape[1])
    for _ in range(50):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-10, None)
        H = X.T @ (X * w[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    cov = np.linalg.inv(H)
    b, se = beta[1], np.sqrt(cov[1, 1])
    return dict(aROR=np.exp(b), a_lower=np.exp(b - 1.96 * se), a_upper=np.exp(b + 1.96 * se))


def exp_col(agent, prefix="glp1_"):
    if agent == "GLP-1 RA class":
        return r["glp1_any"] if prefix == "glp1_" else r[f"{prefix}any"]
    col = f"{prefix}{agent}"
    return r[col] if col in r else pd.Series(False, index=r.index)


def comparator_rows(event, label, exp_fn, ref_mask, extra_univ=None, adjust=True):
    rows = []
    for agent in EXPOSURES:
        exp = exp_fn(agent) & ~r["dpp4"]
        univ = exp | ref_mask
        if extra_univ is not None:
            univ = univ & extra_univ
        res = crude(exp, r[event], univ)
        if adjust and res["a"] >= 10 and res["c"] >= 10:
            res.update(adjusted(exp, r[event], univ))
        rows.append({"analysis": label, "event": event, "exposure": agent, **res})
    return rows


# ----------------------------------------------------------------------------- 1. product-based indication
rows = []
rows += comparator_rows("pancreatitis", "Diabetes products vs DPP-4i", lambda a: exp_col(a, "diab_") & ~r["wt_any"], r["dpp4_only"])
for a in ["SEMAGLUTIDE", "LIRAGLUTIDE", "TIRZEPATIDE", "GLP-1 RA class"]:
    wt = (r["wt_any"] & ~r["diab_any"]) if a == "GLP-1 RA class" else (r[f"wt_{a}"] & ~r[f"diab_{a}"])
    db = (r["diab_any"] & ~r["wt_any"]) if a == "GLP-1 RA class" else (r[f"diab_{a}"] & ~r[f"wt_{a}"])
    res = crude(wt, r["pancreatitis"], wt | db)
    if res["a"] >= 10 and res["c"] >= 10:
        res.update(adjusted(wt, r["pancreatitis"], wt | db))
    rows.append({"analysis": "Weight vs diabetes products", "event": "pancreatitis", "exposure": a, **res})
brand_res = pd.DataFrame(rows)
brand_res.to_csv(C.RESULTS_DIR / "Brand_indication.csv", index=False)
cover = pd.Series({
    "GLP-1 RA reports": int(r["glp1_any"].sum()),
    "diabetes product only": int((r["diab_any"] & ~r["wt_any"]).sum()),
    "weight product only": int((r["wt_any"] & ~r["diab_any"]).sum()),
    "both": int((r["wt_any"] & r["diab_any"]).sum()),
    "unclassified (generic name only)": int((r["glp1_any"] & ~r["wt_any"] & ~r["diab_any"]).sum()),
})
cover.to_csv(C.RESULTS_DIR / "Brand_indication_coverage.csv", header=["reports"])
log("Product-based indication done")

# ----------------------------------------------------------------------------- 2. adjusted comparator
rows = []
rows += comparator_rows("pancreatitis", "vs DPP-4i (all)", lambda a: exp_col(a), r["dpp4_only"])
rows += comparator_rows("pancreatitis", "vs DPP-4i, health-professional reports", lambda a: exp_col(a), r["dpp4_only"], r["hcp"])
rows += comparator_rows("pancreatitis", "vs DPP-4i, diabetes products", lambda a: exp_col(a, "diab_") & ~r["wt_any"], r["dpp4_only"])
rows += comparator_rows("pancreatitis", "vs DPP-4i, diabetes products, health-professional reports", lambda a: exp_col(a, "diab_") & ~r["wt_any"], r["dpp4_only"], r["hcp"])
# recorded indication (INDI): diabetes recorded for the agent's own drug rows (any GLP-1 RA for the class)
ind_exp = lambda a: (r["glp1_any"] & r["glp1_ind_diabetes"]) if a == "GLP-1 RA class" else (r[f"glp1_{a}"] & r[f"{a}_ind_diabetes"])
rows += comparator_rows("pancreatitis", "vs DPP-4i, recorded diabetes indication", ind_exp, r["dpp4_only"] & r["dpp4_ind_diabetes"])
adj = pd.DataFrame(rows)
adj.to_csv(C.RESULTS_DIR / "Adjusted_comparator.csv", index=False)
log("Adjusted comparator done")

# stepwise adjustment (vs DPP-4i, all reports)
STEPS_ADJ = [("Year only", ["year_c"]), ("Reporter type only", ["rep_grp"]), ("Year and reporter type", ["year_c", "rep_grp"]),
             ("Fully adjusted", COVS)]
rows = []
for event in ["pancreatitis", "gallbladder_biliary"]:
    for agent in (EXPOSURES if event == "pancreatitis" else ["GLP-1 RA class"]):
        exp = exp_col(agent) & ~r["dpp4"]
        univ = exp | r["dpp4_only"]
        res = crude(exp, r[event], univ)
        rows.append({"event": event, "exposure": agent, "model": "Crude", "ROR": res["ROR"], "lower": res["lower"], "upper": res["upper"]})
        if res["a"] >= 10 and res["c"] >= 10:
            for lab, covs in STEPS_ADJ:
                a_ = adjusted(exp, r[event], univ, covs)
                rows.append({"event": event, "exposure": agent, "model": lab, "ROR": a_["aROR"], "lower": a_["a_lower"], "upper": a_["a_upper"]})
pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "Adjustment_steps.csv", index=False)
log("Stepwise adjustment done")

# ----------------------------------------------------------------------------- 3. control outcomes
EV2 = "gallbladder_biliary"   # positive control
ALL = pd.Series(True, index=r.index)
rows = []
for event in ["pancreatitis", EV2]:
    for agent in EXPOSURES:
        rows.append({"analysis": "1 Main analysis (all reports)", "event": event, "exposure": agent, **crude(exp_col(agent), r[event], ALL)})
        sus = r["glp1_any_suspect"] if agent == "GLP-1 RA class" else r[f"glp1_{agent}_suspect"]
        rows.append({"analysis": "2 Suspect drug only", "event": event, "exposure": agent, **crude(sus, r[event], ALL)})
        rows.append({"analysis": "3 Health-professional reports only", "event": event, "exposure": agent, **crude(exp_col(agent), r[event], r["hcp"])})
    rows += comparator_rows(event, "5 vs DPP-4 inhibitors", lambda a: exp_col(a), r["dpp4_only"])
    rows += comparator_rows(event, "6 vs DPP-4 inhibitors, diabetes products", lambda a: exp_col(a, "diab_") & ~r["wt_any"], r["dpp4_only"])
    rows += comparator_rows(event, "7 vs DPP-4 inhibitors, diabetes products, health professionals", lambda a: exp_col(a, "diab_") & ~r["wt_any"], r["dpp4_only"], r["hcp"])
pc = pd.DataFrame(rows)
pc.to_csv(C.RESULTS_DIR / "Positive_control.csv", index=False)

# Figure: class-level crude and adjusted RORs for pancreatitis vs gallbladder across comparisons
cls = pc[pc.exposure == "GLP-1 RA class"].copy()
steps = list(dict.fromkeys(cls["analysis"]))
fig, ax = plt.subplots(figsize=(7.2, 4))
for j, (event, col, lab) in enumerate([("pancreatitis", "#d62728", "Pancreatitis"), (EV2, "#1f77b4", "Gallbladder/biliary (positive control)")]):
    d = cls[cls.event == event].set_index("analysis").reindex(steps)
    y = np.arange(len(steps))[::-1] + (0.15 if j == 0 else -0.15)
    ax.errorbar(d["ROR"], y, xerr=[d["ROR"] - d["lower"], d["upper"] - d["ROR"]], fmt="o", color=col, capsize=2, ms=4, label=lab + " (crude)")
    if "aROR" in d:
        ok = d["aROR"].notna().values
        ax.errorbar(d["aROR"][ok], y[ok] - 0.08 * (1 if j == 0 else -1), xerr=[(d["aROR"] - d["a_lower"])[ok], (d["a_upper"] - d["aROR"])[ok]],
                    fmt="D", mfc="white", color=col, capsize=2, ms=4, label=lab + " (adjusted)")
ax.axvline(1, ls="--", color="grey", lw=0.8)
ax.set_xscale("log")
ax.set_yticks(np.arange(len(steps))[::-1])
ax.set_yticklabels([s[2:] for s in steps], fontsize=8)
ax.set_xlabel("GLP-1 RA class ROR (95% CI, log scale)")
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(C.FIG_DIR / "Fig_positive_control.png", dpi=C.FIG_DPI)
plt.close(fig)

pd.set_option("display.width", 220)
show = lambda df: df[df.exposure == "GLP-1 RA class"][[c for c in ["analysis", "event", "a", "c", "ROR", "lower", "upper", "aROR", "a_lower", "a_upper"] if c in df]].round(2).to_string(index=False)
print(cover.to_string())
print(show(brand_res))
print(show(adj))
print(show(pc))
print(pd.read_csv(C.RESULTS_DIR / "Adjustment_steps.csv").query("exposure == 'GLP-1 RA class'").round(2).to_string(index=False))
log("Done. All results are in the results folder.")
