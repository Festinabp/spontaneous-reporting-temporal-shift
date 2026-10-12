"""
09d_year_only_subgroups.py - year-only adjusted RORs for the DPP-4 inhibitor subgroup comparisons.

Year of initial FDA receipt is the primary adjustment (fewest modelling assumptions). Step 09 reports
year-only estimates for all DPP-4 inhibitor reports (Adjustment_steps.csv) and step 09b for all SGLT2
comparisons (SGLT2_comparator.csv, year_ROR); this script adds year-only estimates for the DPP-4
inhibitor subgroups (health-professional reports, diabetes products, both combined, recorded diabetes
indication). It uses the same exposure definitions, covariate coding and logistic regression as step 09,
and re-estimates the fully adjusted model as a check against Adjusted_comparator.csv.

Input: work/glp1_reports.csv.gz (step 06), work/brand_*.pkl (step 09), work/extra_reports.pkl (step 07)
Output: results/Year_only_subgroups_dpp4.csv
Run after 06 and 09, and before 11.
"""
import numpy as np
import pandas as pd
import config as C

HCP = {"MD", "PH", "HP", "OT"}
WEIGHT_PRODUCTS = ["SEMAGLUTIDE", "LIRAGLUTIDE", "TIRZEPATIDE"]
EXPOSURES = C.GLP1_AGENTS + ["GLP-1 RA class"]

r = pd.read_csv(C.WORK_DIR / "glp1_reports.csv.gz")
brand = pd.concat([pd.read_pickle(p) for p in sorted(C.WORK_DIR.glob("brand_*.pkl"))], ignore_index=True)
extra = pd.read_pickle(C.WORK_DIR / "extra_reports.pkl")
r = r.merge(brand, on="primaryid", how="left").merge(extra, on="primaryid", how="left")
for c in [c for c in r.columns if c.startswith(("diab_", "wt_")) or "_ind_" in c]:
    r[c] = r[c].astype("boolean").fillna(False).astype(bool)
for c in ["glp1_any", "dpp4", "pancreatitis"] + [f"glp1_{a}" for a in C.GLP1_AGENTS]:
    r[c] = r[c].astype(bool)
r["hcp"] = r["reporter_type"].isin(HCP)
r["dpp4_only"] = r["dpp4"] & ~r["glp1_any"]
r["diab_any"] = r[[f"diab_{a}" for a in C.GLP1_AGENTS]].any(axis=1)
r["wt_any"] = r[[f"wt_{a}" for a in WEIGHT_PRODUCTS]].any(axis=1)
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


def crude(exp, ev, univ):
    e, y = exp[univ].values.astype(bool), ev[univ].values.astype(bool)
    a, b, c, d = int((e & y).sum()), int((e & ~y).sum()), int((~e & y).sum()), int((~e & ~y).sum())
    if min(a, b, c, d) == 0:
        return dict(a=a, b=b, c=c, d=d, ROR=np.nan, lower=np.nan, upper=np.nan)
    est = a * d / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return dict(a=a, b=b, c=c, d=d, ROR=est, lower=est * np.exp(-1.96 * se), upper=est * np.exp(1.96 * se))


def adjusted(exp, ev, univ, covs):
    """Logistic regression (IRLS) as in step 09; returns ROR with Wald 95% CI."""
    d = r.loc[univ, covs].copy()
    X = pd.get_dummies(d, drop_first=True, dtype=float)
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


def exp_col(agent, prefix="glp1_"):
    if agent == "GLP-1 RA class":
        return r["glp1_any"] if prefix == "glp1_" else r[f"{prefix}any"]
    col = f"{prefix}{agent}"
    return r[col] if col in r else pd.Series(False, index=r.index)


ind_exp = lambda a: (r["glp1_any"] & r["glp1_ind_diabetes"]) if a == "GLP-1 RA class" else (r[f"glp1_{a}"] & r[f"{a}_ind_diabetes"])
ALL = pd.Series(True, index=r.index)
ANALYSES = [
    ("vs DPP-4i (all)", lambda a: exp_col(a), r["dpp4_only"], ALL),
    ("vs DPP-4i, health-professional reports", lambda a: exp_col(a), r["dpp4_only"], r["hcp"]),
    ("vs DPP-4i, diabetes products", lambda a: exp_col(a, "diab_") & ~r["wt_any"], r["dpp4_only"], ALL),
    ("vs DPP-4i, diabetes products, health-professional reports", lambda a: exp_col(a, "diab_") & ~r["wt_any"], r["dpp4_only"], r["hcp"]),
    ("vs DPP-4i, recorded diabetes indication", ind_exp, r["dpp4_only"] & r["dpp4_ind_diabetes"], ALL),
]
rows = []
for label, exp_fn, ref, sub in ANALYSES:
    for agent in EXPOSURES:
        exp = exp_fn(agent) & ~r["dpp4"]
        univ = (exp | ref) & sub
        res = crude(exp, r["pancreatitis"], univ)
        if res["a"] >= 10 and res["c"] >= 10:
            res["year_ROR"], res["year_lower"], res["year_upper"] = adjusted(exp, r["pancreatitis"], univ, ["year_c"])
            res["adj_ROR"], res["adj_lower"], res["adj_upper"] = adjusted(exp, r["pancreatitis"], univ, COVS)
        rows.append({"analysis": label, "exposure": agent, **res})
out = pd.DataFrame(rows)
out.to_csv(C.RESULTS_DIR / "Year_only_subgroups_dpp4.csv", index=False)
print(out[out.exposure == "GLP-1 RA class"][["analysis", "a", "ROR", "year_ROR", "year_lower", "year_upper", "adj_ROR"]].round(2).to_string(index=False))
