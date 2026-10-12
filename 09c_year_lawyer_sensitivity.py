"""
09c_year_lawyer_sensitivity.py

Two further sensitivity analyses for the active-comparator results (run after 09 and 09b; uses their caches):
  1. Year-specific RORs for pancreatitis, GLP-1 RA class vs DPP-4 inhibitors and vs SGLT2 inhibitors, with the
     Mantel-Haenszel ROR stratified by year and a likelihood-ratio test of whether the ROR differs between years
     (logistic regression with and without an exposure x year interaction).
  2. Exclusion of reports submitted by lawyers: crude and fully adjusted RORs for pancreatitis against DPP-4
     inhibitors and SGLT2 inhibitors, for the class and each agent.

Outputs (RESULTS_DIR): Comparator_ROR_by_year.csv, Comparator_ROR_by_year_tests.csv, Lawyer_exclusion.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
import config as C

HCP = {"MD", "PH", "HP", "OT"}
EV = C.PRIMARY_OUTCOME

r = pd.read_pickle(C.WORK_DIR / "reports.pkl")
sg = pd.concat([pd.read_pickle(p) for p in sorted(C.WORK_DIR.glob("sglt2_*.pkl"))]) if list(C.WORK_DIR.glob("sglt2_*.pkl")) else pd.Series(dtype="int64")
r["sglt2"] = r["primaryid"].isin(set(sg.tolist()))
r["hcp"] = r["reporter_type"].isin(HCP)
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


def crude(e, y):
    e, y = np.asarray(e, bool), np.asarray(y, bool)
    a, b, c, d = int((e & y).sum()), int((e & ~y).sum()), int((~e & y).sum()), int((~e & ~y).sum())
    if min(a, b, c, d) == 0:
        return dict(a=a, b=b, c=c, d=d, ROR=np.nan, lower=np.nan, upper=np.nan)
    est = a * d / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return dict(a=a, b=b, c=c, d=d, ROR=est, lower=est * np.exp(-1.96 * se), upper=est * np.exp(1.96 * se))


def irls(X, y):
    beta = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-(X @ beta)))
        w = np.clip(p * (1 - p), 1e-10, None)
        H = X.T @ (X * w[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = np.clip(1 / (1 + np.exp(-(X @ beta))), 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return beta, H, ll


def adjusted(e, y, d):
    X = pd.get_dummies(d[COVS], drop_first=True, dtype=float)
    X = X.loc[:, X.sum(axis=0) > 0]
    X.insert(0, "exposure", np.asarray(e, float))
    X.insert(0, "const", 1.0)
    beta, H, _ = irls(X.values, np.asarray(y, float))
    se = np.sqrt(np.linalg.inv(H)[1, 1])
    return dict(aROR=np.exp(beta[1]), a_lower=np.exp(beta[1] - 1.96 * se), a_upper=np.exp(beta[1] + 1.96 * se))


COMPARATORS = {
    "DPP-4 inhibitors": (lambda d, e: e & ~d["dpp4"], lambda d: d["dpp4"] & ~d["glp1_any"]),
    "SGLT2 inhibitors": (lambda d, e: e & ~d["sglt2"], lambda d: d["sglt2"] & ~d["glp1_any"]),
}

# ----------------------------------------------------------------------------- 1. year-specific RORs
rows, tests = [], []
for comp, (fexp, fref) in COMPARATORS.items():
    exp, ref = fexp(r, r["glp1_any"]), fref(r)
    u = r[exp | ref]
    e, y = exp[exp | ref].values, u[EV].values
    for yr in range(C.YEAR_START, C.YEAR_END + 1):
        m = (u["report_year"] == yr).values
        rows.append({"comparator": comp, "year": yr, **crude(e[m], y[m])})
    # Mantel-Haenszel ROR stratified by year
    num = den = 0.0
    for yr in range(C.YEAR_START, C.YEAR_END + 1):
        m = (u["report_year"] == yr).values
        n = m.sum()
        if n == 0:
            continue
        res = crude(e[m], y[m])
        num += res["a"] * res["d"] / n; den += res["b"] * res["c"] / n
    # likelihood-ratio test for exposure x year interaction
    Y = pd.get_dummies(u["year_c"], drop_first=True, dtype=float)
    X0 = np.column_stack([np.ones(len(u)), e.astype(float), Y.values])
    X1 = np.column_stack([X0, Y.values * e[:, None]])
    _, _, ll0 = irls(X0, y.astype(float)); _, _, ll1 = irls(X1, y.astype(float))
    lr = 2 * (ll1 - ll0)
    tests.append({"comparator": comp, "MH_ROR_by_year": num / den, "LR_chi2_interaction": lr, "df": Y.shape[1],
                  "p_value": stats.chi2.sf(lr, Y.shape[1])})
pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "Comparator_ROR_by_year.csv", index=False)
pd.DataFrame(tests).to_csv(C.RESULTS_DIR / "Comparator_ROR_by_year_tests.csv", index=False)
print("Year-specific RORs done")

# ----------------------------------------------------------------------------- 2. lawyer exclusion
nl = r[~r["reporter_type"].eq("LW")]
rows = []
for comp, (fexp, fref) in COMPARATORS.items():
    ref = fref(nl)
    for agent in EXPOSURES:
        base = nl["glp1_any"] if agent == "GLP-1 RA class" else nl[f"glp1_{agent}"]
        exp = fexp(nl, base)
        univ = exp | ref
        res = crude(exp[univ].values, nl.loc[univ, EV].values)
        if res["a"] >= 10 and res["c"] >= 10:
            res.update(adjusted(exp[univ].values, nl.loc[univ, EV].values, nl.loc[univ]))
        rows.append({"comparator": comp, "exposure": agent, **res})
law = pd.DataFrame(rows)
law.to_csv(C.RESULTS_DIR / "Lawyer_exclusion.csv", index=False)

pd.set_option("display.width", 200)
print(pd.DataFrame(tests).round(4).to_string(index=False))
print(law[law.exposure == "GLP-1 RA class"].round(2).to_string(index=False))
print(f"Reports excluded as lawyer-submitted: {int(r['reporter_type'].eq('LW').sum()):,}")
print("Done.")
