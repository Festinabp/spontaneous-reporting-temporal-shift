"""
02_disproportionality.py

Frequentist disproportionality analysis (PRR, ROR, IC) for each GLP-1 RA and the class, plus:
  - sensitivity analysis 1: suspect-only exposure (drug reported as primary or secondary suspect)
  - sensitivity analysis 2: active comparator (GLP-1 RA vs DPP-4 inhibitor reports)
  - year-specific counts and RORs
The EBGM input is written by 01b_ebgm_counts.py; run 03_run_openebgm.R after this script.

Outputs (RESULTS_DIR)
  dpa_main.csv, dpa_suspect_only.csv, dpa_active_comparator_dpp4.csv,
  yearly_counts_pancreatitis.csv, yearly_ror_pancreatitis.csv
"""
import numpy as np
import pandas as pd
import config as C

r = pd.read_pickle(C.WORK_DIR / "reports.pkl")
N_ALL = len(r)
print(f"Reports in analytic dataset: {N_ALL:,}")
EXPOSURES = C.GLP1_AGENTS + ["GLP-1 RA class"]


def exposure(df, agent, suspect=False):
    suffix = "_suspect" if suspect else ""
    if agent == "GLP-1 RA class":
        return df[f"glp1_any{suffix}"]
    return df[f"glp1_{agent}{suffix}"]


def two_by_two(exp, ev):
    exp, ev = exp.astype(bool).values, ev.astype(bool).values
    a = int((exp & ev).sum()); b = int((exp & ~ev).sum())
    c = int((~exp & ev).sum()); d = int((~exp & ~ev).sum())
    return a, b, c, d


def measures(a, b, c, d):
    out = {"a": a, "b": b, "c": c, "d": d}
    n = a + b + c + d
    if n == 0:
        return {**out, "PRR": np.nan, "PRR_lower": np.nan, "PRR_upper": np.nan, "ROR": np.nan, "ROR_lower": np.nan,
                "ROR_upper": np.nan, "expected": np.nan, "IC": np.nan, "IC025": np.nan, "IC975": np.nan,
                "min_count_ok": False, "PRR_signal": False, "ROR_signal": False, "IC_signal": False}
    with np.errstate(divide="ignore", invalid="ignore"):
        # PRR with 95% CI
        prr = (a / (a + b)) / (c / (c + d)) if a and c else np.nan
        se = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d)) if a and c else np.nan
        out.update(PRR=prr, PRR_lower=prr * np.exp(-1.96 * se), PRR_upper=prr * np.exp(1.96 * se))
        # ROR with 95% CI
        ror = (a * d) / (b * c) if a and b and c and d else np.nan
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d) if a and b and c and d else np.nan
        out.update(ROR=ror, ROR_lower=ror * np.exp(-1.96 * se), ROR_upper=ror * np.exp(1.96 * se))
        # IC = log2(observed/expected) without shrinkage; approximate 95% interval (normal approximation)
        expected = (a + b) * (a + c) / n
        ic = np.log2(a / expected) if a > 0 and expected > 0 else np.nan
        se_ic = np.sqrt(1 / a + 1 / (a + b) + 1 / (a + c) + 1 / n) / np.log(2) if a > 0 else np.nan
        out.update(expected=expected, IC=ic, IC025=ic - 1.96 * se_ic, IC975=ic + 1.96 * se_ic)
    out["min_count_ok"] = a >= C.MIN_CASES
    out["PRR_signal"] = bool(out["min_count_ok"] and prr >= 2)
    out["ROR_signal"] = bool(out["min_count_ok"] and out["ROR_lower"] > 1)
    out["IC_signal"] = bool(out["min_count_ok"] and out["IC025"] > 0)
    return out


def run(df, suspect=False):
    rows = []
    for ev in C.OUTCOMES:
        for agent in EXPOSURES:
            m = measures(*two_by_two(exposure(df, agent, suspect), df[ev]))
            rows.append({"event": ev, "exposure": agent, **m})
    return pd.DataFrame(rows)


# ---- Main analysis: any role, all other reports as comparator
main = run(r)
main.to_csv(C.RESULTS_DIR / "dpa_main.csv", index=False)
print(main.loc[main.event == C.PRIMARY_OUTCOME, ["exposure", "a", "PRR", "ROR", "ROR_lower", "ROR_upper", "IC025"]].round(2).to_string(index=False))

# ---- Sensitivity 1: suspect-only exposure (PS/SS); all other reports as comparator
sus = run(r, suspect=True)
sus.to_csv(C.RESULTS_DIR / "dpa_suspect_only.csv", index=False)

# ---- Sensitivity 2: active comparator. Universe = reports with the GLP-1 RA (no DPP-4i) + reports with a
#      DPP-4i (no GLP-1 RA). Exposed = GLP-1 RA, reference = DPP-4i.
rows = []
dpp4_only = r["dpp4"] & ~r["glp1_any"]
for ev in C.OUTCOMES:
    for agent in EXPOSURES:
        exp = exposure(r, agent) & ~r["dpp4"]
        univ = exp | dpp4_only
        m = measures(*two_by_two(exp[univ], r.loc[univ, ev]))
        rows.append({"event": ev, "exposure": agent, "comparator": "DPP-4 inhibitors",
                     **{k: m[k] for k in ["a", "b", "c", "d", "ROR", "ROR_lower", "ROR_upper", "min_count_ok"]}})
ac = pd.DataFrame(rows)
ac.to_csv(C.RESULTS_DIR / "dpa_active_comparator_dpp4.csv", index=False)

# ---- Year-specific counts and RORs (pancreatitis)
years = range(C.YEAR_START, C.YEAR_END + 1)
cnt, yr_rows = [], []
for y in years:
    ry = r[r["report_year"] == y]
    row = {"year": y, "all_reports": len(ry)}
    for agent in EXPOSURES:
        exp = exposure(ry, agent)
        row[agent] = int((exp & ry[C.PRIMARY_OUTCOME]).sum())
        m = measures(*two_by_two(exp, ry[C.PRIMARY_OUTCOME]))
        yr_rows.append({"year": y, "exposure": agent, **{k: m[k] for k in ["a", "ROR", "ROR_lower", "ROR_upper", "ROR_signal"]}})
    cnt.append(row)
pd.DataFrame(cnt).to_csv(C.RESULTS_DIR / "yearly_counts_pancreatitis.csv", index=False)
pd.DataFrame(yr_rows).to_csv(C.RESULTS_DIR / "yearly_ror_pancreatitis.csv", index=False)

print("Done. Next: run 03_run_openebgm.R in R/RStudio (its input is written by 01b_ebgm_counts.py).")
