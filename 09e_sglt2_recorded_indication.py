"""
09e_sglt2_recorded_indication.py - SGLT2 inhibitor comparison restricted to reports with a recorded diabetes indication.

Mirrors the DPP-4 inhibitor recorded-indication comparison (steps 09/09d):
  Exposed:   reports with the GLP-1 RA (class: any GLP-1 RA) and no SGLT2 inhibitor, with diabetes recorded as the
             indication for the GLP-1 RA's own drug rows (INDI table; from step 07, work/extra_reports.pkl)
  Reference: reports with an SGLT2 inhibitor and no GLP-1 RA, with diabetes recorded as the indication for the
             SGLT2 inhibitor's own drug rows (INDI table; extracted here)
Indication categories use the same rule as step 07 (diabetes excluding type 1, gestational and insipidus).

Crude RORs and logistic-regression RORs adjusted for year only (primary), for year and reporter type, and for all
covariates, as in step 09b.
Also re-estimates the unrestricted SGLT2 comparison as a check against SGLT2_comparator.csv, and describes the
recorded indications of SGLT2 inhibitor reports.

Low-memory: works quarter by quarter from the per-quarter caches of step 01 (work/demo_*.pkl, work/reports_*.pkl)
and step 09b (work/sglt2_*.pkl), so work/reports.pkl does not need to be loaded.

Outputs (RESULTS_DIR): SGLT2_recorded_indication.csv, SGLT2_indication_categories.csv
Run after 07 and 09b.
"""
import re
import time
import zipfile
import numpy as np
import pandas as pd
import config as C

T0 = time.time()
HCP = {"MD", "PH", "HP", "OT"}
AGE_FACTORS = {"YR": 1.0, "DEC": 10.0, "MON": 1 / 12, "WK": 1 / 52.1775, "DY": 1 / 365.25, "HR": 1 / 8766}


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def find_member(zf, table):
    pat = re.compile(rf"(^|/){table}\d{{2}}q[1-4](_new)?\.txt$", re.I)
    hits = [n for n in zf.namelist() if pat.search(n)]
    if len(hits) != 1:
        raise RuntimeError(f"{zf.filename}: expected one {table} file, found {hits}")
    return hits[0]


def read_filtered(zf, member, cols, ids):
    """Read a $-delimited table in chunks, keeping only rows whose primaryid is in ids."""
    parts = []
    with zf.open(member) as fh:
        for ch in pd.read_csv(fh, sep="$", dtype=str, encoding="latin-1", quoting=3,
                              usecols=lambda c: c.strip().lower() in cols, chunksize=500_000,
                              on_bad_lines="skip", keep_default_na=False, na_values=[""], engine="c"):
            ch.columns = [c.strip().lower() for c in ch.columns]
            ch["primaryid"] = pd.to_numeric(ch["primaryid"], errors="coerce")
            parts.append(ch[ch["primaryid"].isin(ids)])
    return pd.concat(parts, ignore_index=True)


def clean(s):
    return s.fillna("").str.upper().str.replace(r"\s+", " ", regex=True).str.strip(" .,;")


def indication_category(pt):
    """Same rule as step 07."""
    if pt == "" or "UNKNOWN INDICATION" in pt:
        return "unknown"
    if "DIABET" in pt and "TYPE 1" not in pt and "GESTATIONAL" not in pt and "INSIPIDUS" not in pt:
        return "diabetes"
    if re.search(r"OBES|OVERWEIGHT|WEIGHT", pt):
        return "weight"
    return "other"


SGLT2_PAT = re.compile(r"\b(?:" + "|".join(re.escape(n) for v in C.SGLT2_MAP.values() for n in v) + r")\b")

zips = {}
for z in C.BASE_DIR.glob("faers_ascii_*.zip"):
    m = re.search(r"(\d{4})q([1-4])", z.name.lower())
    if m:
        zips[f"{m.group(1)}q{m.group(2)}"] = z
quarters = sorted(zips)

# ----------------------------------------------------------------------------- per quarter
parts, ind_terms = [], {}
for q in quarters:
    cache = C.WORK_DIR / f"sglt2ind_{q}.pkl"
    if cache.exists():
        df_q, it_q = pd.read_pickle(cache)
    else:
        rep = pd.read_pickle(C.WORK_DIR / f"reports_{q}.pkl")
        rep["glp1_any"] = rep[[f"glp1_{a}" for a in C.GLP1_AGENTS]].any(axis=1)
        sg = set(pd.read_pickle(C.WORK_DIR / f"sglt2_{q}.pkl").astype("int64"))
        rep["sglt2"] = rep["primaryid"].isin(sg)
        rep = rep[rep["glp1_any"] | rep["sglt2"]]
        keep = ["primaryid", "pancreatitis", "sglt2", "glp1_any"] + [f"glp1_{a}" for a in C.GLP1_AGENTS]
        rep = rep[keep].copy()
        ids = set(rep["primaryid"].astype("int64"))

        demo = pd.read_pickle(C.WORK_DIR / f"demo_{q}.pkl")
        demo = demo[demo["primaryid"].isin(ids)].copy()
        demo["primaryid"] = demo["primaryid"].astype("int64")
        demo = demo.sort_values("primaryid").drop_duplicates("primaryid", keep="last")
        for c in ["age_cod", "sex", "occp_cod", "reporter_country", "occr_country"]:
            demo[c] = demo[c].astype(object)
        init = demo["init_fda_dt_n"].fillna(demo["fda_dt_n"])
        demo["report_year"] = (init // 10000).astype("Int64")
        age = demo["age"].astype(float) * demo["age_cod"].map(AGE_FACTORS).astype(float)
        demo["age_years"] = age.where(age.between(0, 120))
        demo["sex"] = demo["sex"].where(demo["sex"].isin(["F", "M"]))
        demo["reporter_type"] = demo["occp_cod"].where(demo["occp_cod"].isin(["MD", "PH", "OT", "LW", "CN", "HP"]))
        demo["country"] = demo["reporter_country"].fillna(demo["occr_country"])
        demo = demo[["primaryid", "report_year", "age_years", "sex", "reporter_type", "country"]]
        df_q = rep.merge(demo, on="primaryid", how="left", validate="one_to_one")

        # SGLT2 inhibitor indications (SGLT2-only reports; the SGLT2 inhibitor's own drug rows)
        sg_ids = set(df_q.loc[df_q["sglt2"] & ~df_q["glp1_any"], "primaryid"])
        with zipfile.ZipFile(zips[q]) as zf:
            drug = read_filtered(zf, find_member(zf, "DRUG"), {"primaryid", "drug_seq", "drugname", "prod_ai"}, sg_ids)
            text = clean(drug["drugname"]) + " | " + clean(drug["prod_ai"])
            u = pd.Series(text.unique())
            hit = set(u[u.str.contains(SGLT2_PAT)])
            drug = drug[text.isin(hit)][["primaryid", "drug_seq"]]
            indi = read_filtered(zf, find_member(zf, "INDI"), {"primaryid", "indi_drug_seq", "indi_pt"}, sg_ids)
        indi["indi_pt"] = clean(indi["indi_pt"])
        ind = drug.merge(indi.rename(columns={"indi_drug_seq": "drug_seq"}), on=["primaryid", "drug_seq"], how="left")
        ind["indi_pt"] = ind["indi_pt"].fillna("")
        ind["cat"] = ind["indi_pt"].map(indication_category)
        it_q = ind.groupby(["indi_pt", "cat"]).size().to_dict()
        for cat in ["diabetes", "weight", "other", "unknown"]:
            ids_cat = set(ind.loc[ind["cat"] == cat, "primaryid"].astype("int64"))
            df_q[f"sglt2_ind_{cat}"] = df_q["primaryid"].isin(ids_cat)
        pd.to_pickle((df_q, it_q), cache)
    parts.append(df_q)
    for k, n in it_q.items():
        ind_terms[k] = ind_terms.get(k, 0) + n
    log(f"{q}: {int((df_q['sglt2'] & ~df_q['glp1_any']).sum()):,} SGLT2-only, {int(df_q['glp1_any'].sum()):,} GLP-1 RA reports")

r = pd.concat(parts, ignore_index=True)
extra = pd.read_pickle(C.WORK_DIR / "extra_reports.pkl")
ind_cols = ["glp1_ind_diabetes"] + [f"{a}_ind_diabetes" for a in C.GLP1_AGENTS]
r = r.merge(extra[["primaryid"] + ind_cols], on="primaryid", how="left")
for c in ind_cols:
    r[c] = r[c].astype("boolean").fillna(False).astype(bool)

r["sglt2_only"] = r["sglt2"] & ~r["glp1_any"]
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


def crude(exp, ev, univ):
    e, y = exp[univ].values.astype(bool), ev[univ].values.astype(bool)
    a, b, c, d = int((e & y).sum()), int((e & ~y).sum()), int((~e & y).sum()), int((~e & ~y).sum())
    if min(a, b, c, d) == 0:
        return dict(a=a, b=b, c=c, d=d, ROR=np.nan, lower=np.nan, upper=np.nan)
    est = a * d / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return dict(a=a, b=b, c=c, d=d, ROR=est, lower=est * np.exp(-1.96 * se), upper=est * np.exp(1.96 * se))


def adjusted(exp, ev, univ, covs):
    """Logistic regression (IRLS) as in steps 09/09b; returns ROR with Wald 95% CI."""
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


glp = lambda a: r["glp1_any"] if a == "GLP-1 RA class" else r[f"glp1_{a}"]
glp_ind = lambda a: (r["glp1_any"] & r["glp1_ind_diabetes"]) if a == "GLP-1 RA class" else (r[f"glp1_{a}"] & r[f"{a}_ind_diabetes"])
ALL = pd.Series(True, index=r.index)
ANALYSES = [
    ("vs SGLT2i (all) - check", glp, r["sglt2_only"], ALL),
    ("vs SGLT2i, recorded diabetes indication", glp_ind, r["sglt2_only"] & r["sglt2_ind_diabetes"], ALL),
    ("vs SGLT2i, recorded diabetes indication, health-professional reports", glp_ind, r["sglt2_only"] & r["sglt2_ind_diabetes"], r["hcp"]),
]
rows = []
for label, exp_fn, ref, sub in ANALYSES:
    for agent in EXPOSURES:
        exp = exp_fn(agent) & ~r["sglt2"]
        univ = (exp | ref) & sub
        res = crude(exp, r["pancreatitis"], univ)
        if res["a"] >= 10 and res["c"] >= 10:
            res["year_ROR"], res["year_lower"], res["year_upper"] = adjusted(exp, r["pancreatitis"], univ, ["year_c"])
            if sub is ALL:   # reporter type is constant within health-professional reports
                res["yr_rep_ROR"], res["yr_rep_lower"], res["yr_rep_upper"] = adjusted(exp, r["pancreatitis"], univ, ["year_c", "rep_grp"])
            res["adj_ROR"], res["adj_lower"], res["adj_upper"] = adjusted(exp, r["pancreatitis"], univ, COVS)
        rows.append({"analysis": label, "exposure": agent, **res})
    log(f"{label} done")
out = pd.DataFrame(rows)
out.to_csv(C.RESULTS_DIR / "SGLT2_recorded_indication.csv", index=False)

# Recorded indication of SGLT2 inhibitor reports (SGLT2-only), and of GLP-1 RA reports (no SGLT2 inhibitor)
s = r[r["sglt2_only"]]
g = r[r["glp1_any"] & ~r["sglt2"]]
desc = pd.DataFrame({
    "SGLT2i reports (n)": [len(s), int(s["pancreatitis"].sum())],
    "SGLT2i diabetes recorded (%)": [s["sglt2_ind_diabetes"].mean() * 100, s.loc[s["pancreatitis"], "sglt2_ind_diabetes"].mean() * 100],
    "SGLT2i other indication recorded (%)": [s["sglt2_ind_other"].mean() * 100, s.loc[s["pancreatitis"], "sglt2_ind_other"].mean() * 100],
    "GLP-1 RA reports (n)": [len(g), int(g["pancreatitis"].sum())],
    "GLP-1 RA diabetes recorded (%)": [g["glp1_ind_diabetes"].mean() * 100, g.loc[g["pancreatitis"], "glp1_ind_diabetes"].mean() * 100],
}, index=["all reports", "pancreatitis reports"])
desc.round(1).to_csv(C.RESULTS_DIR / "SGLT2_indication_categories.csv")
it = pd.DataFrame([(t, k, n) for (t, k), n in ind_terms.items()], columns=["indication_term", "category", "n_rows"])
it.sort_values("n_rows", ascending=False).head(100).to_csv(C.RESULTS_DIR / "check_indication_terms_sglt2.csv", index=False)

pd.set_option("display.width", 250)
print(out[out.exposure == "GLP-1 RA class"].round(2).to_string(index=False))
print(out[out.analysis == "vs SGLT2i, recorded diabetes indication"][["exposure", "a", "c", "ROR", "lower", "upper", "year_ROR", "year_lower", "year_upper", "adj_ROR", "adj_lower", "adj_upper"]].round(2).to_string(index=False))
print(desc.round(1).to_string())
print(it.sort_values("n_rows", ascending=False).head(25).to_string(index=False))
log("Done")
