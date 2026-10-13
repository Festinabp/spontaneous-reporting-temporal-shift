"""
09f_sglt2_competition_suspect.py - two further sensitivity analyses for the SGLT2 inhibitor comparison.

  1. Competition bias. SGLT2 inhibitor reports are dominated by other publicised events (diabetic ketoacidosis,
     Fournier's gangrene, amputation), which may lower the proportion of SGLT2 inhibitor reports that include
     pancreatitis and so inflate RORs against this comparator. Reports with a reaction term for any of these events
     are removed from both groups (all terms matched are written to check_competing_terms.csv for review).
  2. Suspect drug only. Exposed: reports in which a GLP-1 RA is a primary or secondary suspect (no SGLT2 inhibitor);
     reference: reports in which an SGLT2 inhibitor is a primary or secondary suspect (no GLP-1 RA).

Crude RORs and logistic-regression RORs adjusted for year only (primary), year and reporter type, and all covariates,
as in step 09b. Low-memory: works from the per-quarter caches of steps 01, 09b and 09e (run 09e first).

Outputs (RESULTS_DIR): SGLT2_competition_suspect.csv, check_competing_terms.csv
"""
import re
import time
import zipfile
import numpy as np
import pandas as pd
import config as C

T0 = time.time()
HCP = {"MD", "PH", "HP", "OT"}
COMPETING = {
    "ketoacidosis": r"KETOACIDO",                       # diabetic / euglycaemic diabetic ketoacidosis, ketoacidotic coma
    "fournier": r"FOURNIER",                            # Fournier's gangrene
    "amputation": r"AMPUTATION",                        # leg, foot, toe, limb amputation
}
COMP_PAT = re.compile("|".join(COMPETING.values()))
SGLT2_PAT = re.compile(r"\b(?:" + "|".join(re.escape(n) for v in C.SGLT2_MAP.values() for n in v) + r")\b")


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def find_member(zf, table):
    pat = re.compile(rf"(^|/){table}\d{{2}}q[1-4](_new)?\.txt$", re.I)
    hits = [n for n in zf.namelist() if pat.search(n)]
    if len(hits) != 1:
        raise RuntimeError(f"{zf.filename}: expected one {table} file, found {hits}")
    return hits[0]


def read_filtered(zf, member, cols, ids):
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


zips = {}
for z in C.BASE_DIR.glob("faers_ascii_*.zip"):
    m = re.search(r"(\d{4})q([1-4])", z.name.lower())
    if m:
        zips[f"{m.group(1)}q{m.group(2)}"] = z

parts, comp_terms = [], {}
for q in sorted(zips):
    cache = C.WORK_DIR / f"sglt2comp_{q}.pkl"
    if cache.exists():
        x_q, ct_q = pd.read_pickle(cache)
    else:
        df_q, _ = pd.read_pickle(C.WORK_DIR / f"sglt2ind_{q}.pkl")        # from step 09e
        rep = pd.read_pickle(C.WORK_DIR / f"reports_{q}.pkl")
        sus = rep[["primaryid"] + [f"glp1_{a}_suspect" for a in C.GLP1_AGENTS]]
        ids = set(df_q["primaryid"].astype("int64"))
        sg_ids = set(df_q.loc[df_q["sglt2"], "primaryid"].astype("int64"))
        with zipfile.ZipFile(zips[q]) as zf:
            reac = read_filtered(zf, find_member(zf, "REAC"), {"primaryid", "pt"}, ids)
            drug = read_filtered(zf, find_member(zf, "DRUG"), {"primaryid", "role_cod", "drugname", "prod_ai"}, sg_ids)
        reac["pt"] = clean(reac["pt"])
        hit = reac[reac["pt"].str.contains(COMP_PAT)]
        ct_q = hit["pt"].value_counts().to_dict()
        text = clean(drug["drugname"]) + " | " + clean(drug["prod_ai"])
        u = pd.Series(text.unique())
        names = set(u[u.str.contains(SGLT2_PAT)])
        s_sus = drug[text.isin(names) & drug["role_cod"].str.upper().str.strip().isin(C.SUSPECT_ROLES)]
        x_q = df_q[["primaryid"]].merge(sus, on="primaryid", how="left")
        x_q["competing"] = x_q["primaryid"].isin(set(hit["primaryid"].astype("int64")))
        for k, pat in COMPETING.items():
            x_q[f"comp_{k}"] = x_q["primaryid"].isin(set(hit.loc[hit["pt"].str.contains(pat), "primaryid"].astype("int64")))
        x_q["sglt2_suspect"] = x_q["primaryid"].isin(set(s_sus["primaryid"].astype("int64")))
        pd.to_pickle((x_q, ct_q), cache)
    parts.append(x_q)
    for k, n in ct_q.items():
        comp_terms[k] = comp_terms.get(k, 0) + n
    log(f"{q}: done")

x = pd.concat(parts, ignore_index=True)
r = pd.concat([pd.read_pickle(C.WORK_DIR / f"sglt2ind_{q}.pkl")[0] for q in sorted(zips)], ignore_index=True)
r = r.merge(x, on="primaryid", how="left", validate="one_to_one")
for c in [c for c in x.columns if c != "primaryid"]:
    r[c] = r[c].astype("boolean").fillna(False).astype(bool)
r["glp1_any_suspect"] = r[[f"glp1_{a}_suspect" for a in C.GLP1_AGENTS]].any(axis=1)
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
glp_s = lambda a: r["glp1_any_suspect"] if a == "GLP-1 RA class" else r[f"glp1_{a}_suspect"]
ALL = pd.Series(True, index=r.index)
ANALYSES = [
    ("vs SGLT2i (all) - check", glp, r["sglt2_only"], ALL),
    ("vs SGLT2i, excluding reports with competing events", glp, r["sglt2_only"], ~r["competing"]),
    ("vs SGLT2i, suspect drug only", glp_s, r["sglt2_only"] & r["sglt2_suspect"], ALL),
]
MODELS = [("year", ["year_c"]), ("yr_rep", ["year_c", "rep_grp"]), ("adj", COVS)]
rows = []
for label, exp_fn, ref, sub in ANALYSES:
    for agent in EXPOSURES:
        exp = exp_fn(agent) & ~r["sglt2"]
        univ = (exp | ref) & sub
        res = crude(exp, r["pancreatitis"], univ)
        if res["a"] >= 10 and res["c"] >= 10:
            for key, covs in MODELS:
                res[f"{key}_ROR"], res[f"{key}_lower"], res[f"{key}_upper"] = adjusted(exp, r["pancreatitis"], univ, covs)
        rows.append({"analysis": label, "exposure": agent, **res})
    log(f"{label} done")
out = pd.DataFrame(rows)
out.to_csv(C.RESULTS_DIR / "SGLT2_competition_suspect.csv", index=False)
pd.Series(comp_terms).sort_values(ascending=False).rename_axis("reaction_term").rename("n_rows") \
  .to_csv(C.RESULTS_DIR / "check_competing_terms.csv")

s, g = r[r["sglt2_only"]], r[r["glp1_any"] & ~r["sglt2"]]
desc = pd.DataFrame({
    "SGLT2i reports": [len(s)] + [int(s[c].sum()) for c in ["competing", "comp_ketoacidosis", "comp_fournier", "comp_amputation", "sglt2_suspect"]],
    "GLP-1 RA reports": [len(g)] + [int(g[c].sum()) for c in ["competing", "comp_ketoacidosis", "comp_fournier", "comp_amputation", "glp1_any_suspect"]],
}, index=["all", "any competing event", "ketoacidosis", "Fournier's gangrene", "amputation", "suspect drug"])
desc.to_csv(C.RESULTS_DIR / "SGLT2_competition_suspect_counts.csv")

pd.set_option("display.width", 250)
print(desc.to_string())
print(pd.Series(comp_terms).sort_values(ascending=False).head(30).to_string())
f = lambda a, b, c: "" if pd.isna(a) else f"{a:.2f} ({b:.2f}–{c:.2f})"
for _, x_ in out.iterrows():
    print(x_["analysis"][:45], "|", x_["exposure"], "|", x_["a"], "|", x_["c"], "|", f(x_["ROR"], x_["lower"], x_["upper"]), "|",
          f(x_.get("year_ROR"), x_.get("year_lower"), x_.get("year_upper")), "|", f(x_.get("yr_rep_ROR"), x_.get("yr_rep_lower"), x_.get("yr_rep_upper")),
          "|", f(x_.get("adj_ROR"), x_.get("adj_lower"), x_.get("adj_upper")))
log("Done")
