"""
05b_ml_initial_version.py

Sensitivity analysis for the primary machine-learning models: predictors taken from the INITIAL version of each
case (the first report received by the FDA) instead of the latest version, so that no follow-up information is
used. The cohort (GLP-1 RA cases), the outcome (pancreatitis in the latest version) and the year of initial receipt
are unchanged. Both predictor sources are evaluated with the same splits for a direct comparison.

Run after 05_machine_learning.py. Uses the DEMO caches written by step 01 (work/demo_*.pkl) and reads the DRUG table
of the quarter that contains each initial version (cached in work/initdrug_*.pkl).

Outputs (RESULTS_DIR): ML_initial_version.csv
Runtime: about 15-40 minutes for the first run (one pass over the DRUG tables), then a few minutes.
"""
import re
import time
import zipfile
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import config as C

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def find_member(zf, table):
    pat = re.compile(rf"(^|/){table}\d{{2}}q[1-4](_new)?\.txt$", re.I)
    hits = [n for n in zf.namelist() if pat.search(n)]
    if len(hits) != 1:
        raise RuntimeError(f"{zf.filename}: expected one {table} file, found {hits}")
    return hits[0]


def clean(s):
    return s.fillna("").str.upper().str.replace(r"\s+", " ", regex=True).str.strip(" .,;")


def pattern(names):
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b")


GLP1_PAT = {a: pattern(n) for a, n in C.GLP1_MAP.items()}
DPP4_PAT = pattern([n for v in C.DPP4_MAP.values() for n in v])
AGE_FACTORS = {"YR": 1.0, "DEC": 10.0, "MON": 1 / 12, "WK": 1 / 52.1775, "DY": 1 / 365.25, "HR": 1 / 8766}

# ----------------------------------------------------------------------------- latest-version cohort
r = pd.read_pickle(C.WORK_DIR / "reports.pkl")
g = r[r["glp1_any"].astype(bool)].copy().reset_index(drop=True)
del r
cases = set(g["caseid"].astype("int64"))
log(f"GLP-1 RA cases: {len(g):,}")

# ----------------------------------------------------------------------------- initial version of each case
parts = []
for p in sorted(C.WORK_DIR.glob("demo_*.pkl")):
    d = pd.read_pickle(p)
    d = d[d["caseid"].isin(cases)]
    for c in ["age_cod", "sex", "wt_cod", "occp_cod", "reporter_country", "occr_country", "quarter"]:
        d[c] = d[c].astype(object)
    parts.append(d)
dm = pd.concat(parts, ignore_index=True)
del parts
dm = dm.sort_values(["primaryid", "quarter"]).drop_duplicates("primaryid", keep="last")   # same rule as step 01
dm = dm.sort_values(["caseid", "fda_dt_n", "primaryid"]).drop_duplicates("caseid", keep="first")
dm["caseid"] = dm["caseid"].astype("int64")
dm["primaryid"] = dm["primaryid"].astype("int64")
age = dm["age"].astype(float) * dm["age_cod"].map(AGE_FACTORS).astype(float)
dm["age_years"] = age.where(age.between(0, 120))
wt = dm["wt"].astype(float)
wt = np.where(dm["wt_cod"].eq("LBS"), wt * 0.453592, np.where(dm["wt_cod"].eq("KG"), wt, np.nan))
dm["weight_kg"] = pd.Series(wt, index=dm.index).where(lambda s: s.between(0.5, 400))
dm["sex"] = dm["sex"].where(dm["sex"].isin(["F", "M"]))
dm["reporter_type"] = dm["occp_cod"].where(dm["occp_cod"].isin(["MD", "PH", "OT", "LW", "CN", "HP"]))
dm["country"] = dm["reporter_country"].fillna(dm["occr_country"])
init = dm[["caseid", "primaryid", "quarter", "age_years", "weight_kg", "sex", "reporter_type", "country"]].rename(columns={"primaryid": "init_primaryid"})
del dm
log(f"Initial versions found for {len(init):,} cases")

# ----------------------------------------------------------------------------- drugs of the initial version
zips = {}
for z in C.BASE_DIR.glob("faers_ascii_*.zip"):
    m = re.search(r"(\d{4})q([1-4])", z.name.lower())
    if m:
        zips[f"{m.group(1)}q{m.group(2)}"] = z
drug_parts = []
for q in sorted(init["quarter"].unique()):
    cache = C.WORK_DIR / f"initdrug_{q}.pkl"
    if cache.exists():
        drug_parts.append(pd.read_pickle(cache))
        continue
    ids = set(init.loc[init["quarter"] == q, "init_primaryid"])
    with zipfile.ZipFile(zips[q]) as zf, zf.open(find_member(zf, "DRUG")) as fh:
        d = pd.read_csv(fh, sep="$", dtype=str, encoding="latin-1", quoting=3,
                        usecols=lambda c: c.strip().lower() in {"primaryid", "role_cod", "drugname", "prod_ai"},
                        on_bad_lines="skip", keep_default_na=False, na_values=[""], engine="c")
    d.columns = [c.strip().lower() for c in d.columns]
    d["primaryid"] = pd.to_numeric(d["primaryid"], errors="coerce")
    d = d[d["primaryid"].isin(ids)].copy()
    text = clean(d["drugname"]) + " | " + clean(d["prod_ai"])
    suspect = d["role_cod"].str.upper().str.strip().isin(C.SUSPECT_ROLES)
    out = pd.DataFrame(index=pd.Index(sorted(ids), name="init_primaryid"))
    out["n_drugs"] = d.groupby("primaryid").size()
    for a, pat in GLP1_PAT.items():
        m = text.str.contains(pat, na=False)
        out[f"glp1_{a}"] = d.loc[m].groupby("primaryid").size().gt(0)
        out[f"glp1_{a}_suspect"] = d.loc[m & suspect].groupby("primaryid").size().gt(0)
    out["dpp4"] = d.loc[text.str.contains(DPP4_PAT, na=False)].groupby("primaryid").size().gt(0)
    out = out.reset_index()
    out.to_pickle(cache)
    drug_parts.append(out)
    log(f"{q}: initial-version drugs read")
idr = pd.concat(drug_parts, ignore_index=True)
init = init.merge(idr, on="init_primaryid", how="left")

# ----------------------------------------------------------------------------- feature sets
FLAGS = [f"glp1_{a}" for a in C.GLP1_AGENTS] + [f"glp1_{a}_suspect" for a in C.GLP1_AGENTS] + ["dpp4"]
NUMERIC = ["age_years", "weight_kg", "n_drugs"]
BINARY = FLAGS + ["age_missing", "weight_missing", "multiple_glp1_exposure", "polypharmacy"]
CATEG = ["sex_cat", "reporter_type", "country_grp"]
FEATURES = NUMERIC + BINARY + CATEG
train_years = g["report_year"].between(*C.TEMPORAL_TRAIN)


def features(d):
    d = d.copy()
    for c in FLAGS:
        d[c] = d[c].fillna(False).astype(bool)
    d["n_drugs"] = d["n_drugs"].fillna(0)
    top = d.loc[train_years.values, "country"].value_counts().head(15).index
    d["country_grp"] = np.where(d["country"].isna(), "UNKNOWN", np.where(d["country"].isin(top), d["country"], "OTHER"))
    d["reporter_type"] = d["reporter_type"].fillna("UNKNOWN")
    d["sex_cat"] = d["sex"].fillna("UNKNOWN")
    d["age_missing"] = d["age_years"].isna()
    d["weight_missing"] = d["weight_kg"].isna()
    d["multiple_glp1_exposure"] = d[[f"glp1_{a}" for a in C.GLP1_AGENTS]].sum(axis=1) > 1
    d["polypharmacy"] = d["n_drugs"] >= 5
    X = d[FEATURES].copy()
    X[BINARY] = X[BINARY].astype(float)
    X[NUMERIC] = X[NUMERIC].astype(float)
    return X


latest = g[["caseid", "age_years", "weight_kg", "sex", "reporter_type", "country", "n_drugs"] + FLAGS]
initial = g[["caseid"]].merge(init.drop(columns=["init_primaryid", "quarter"]), on="caseid", how="left")
assert (initial["caseid"].values == g["caseid"].values).all()
X_sets = {"Latest case version": features(latest), "Initial case version": features(initial)}
y = g[C.PRIMARY_OUTCOME].astype(int).values
year = g["report_year"].astype(int).values
same = (g["primaryid"].astype("int64").values == g[["caseid"]].merge(init[["caseid", "init_primaryid"]], on="caseid", how="left")["init_primaryid"].values)
info = {"cases": len(g), "cases_with_follow_up_versions": int((~same).sum()),
        "pct_with_follow_up": 100 * float((~same).mean()),
        "initial_version_without_GLP1_RA_named": int((initial[[f"glp1_{a}" for a in C.GLP1_AGENTS]].fillna(False).sum(axis=1) == 0).sum())}
log(info)


def make_model(kind):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEG),
    ])
    clf = (LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=C.RANDOM_STATE)
           if kind == "lr" else
           HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=C.RANDOM_STATE))
    return Pipeline([("pre", pre), ("clf", clf)])


idx = np.arange(len(g))
tr_r, te_r = train_test_split(idx, test_size=0.2, stratify=y, random_state=C.RANDOM_STATE)
tr_t = idx[(year >= C.TEMPORAL_TRAIN[0]) & (year <= C.TEMPORAL_TRAIN[1])]
te_t = idx[(year >= C.TEMPORAL_TEST[0]) & (year <= C.TEMPORAL_TEST[1])]
rows = []
for src, X in X_sets.items():
    for split, tr, te in [("Random split", tr_r, te_r), ("Temporal validation", tr_t, te_t)]:
        for k, name in [("lr", "Logistic regression"), ("gb", "Gradient boosting")]:
            p = make_model(k).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            row = {"predictors": src, "validation": split, "model": name, "test_n": len(te), "positives": int(y[te].sum()),
                   "AUROC": roc_auc_score(y[te], p), "AUPRC": average_precision_score(y[te], p)}
            if split == "Temporal validation":
                rng = np.random.default_rng(C.RANDOM_STATE)
                yt = y[te]
                b = [roc_auc_score(yt[i], p[i]) for i in (rng.integers(0, len(yt), len(yt)) for _ in range(C.BOOTSTRAP_N))]
                row.update(AUROC_lower=np.percentile(b, 2.5), AUROC_upper=np.percentile(b, 97.5))
            rows.append(row)
            log(f"{src} | {split} | {name}: AUROC {row['AUROC']:.3f}")
res = pd.DataFrame(rows)
for k, v in info.items():
    res[k] = v
res.to_csv(C.RESULTS_DIR / "ML_initial_version.csv", index=False)
pd.set_option("display.width", 200)
print(res.iloc[:, :9].round(3).to_string(index=False))
log("Done.")
