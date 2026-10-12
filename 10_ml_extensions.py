"""
10_ml_extensions.py

Extensions of the machine-learning analysis (primary intake predictors, pancreatitis outcome):
  1. Simple baselines: (a) historical pancreatitis rate of the reported GLP-1 RA(s) in the training data;
     (b) one rule: health-professional reporter, ties broken by the historical agent rate
     with a paired bootstrap of AUROC differences (gradient boosting vs the other methods, temporal validation)
  2. Annual retraining: for each test year from 2018 to 2025, train on all earlier years (expanding window) or on
     the three preceding years (sliding window), and compare with a frozen model trained once on 2015-2017
  3. Subgroup performance under temporal validation, by reporter type and by GLP-1 RA
  4. Population shift: a classifier distinguishing 2015-2022 from 2023-2025 reports (adversarial validation)
  5. Review workload: reports that must be reviewed per pancreatitis case found, and the share of reports
     that must be reviewed to find 50% of cases (ties in the ranking score broken at random; expected values)

Outputs (RESULTS_DIR): ML_baselines.csv, ML_auroc_differences.csv, ML_annual_retraining.csv, ML_subgroups.csv, ML_population_shift.csv,
  ML_workload.csv; figures/Fig_ml_retraining.png
Runtime: about 5-15 minutes.
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import config as C

T0 = time.time()
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
HCP = {"MD", "PH", "HP", "OT"}
OUT = C.PRIMARY_OUTCOME


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


# ----------------------------------------------------------------------------- data (same as 05, primary predictors)
_csv = os.environ.get("FAERS_ML_INPUT")
r = pd.read_csv(_csv, low_memory=False) if _csv else pd.read_pickle(C.WORK_DIR / "reports.pkl")
df = r[r["glp1_any"].astype(bool)].copy().reset_index(drop=True)
del r
# most frequent reporting countries, determined from the temporal training years only (2015-2022)
top_countries = df.loc[df["report_year"].between(*C.TEMPORAL_TRAIN), "country"].value_counts().head(15).index
df["country_grp"] = np.where(df["country"].isna(), "UNKNOWN", np.where(df["country"].isin(top_countries), df["country"], "OTHER"))
df["reporter_type"] = df["reporter_type"].fillna("UNKNOWN")
df["sex_cat"] = df["sex"].fillna("UNKNOWN")
df["age_missing"] = df["age_years"].isna()
df["weight_missing"] = df["weight_kg"].isna()
df["multiple_glp1_exposure"] = df["n_glp1_agents"] > 1
df["polypharmacy"] = df["n_drugs"] >= 5
df["rep_grp"] = np.select([df["reporter_type"].isin(HCP), df["reporter_type"].eq("CN")], ["Health professional", "Consumer"], "Other/unknown")
NUMERIC = ["age_years", "weight_kg", "n_drugs"]
BINARY = ([f"glp1_{a}" for a in C.GLP1_AGENTS] + [f"glp1_{a}_suspect" for a in C.GLP1_AGENTS]
          + ["dpp4", "age_missing", "weight_missing", "multiple_glp1_exposure", "polypharmacy"])
CATEG = ["sex_cat", "reporter_type", "country_grp"]
FEATURES = NUMERIC + BINARY + CATEG
X = df[FEATURES].copy()
X[BINARY] = X[BINARY].astype(float)
X[NUMERIC] = X[NUMERIC].astype(float)
y = df[OUT].astype(int).values
year = df["report_year"].astype(int).values
log(f"GLP-1 RA reports: {len(df):,}")


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


def agent_rate_score(train_idx, test_idx):
    """Highest training-period pancreatitis proportion among the GLP-1 RAs reported (baseline a)."""
    rates = {a: y[train_idx][df.loc[train_idx, f"glp1_{a}"].values.astype(bool)].mean() if df.loc[train_idx, f"glp1_{a}"].any() else np.nan
             for a in C.GLP1_AGENTS}
    overall = y[train_idx].mean()
    mat = np.column_stack([np.where(df.loc[test_idx, f"glp1_{a}"].values.astype(bool), np.nan_to_num(rates[a], nan=overall), -1)
                           for a in C.GLP1_AGENTS])
    s = mat.max(axis=1)
    return np.where(s < 0, overall, s)


def rule_score(train_idx, test_idx):
    """Health-professional reporter first, then the historical agent rate (baseline b)."""
    return df.loc[test_idx, "rep_grp"].eq("Health professional").values.astype(float) + agent_rate_score(train_idx, test_idx)


# Ranking metrics are computed with ties in the score broken at random (expected values), because the simple
# baselines give many reports the same score and a fixed order would otherwise decide which tied reports come first.
def cases_in_top(yt, s, n):
    """Expected number of positive reports among the n highest-ranked reports, ties broken at random."""
    if n <= 0:
        return 0.0
    t = np.sort(s)[::-1][n - 1]                      # score of the n-th ranked report
    above, tied = s > t, s == t
    return float(yt[above].sum() + yt[tied].sum() * (n - above.sum()) / tied.sum())


def recall_at(yt, s, pct):
    n = int(np.ceil(len(yt) * pct / 100))
    return cases_in_top(yt, s, n) / max(yt.sum(), 1)


def share_for_half(yt, s, reps=100):
    """Percentage of reports that must be reviewed to find half of all cases (mean over random tie-breaks)."""
    rng = np.random.default_rng(C.RANDOM_STATE)
    out = []
    for _ in range(reps if len(np.unique(s)) < len(s) else 1):
        order = np.lexsort((rng.random(len(s)), -s))
        cum = np.cumsum(yt[order])
        out.append(100 * (np.searchsorted(cum, 0.5 * yt.sum()) + 1) / len(yt))
    return float(np.mean(out))


def metrics(yt, s):
    out = {"AUROC": roc_auc_score(yt, s), "AUPRC": average_precision_score(yt, s), "prevalence": yt.mean()}
    for k in [1, 5, 10]:
        out[f"recall_top{k}"] = recall_at(yt, s, k)
    return out


# ----------------------------------------------------------------------------- 1 + 5. baselines and workload
idx_all = np.arange(len(df))
tr_r, te_r = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=C.RANDOM_STATE)
tr_t = idx_all[(year >= C.TEMPORAL_TRAIN[0]) & (year <= C.TEMPORAL_TRAIN[1])]
te_t = idx_all[(year >= C.TEMPORAL_TEST[0]) & (year <= C.TEMPORAL_TEST[1])]
scores = {}
rows, work = [], []
for split, tr, te in [("Random split", tr_r, te_r), ("Temporal validation", tr_t, te_t)]:
    yt = y[te]
    cand = {"Baseline: historical agent rate": agent_rate_score(tr, te),
            "Baseline: health-professional rule": rule_score(tr, te)}
    for k, name in [("lr", "Logistic regression"), ("gb", "Gradient boosting")]:
        cand[name] = make_model(k).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    for name, s in cand.items():
        scores[(split, name)] = s
        rows.append({"validation": split, "method": name, **metrics(yt, s)})
        for k in [1, 5, 10]:
            n = int(np.ceil(len(yt) * k / 100))
            cap = cases_in_top(yt, s, n)
            work.append({"validation": split, "method": name, "top_pct": k, "reports_reviewed": n, "cases_found": cap,
                         "reports_per_case": n / cap if cap else np.nan, "random_reports_per_case": 1 / yt.mean(),
                         "share_reviewed_to_find_50pct_cases": share_for_half(yt, s)})
    log(f"{split}: baselines and models done")
pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "ML_baselines.csv", index=False)
pd.DataFrame(work).to_csv(C.RESULTS_DIR / "ML_workload.csv", index=False)

# paired bootstrap of AUROC differences under temporal validation (same resampled reports for both methods)
rng = np.random.default_rng(C.RANDOM_STATE)
yt = y[te_t]
pairs = [("Gradient boosting", m) for m in ["Logistic regression", "Baseline: historical agent rate", "Baseline: health-professional rule"]]
boot = {p: [] for p in pairs}
for _ in range(C.BOOTSTRAP_N):
    i = rng.integers(0, len(yt), len(yt))
    for a_, b_ in pairs:
        boot[(a_, b_)].append(roc_auc_score(yt[i], scores[("Temporal validation", a_)][i]) - roc_auc_score(yt[i], scores[("Temporal validation", b_)][i]))
diff = [{"comparison": f"{a_} minus {b_}",
         "AUROC_difference": roc_auc_score(yt, scores[("Temporal validation", a_)]) - roc_auc_score(yt, scores[("Temporal validation", b_)]),
         "ci_lower": np.percentile(boot[(a_, b_)], 2.5), "ci_upper": np.percentile(boot[(a_, b_)], 97.5)} for a_, b_ in pairs]
pd.DataFrame(diff).to_csv(C.RESULTS_DIR / "ML_auroc_differences.csv", index=False)
log("Paired bootstrap of AUROC differences done")

# ----------------------------------------------------------------------------- 2. annual retraining vs frozen model
rows = []
frozen = {k: make_model(k).fit(X[year <= 2017], y[year <= 2017]) for k in ["lr", "gb"]}
for ty in range(2018, C.YEAR_END + 1):
    tr, te = year < ty, year == ty
    tr3 = (year >= ty - 3) & (year < ty)        # sliding window: the three preceding years only
    if y[te].sum() < 5:
        continue
    for k, name in [("lr", "Logistic regression"), ("gb", "Gradient boosting")]:
        s_re = make_model(k).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        s_sw = make_model(k).fit(X[tr3], y[tr3]).predict_proba(X[te])[:, 1]
        s_fr = frozen[k].predict_proba(X[te])[:, 1]
        for scheme, s in [("Retrained annually (all earlier years)", s_re), ("Retrained annually (previous 3 years)", s_sw),
                          ("Frozen (trained 2015-2017)", s_fr)]:
            rows.append({"test_year": ty, "model": name, "scheme": scheme, "n": int(te.sum()), **metrics(y[te], s)})
    log(f"Annual retraining: {ty} done")
ann = pd.DataFrame(rows)
ann.to_csv(C.RESULTS_DIR / "ML_annual_retraining.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, met in zip(axes, ["AUROC", "AUPRC"]):
    for (name, scheme), d in ann[~ann["scheme"].str.contains("previous 3")].groupby(["model", "scheme"]):
        ls = "-" if scheme.startswith("Retrained") else "--"
        col = "#1f77b4" if name == "Gradient boosting" else "#ff7f0e"
        ax.plot(d["test_year"], d[met], ls, marker="o", ms=3, color=col, label=f"{name}, {scheme.split(' (')[0].lower()}")
    if met == "AUPRC":
        p = ann.drop_duplicates("test_year")
        ax.plot(p["test_year"], p["prevalence"], ":", color="grey", label="Prevalence (random ranking)")
    ax.set(xlabel="Test year", ylabel=met)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7, frameon=False)
axes[1].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(C.FIG_DIR / "Fig_ml_retraining.png", dpi=C.FIG_DPI)
plt.close(fig)

# ----------------------------------------------------------------------------- 3. subgroups (temporal test set)
rows = []
sub = df.loc[te_t]
groups = [("Reporter", g, (sub["rep_grp"] == g).values) for g in ["Health professional", "Consumer", "Other/unknown"]]
groups += [("Agent", a.title(), sub[f"glp1_{a}"].values.astype(bool)) for a in ["SEMAGLUTIDE", "TIRZEPATIDE", "DULAGLUTIDE", "LIRAGLUTIDE"]]
for name in ["Gradient boosting", "Logistic regression", "Baseline: historical agent rate"]:
    s = scores[("Temporal validation", name)]
    for gtype, g, m in groups:
        yt = y[te_t][m]
        if yt.sum() >= 10 and (1 - yt).sum() >= 10:
            rows.append({"method": name, "subgroup_type": gtype, "subgroup": g, "n": int(m.sum()), "cases": int(yt.sum()), **metrics(yt, s[m])})
pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "ML_subgroups.csv", index=False)
log("Subgroups done")

# ----------------------------------------------------------------------------- 4. population shift (adversarial validation)
period = (year >= C.TEMPORAL_TEST[0]).astype(int)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
aucs, imp = [], []
for k, (a, b) in enumerate(cv.split(X, period)):
    m = make_model("gb").fit(X.iloc[a], period[a])
    aucs.append(roc_auc_score(period[b], m.predict_proba(X.iloc[b])[:, 1]))
    if k == 0:
        bb = np.random.default_rng(C.RANDOM_STATE).choice(b, min(30000, len(b)), replace=False)
        pi = permutation_importance(m, X.iloc[bb], period[bb], scoring="roc_auc", n_repeats=3, random_state=C.RANDOM_STATE)
        imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
shift = pd.DataFrame({"metric": ["AUROC (5-fold CV) distinguishing 2023-2025 from 2015-2022 reports", "SD"],
                      "value": [np.mean(aucs), np.std(aucs)]})
shift = pd.concat([shift, pd.DataFrame({"metric": ["Importance: " + i for i in imp.index[:10]], "value": imp.values[:10]})])
shift.to_csv(C.RESULTS_DIR / "ML_population_shift.csv", index=False)
log(f"Population shift AUROC {np.mean(aucs):.3f}")

pd.set_option("display.width", 220)
print(pd.read_csv(C.RESULTS_DIR / "ML_baselines.csv").round(3).to_string(index=False))
print(ann[ann.model == "Gradient boosting"][["test_year", "scheme", "AUROC", "AUPRC", "prevalence"]].round(3).to_string(index=False))
print(pd.read_csv(C.RESULTS_DIR / "ML_subgroups.csv").round(3).to_string(index=False))
print(shift.round(3).to_string(index=False))
log("Done. All results are in the results folder.")
