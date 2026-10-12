"""
05_machine_learning.py

Supervised report prioritisation among GLP-1 RA-exposed reports:
  - logistic regression vs histogram-based gradient boosting
  - stratified random 80/20 split (primary) and temporal validation (train 2015-2022, test 2023-2025)
  - top-k enrichment (1%, 5%, 10%), calibration, Platt/isotonic recalibration (train 2015-2021,
    calibrate 2022, test 2023-2025), bootstrap uncertainty, interpretability
  - the same temporal analysis for the secondary exploratory outcomes

Two feature sets:
  python 05_machine_learning.py               PRIMARY: information available at report intake only
                                              (no seriousness criteria, no report type)
  python 05_machine_learning.py --sensitivity SENSITIVITY: adds seriousness criteria and report type
                                              (post-assessment fields that partly encode the event;
                                              outputs are prefixed SENS_)

Outputs (RESULTS_DIR): Table2/3/4 and supplementary CSVs; figures/Fig3, Fig4 and FigS1-S9 (600 dpi)
"""
import os
import sys
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import config as C

T0 = time.time()
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
MODEL_NAMES = {"lr": "Logistic regression", "gb": "Gradient boosting"}
MODEL_COLORS = {"lr": "#ff7f0e", "gb": "#1f77b4"}
TOPK = [1, 5, 10]


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


SENSITIVITY = "--sensitivity" in sys.argv
PREFIX = "SENS_" if SENSITIVITY else ""


def RES(name):
    return C.RESULTS_DIR / f"{PREFIX}{name}"


def FIG(name):
    return C.FIG_DIR / f"{PREFIX}{name}"


# ----------------------------------------------------------------------------- data and features
_csv = os.environ.get("FAERS_ML_INPUT")
r = pd.read_csv(_csv, low_memory=False) if _csv else pd.read_pickle(C.WORK_DIR / "reports.pkl")
df = r[r["glp1_any"].astype(bool)].copy()
del r
log("Feature set: " + ("SENSITIVITY (with seriousness criteria and report type)" if SENSITIVITY else "PRIMARY (intake information only)"))
log(f"GLP-1 RA-exposed reports: {len(df):,}")

# most frequent reporting countries, determined from the temporal training years only (2015-2022)
top_countries = df.loc[df["report_year"].between(*C.TEMPORAL_TRAIN), "country"].value_counts().head(15).index
df["country_grp"] = np.where(df["country"].isna(), "UNKNOWN", np.where(df["country"].isin(top_countries), df["country"], "OTHER"))
df["reporter_type"] = df["reporter_type"].fillna("UNKNOWN")
df["report_type"] = df["report_type"].fillna("UNKNOWN")
df["sex_cat"] = df["sex"].fillna("UNKNOWN")
df["age_missing"] = df["age_years"].isna()
df["weight_missing"] = df["weight_kg"].isna()
df["multiple_glp1_exposure"] = df["n_glp1_agents"] > 1
df["polypharmacy"] = df["n_drugs"] >= 5

# NOTE: the number of reaction terms is deliberately NOT used as a predictor. Every report has at least one
# reaction term, so a report whose only term is the outcome term can only be outcome-positive; any
# reaction-count feature therefore leaks the label (checked: about half of pancreatitis-positive reports
# have no other reaction term, versus none of the negative reports).
NUMERIC = ["age_years", "weight_kg", "n_drugs"]
# Seriousness criteria and report type are assigned after the event has been assessed (pancreatitis is a
# medically important event, so reports containing it are coded as serious and often expedited). They are
# therefore excluded from the PRIMARY model and only added in the SENSITIVITY model.
BINARY = ([f"glp1_{a}" for a in C.GLP1_AGENTS] + [f"glp1_{a}_suspect" for a in C.GLP1_AGENTS]
          + ["dpp4", "age_missing", "weight_missing", "multiple_glp1_exposure", "polypharmacy"])
CATEG = ["sex_cat", "reporter_type", "country_grp"]
if SENSITIVITY:
    BINARY = [f"serious_{v}" for v in C.OUTC_CODES.values()] + BINARY
    CATEG = CATEG + ["report_type"]
FEATURES = NUMERIC + BINARY + CATEG


def make_X(d, outcome):
    X = d.copy()
    X[BINARY] = X[BINARY].astype(float)
    X[NUMERIC] = X[NUMERIC].astype(float)
    return X[FEATURES]


def make_model(kind):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
        ("bin", "passthrough", BINARY),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEG),
    ])
    if kind == "lr":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", random_state=C.RANDOM_STATE)
    else:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=20, random_state=C.RANDOM_STATE)
    return Pipeline([("pre", pre), ("clf", clf)])


# ----------------------------------------------------------------------------- metrics
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calib_int_slope(y, p):
    m = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def ece(y, p, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "q": q})
    g = d.groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g.n / g.n.sum() * (g.y - g.p).abs()).sum())


def topk(y, p):
    order = np.argsort(-p, kind="stable")
    ys = np.asarray(y)[order]
    rows = []
    for k in TOPK:
        n = int(np.ceil(len(ys) * k / 100))
        cap = int(ys[:n].sum())
        rows.append({"top_pct": k, "reports_reviewed": n, "cases_captured": cap,
                     "precision": cap / n, "recall": cap / max(int(ys.sum()), 1)})
    return rows


def summary(y, p):
    i, s = calib_int_slope(y, p)
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "F1_at_0.5": f1_score(y, (p >= 0.5).astype(int), zero_division=0),
            "Brier": brier_score_loss(y, p), "ECE": ece(y, p), "mean_predicted_risk": float(np.mean(p)),
            "observed_prevalence": float(np.mean(y)), "OE_ratio": float(np.mean(y) / np.mean(p)),
            "calibration_intercept": i, "calibration_slope": s}


def years(d, rng):
    return d["report_year"].between(*rng)


# ----------------------------------------------------------------------------- primary outcome
OUT = C.PRIMARY_OUTCOME
X_all, y_all = make_X(df, OUT), df[OUT].astype(int).values
perf, topk_rows, preds = [], [], {}

# Random split
Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=C.RANDOM_STATE)
models_rand = {}
for k in MODEL_NAMES:
    m = make_model(k).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    models_rand[k] = m
    preds[("random", k)] = (yte, p)
    perf.append({"validation": "Random split", "model": MODEL_NAMES[k], "train_n": len(ytr), "test_n": len(yte),
                 "test_positives": int(yte.sum()), **summary(yte, p)})
    topk_rows += [{"validation": "Random split", "model": MODEL_NAMES[k], **t} for t in topk(yte, p)]
    log(f"Random split {k}: AUROC {perf[-1]['AUROC']:.3f}, AUPRC {perf[-1]['AUPRC']:.3f}")

# Temporal validation
tr, te = years(df, C.TEMPORAL_TRAIN).values, years(df, C.TEMPORAL_TEST).values
for k in MODEL_NAMES:
    m = make_model(k).fit(X_all[tr], y_all[tr])
    p = m.predict_proba(X_all[te])[:, 1]
    preds[("temporal", k)] = (y_all[te], p)
    perf.append({"validation": "Temporal validation", "model": MODEL_NAMES[k], "train_n": int(tr.sum()), "test_n": int(te.sum()),
                 "test_positives": int(y_all[te].sum()), **summary(y_all[te], p)})
    topk_rows += [{"validation": "Temporal validation", "model": MODEL_NAMES[k], **t} for t in topk(y_all[te], p)]
    log(f"Temporal {k}: AUROC {perf[-1]['AUROC']:.3f}, AUPRC {perf[-1]['AUPRC']:.3f}")

perf_df = pd.DataFrame(perf)
perf_df.to_csv(RES("Table2_model_performance.csv"), index=False)
pd.DataFrame(topk_rows).to_csv(RES("Table3_topk.csv"), index=False)

# Bootstrap (temporal test set)
rng = np.random.default_rng(C.RANDOM_STATE)
boot = []
for k in MODEL_NAMES:
    y, p = preds[("temporal", k)]
    vals = {m: [] for m in ["AUROC", "AUPRC"] + [f"Top {t}% recall" for t in TOPK]}
    for _ in range(C.BOOTSTRAP_N):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if yb.sum() == 0:
            continue
        vals["AUROC"].append(roc_auc_score(yb, pb))
        vals["AUPRC"].append(average_precision_score(yb, pb))
        for t in topk(yb, pb):
            vals[f"Top {t['top_pct']}% recall"].append(t["recall"])
    for met, v in vals.items():
        boot.append({"model": MODEL_NAMES[k], "metric": met, "bootstrap_mean": np.mean(v),
                     "ci_lower": np.percentile(v, 2.5), "ci_upper": np.percentile(v, 97.5)})
pd.DataFrame(boot).to_csv(RES("TableS2_bootstrap_temporal.csv"), index=False)
log("Bootstrap done")

# Recalibration: train 2015-2021, calibrate 2022, test 2023-2025
trn, cal, tst = years(df, C.RECAL_TRAIN).values, years(df, C.RECAL_CALIB).values, years(df, C.RECAL_TEST).values
recal_rows, recal_preds = [], {}
for k in MODEL_NAMES:
    m = make_model(k).fit(X_all[trn], y_all[trn])
    p_cal, p_tst = m.predict_proba(X_all[cal])[:, 1], m.predict_proba(X_all[tst])[:, 1]
    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000).fit(logit(p_cal).reshape(-1, 1), y_all[cal])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, y_all[cal])
    versions = {"Original": p_tst, "Sigmoid/Platt": platt.predict_proba(logit(p_tst).reshape(-1, 1))[:, 1],
                "Isotonic": np.clip(iso.predict(p_tst), 1e-6, 1 - 1e-6)}
    for name, pv in versions.items():
        recal_preds[(k, name)] = pv
        recal_rows.append({"model": MODEL_NAMES[k], "calibration": name, **summary(y_all[tst], pv)})
recal = pd.DataFrame(recal_rows)
recal.to_csv(RES("TableS4_recalibration.csv"), index=False)
calib_t = perf_df[perf_df.validation == "Temporal validation"]
calib_t[["model", "Brier", "ECE", "mean_predicted_risk", "observed_prevalence", "calibration_intercept",
         "calibration_slope"]].to_csv(RES("TableS3_calibration_temporal.csv"), index=False)
t4 = pd.concat([
    calib_t.assign(analysis="Temporal calibration"),
    recal.assign(analysis="Recalibration – " + recal["calibration"].str.lower().str.replace("sigmoid/platt", "Platt")),
])[["analysis", "model", "Brier", "ECE", "mean_predicted_risk", "observed_prevalence", "calibration_intercept", "calibration_slope"]]
t4.to_csv(RES("Table4_calibration_recalibration.csv"), index=False)
log("Recalibration done")

# ----------------------------------------------------------------------------- figures (primary outcome)
def roc_pr_panels(axes, key):
    for k in MODEL_NAMES:
        y, p = preds[(key, k)]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ls = ":" if k == "lr" else "-"
        axes[0].plot(fpr, tpr, ls, color=MODEL_COLORS[k], lw=1.8, label=f"{MODEL_NAMES[k]} (AUROC={roc_auc_score(y, p):.3f})")
        axes[1].plot(rec, prec, ls, color=MODEL_COLORS[k], lw=1.8, label=f"{MODEL_NAMES[k]} (AUPRC={average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="green", lw=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve", ylim=(0, 1.02))
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=7.5, loc="lower right" if a is axes[0] else "upper right")


fig, ax = plt.subplots(2, 2, figsize=(7.2, 7))
roc_pr_panels(ax[0], "random")
roc_pr_panels(ax[1], "temporal")
fig.text(0.01, 0.98, "a", fontsize=13, weight="bold")
fig.text(0.01, 0.49, "b", fontsize=13, weight="bold")
fig.tight_layout(rect=(0.02, 0, 1, 1))
fig.savefig(FIG("Fig3_ROC_PR_random_temporal.png"), dpi=C.FIG_DPI)
plt.close(fig)


def topk_plot(key, fname, title=None):
    t = pd.DataFrame(topk_rows)
    t = t[t.validation == ("Random split" if key == "random" else "Temporal validation")]
    fig, a = plt.subplots(figsize=(5.5, 4))
    for k, mk in zip(MODEL_NAMES, ["^", "o"]):
        s = t[t.model == MODEL_NAMES[k]]
        a.plot(s.top_pct, s.recall, marker=mk, color=MODEL_COLORS[k], label=MODEL_NAMES[k])
    a.set(xticks=TOPK, ylim=(0, 1), xlabel="Top % of reports reviewed", ylabel="Recall (cases captured)")
    if title:
        a.set_title(title)
    a.grid(alpha=0.3)
    a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG(fname), dpi=C.FIG_DPI)
    plt.close(fig)


topk_plot("random", "Fig4_topk_random_split.png")
topk_plot("temporal", "FigS1_topk_temporal.png")


def reliability(ax_top, ax_bot, y, p, title):
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    se = np.sqrt(g.y * (1 - g.y) / g.n)
    ax_top.errorbar(g.p, g.y, yerr=1.96 * se, marker="o", color="#ff7f0e", capsize=2)
    lim = max(g.p.max(), g.y.max()) * 1.1
    ax_top.plot([0, lim], [0, lim], "--", color="#1f77b4", lw=0.8)
    s = summary(y, p)
    ax_top.text(0.02, 0.98, f"Brier={s['Brier']:.4f}\nECE={s['ECE']:.4f}\nMean pred={s['mean_predicted_risk']:.4f}\n"
                f"Prev={s['observed_prevalence']:.4f}\nIntercept={s['calibration_intercept']:.3f}\nSlope={s['calibration_slope']:.3f}",
                transform=ax_top.transAxes, va="top", fontsize=6.5, bbox=dict(fc="white", ec="grey"))
    ax_top.set(title=title, xlabel="Mean predicted probability", ylabel="Observed event rate")
    ax_bot.hist(p, bins=30, color="#4c72b0")
    ax_bot.set(xlabel="Predicted probability", ylabel="Count")


fig, ax = plt.subplots(2, 2, figsize=(7.2, 6), gridspec_kw={"height_ratios": [3, 1.3]})
for j, k in enumerate(MODEL_NAMES):
    y, p = preds[("temporal", k)]
    reliability(ax[0, j], ax[1, j], y, p, MODEL_NAMES[k])
fig.tight_layout()
fig.savefig(FIG("FigS2_calibration_temporal.png"), dpi=C.FIG_DPI)
plt.close(fig)

for k, fname in [("lr", "FigS3_recalibration_logistic.png"), ("gb", "FigS4_recalibration_boosting.png")]:
    fig, ax = plt.subplots(2, 3, figsize=(9, 5.5), gridspec_kw={"height_ratios": [3, 1.3]})
    for j, name in enumerate(["Original", "Sigmoid/Platt", "Isotonic"]):
        reliability(ax[0, j], ax[1, j], y_all[tst], recal_preds[(k, name)], name)
    fig.suptitle(MODEL_NAMES[k])
    fig.tight_layout()
    fig.savefig(FIG(fname), dpi=C.FIG_DPI)
    plt.close(fig)

# Interpretability (random-split models): LR coefficients and permutation importance
lr = models_rand["lr"]
names = lr.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=[n.split("__", 1)[1] for n in names])
coef.sort_values().to_csv(RES("TableS_lr_coefficients.csv"), header=["coefficient"])
top = coef.reindex(coef.abs().sort_values(ascending=False).index[:20]).sort_values()
fig, a = plt.subplots(figsize=(6, 5.5))
a.barh(top.index, top.values, color="#1f77b4")
a.axvline(0, color="black", lw=0.6)
a.set_xlabel("Logistic regression coefficient (standardised/one-hot features)")
fig.tight_layout()
fig.savefig(FIG("FigS5_lr_coefficients.png"), dpi=C.FIG_DPI)
plt.close(fig)

n_perm = min(C.PERMUTATION_MAX_ROWS, len(Xte))
idx = np.random.default_rng(C.RANDOM_STATE).choice(len(Xte), n_perm, replace=False)
for k, fname in [("lr", "FigS6_perm_importance_logistic.png"), ("gb", "FigS7_perm_importance_boosting.png")]:
    pi = permutation_importance(models_rand[k], Xte.iloc[idx], yte[idx], scoring="average_precision",
                                n_repeats=C.PERMUTATION_N_REPEATS, random_state=C.RANDOM_STATE, n_jobs=1)
    s = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    s.to_csv(RES(f"TableS_perm_importance_{k}.csv"), header=["importance_mean"])
    s = s.tail(20)
    fig, a = plt.subplots(figsize=(6, 5.5))
    a.barh(s.index, s.values, color="#1f77b4")
    a.set_xlabel("Permutation importance (decrease in average precision)")
    a.set_title(MODEL_NAMES[k])
    fig.tight_layout()
    fig.savefig(FIG(fname), dpi=C.FIG_DPI)
    plt.close(fig)
log("Interpretability done")

# ----------------------------------------------------------------------------- secondary outcomes (temporal)
sec_rows, sec_topk, sec_preds = [], [], {}
for ev in [e for e in C.OUTCOMES if e != OUT]:
    X, y = make_X(df, ev), df[ev].astype(int).values
    if y[tr].sum() < 10 or y[te].sum() < 5:
        log(f"Secondary {ev}: too few events, skipped")
        continue
    for k in MODEL_NAMES:
        p = make_model(k).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        sec_preds[(ev, k)] = (y[te], p)
        sec_rows.append({"outcome": C.OUTCOME_LABELS[ev], "model": MODEL_NAMES[k], "train_n": int(tr.sum()),
                         "test_n": int(te.sum()), "test_positives": int(y[te].sum()), **summary(y[te], p)})
        sec_topk += [{"outcome": C.OUTCOME_LABELS[ev], "model": MODEL_NAMES[k], **t} for t in topk(y[te], p)]
pd.DataFrame(sec_rows).to_csv(RES("TableS_secondary_outcomes_performance.csv"), index=False)
pd.DataFrame(sec_topk).to_csv(RES("TableS_secondary_outcomes_topk.csv"), index=False)

evs = sorted({e for e, _ in sec_preds}, key=C.OUTCOMES.index)
if evs:
    fig, ax = plt.subplots(len(evs), 2, figsize=(7.2, 3 * len(evs)), squeeze=False)
    fig2, ax2 = plt.subplots(len(evs), 1, figsize=(5, 3 * len(evs)), squeeze=False)
    st = pd.DataFrame(sec_topk)
    for i, ev in enumerate(evs):
        for k in MODEL_NAMES:
            y, p = sec_preds[(ev, k)]
            fpr, tpr, _ = roc_curve(y, p)
            prec, rec, _ = precision_recall_curve(y, p)
            ax[i, 0].plot(fpr, tpr, color=MODEL_COLORS[k], label=f"{MODEL_NAMES[k]} (AUROC={roc_auc_score(y, p):.3f})")
            ax[i, 1].plot(rec, prec, color=MODEL_COLORS[k], label=f"{MODEL_NAMES[k]} (AUPRC={average_precision_score(y, p):.3f})")
            s = st[(st.outcome == C.OUTCOME_LABELS[ev]) & (st.model == MODEL_NAMES[k])]
            ax2[i, 0].plot(s.top_pct, s.recall, marker="o", color=MODEL_COLORS[k], label=MODEL_NAMES[k])
        ax[i, 0].plot([0, 1], [0, 1], "--", color="green", lw=0.8)
        ax[i, 0].set(title=f"{C.OUTCOME_LABELS[ev]}: ROC", xlabel="False positive rate", ylabel="True positive rate")
        ax[i, 1].set(title=f"{C.OUTCOME_LABELS[ev]}: precision–recall", xlabel="Recall", ylabel="Precision")
        ax2[i, 0].set(title=C.OUTCOME_LABELS[ev], xticks=TOPK, ylim=(0, 1), xlabel="Top % of reports reviewed", ylabel="Recall")
        for a in (ax[i, 0], ax[i, 1], ax2[i, 0]):
            a.grid(alpha=0.3)
            a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG("FigS8_secondary_ROC_PR.png"), dpi=C.FIG_DPI); plt.close(fig)
    fig2.tight_layout(); fig2.savefig(FIG("FigS9_secondary_topk.png"), dpi=C.FIG_DPI); plt.close(fig2)

print(perf_df[["validation", "model", "train_n", "test_n", "test_positives", "AUROC", "AUPRC", "F1_at_0.5"]].round(3).to_string(index=False))
log("Done. All results are in the results folder.")
