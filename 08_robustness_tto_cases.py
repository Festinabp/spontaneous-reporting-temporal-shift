"""
08_robustness_tto_cases.py

Robustness, reporting-composition, case-description and time-to-onset analyses for pancreatitis.

  A. Robustness ladder: ROR for each GLP-1 RA and the class under progressively stricter comparisons
       1 Main (any role, all other reports)            2 Suspect drug only
       3 Health-professional reports only              4 Excluding reports with other pancreatitis causes
       5 vs DPP-4 inhibitors                           6 vs DPP-4 inhibitors, diabetes indication
       7 vs DPP-4 inhibitors, fully restricted (diabetes indication, suspect, health professional,
         no other pancreatitis causes)
  B. Indication contrast within GLP-1 RA reports: weight/obesity vs diabetes indication
  C. Reporting composition over time (reporter type of GLP-1 RA reports, % with pancreatitis)
  D. Characteristics of GLP-1 RA pancreatitis reports (and DPP-4i pancreatitis reports for comparison)
  E. Time to onset (event date minus therapy start date) with Weibull shape parameter

Outputs (RESULTS_DIR): Robustness_ladder.csv, Indication_contrast.csv, Reporting_composition_by_year.csv,
  Case_characteristics.csv, Descriptive_numbers.csv, Time_to_onset.csv; figures/Fig_robustness_ladder.png,
  figures/Fig_reporting_composition.png, figures/Fig_time_to_onset.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import config as C

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
HCP = {"MD", "PH", "HP", "OT"}
EV = C.PRIMARY_OUTCOME

r = pd.read_pickle(C.WORK_DIR / "reports.pkl")
extra = pd.read_pickle(C.WORK_DIR / "extra_reports.pkl")
rf = pd.read_pickle(C.WORK_DIR / "riskfactor_ids.pkl")
tto_raw = pd.read_pickle(C.WORK_DIR / "extra_tto.pkl")

r = r.merge(extra, on="primaryid", how="left")
for c in [c for c in extra.columns if "_ind_" in c]:
    r[c] = r[c].fillna(False).astype(bool)
rf_any = set().union(*rf.values())
r["other_cause"] = r["primaryid"].isin(rf_any)
r["hcp"] = r["reporter_type"].isin(HCP)
r["dpp4_only"] = r["dpp4"] & ~r["glp1_any"]
EXPOSURES = C.GLP1_AGENTS + ["GLP-1 RA class"]


def exp_col(agent, suspect=False):
    s = "_suspect" if suspect else ""
    return r["glp1_any" + s] if agent == "GLP-1 RA class" else r[f"glp1_{agent}{s}"]


def ror(exp, ev, univ):
    e, y = exp[univ].values.astype(bool), ev[univ].values.astype(bool)
    a, b, c, d = int((e & y).sum()), int((e & ~y).sum()), int((~e & y).sum()), int((~e & ~y).sum())
    if min(a, b, c, d) == 0:
        return dict(a=a, b=b, c=c, d=d, ROR=np.nan, lower=np.nan, upper=np.nan)
    est = a * d / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return dict(a=a, b=b, c=c, d=d, ROR=est, lower=est * np.exp(-1.96 * se), upper=est * np.exp(1.96 * se))


# ----------------------------------------------------------------------------- A. robustness ladder
ALL = pd.Series(True, index=r.index)
STEPS = [
    ("1 Main analysis (all reports)", False, ALL, False, False),
    ("2 Suspect drug only", True, ALL, False, False),
    ("3 Health-professional reports only", False, r["hcp"], False, False),
    ("4 Excluding other pancreatitis causes", False, ~r["other_cause"], False, False),
    ("5 vs DPP-4 inhibitors", False, ALL, True, False),
    ("6 vs DPP-4 inhibitors, diabetes indication", False, ALL, True, True),
    ("7 vs DPP-4 inhibitors, fully restricted", True, r["hcp"] & ~r["other_cause"], True, True),
]
rows = []
for name, suspect, base, comparator, diab in STEPS:
    for agent in EXPOSURES:
        exp = exp_col(agent, suspect)
        if comparator:
            ref = r["dpp4_only"] & (r["dpp4_suspect"] if suspect else True)
            if diab:  # diabetes indication recorded for the agent's own drug rows (any GLP-1 RA for the class)
                exp = exp & (r["glp1_ind_diabetes"] if agent == "GLP-1 RA class" else r[f"{agent}_ind_diabetes"])
                ref = ref & r["dpp4_ind_diabetes"]
            exp = exp & ~r["dpp4"]
            univ = base & (exp | ref)
        else:
            univ = base
        rows.append({"step": name, "exposure": agent, **ror(exp, r[EV], univ)})
ladder = pd.DataFrame(rows)
ladder.to_csv(C.RESULTS_DIR / "Robustness_ladder.csv", index=False)

show = ["GLP-1 RA class", "LIRAGLUTIDE", "SEMAGLUTIDE", "DULAGLUTIDE", "TIRZEPATIDE", "EXENATIDE"]
fig, axes = plt.subplots(2, 3, figsize=(9, 6.2), sharey=True)
labels = [s[0] for s in STEPS]
for ax, ag in zip(axes.flat, show):
    d = ladder[ladder.exposure == ag].reset_index(drop=True)
    y = np.arange(len(d))[::-1]
    ok = d.ROR.notna().values
    cols = ["#333333"] * 4 + ["#d62728"] * 3
    for yi, (_, row), col in zip(y, d.iterrows(), cols):
        if pd.notna(row.ROR):
            ax.errorbar(row.ROR, yi, xerr=[[row.ROR - row.lower], [row.upper - row.ROR]], fmt="o", color=col, capsize=2, ms=4)
    ax.axvline(1, ls="--", color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_title(ag.title().replace("Glp-1 Ra Class", "GLP-1 RA class"), fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(alpha=0.3, axis="x")
for ax in axes[-1]:
    ax.set_xlabel("ROR for pancreatitis (95% CI)")
fig.text(0.5, 0.005, "Steps: 1 main; 2 suspect only; 3 health-professional reports; 4 excluding other causes; "
         "5 vs DPP-4i; 6 vs DPP-4i, diabetes indication; 7 vs DPP-4i, fully restricted", ha="center", fontsize=7.5)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(C.FIG_DIR / "Fig_robustness_ladder.png", dpi=C.FIG_DPI)
plt.close(fig)

# ----------------------------------------------------------------------------- B. indication contrast
rows = []
for agent in EXPOSURES:
    p = "glp1" if agent == "GLP-1 RA class" else agent   # indication recorded for the agent's own drug rows
    diab_only = r[f"{p}_ind_diabetes"] & ~r[f"{p}_ind_weight"]
    weight_only = r[f"{p}_ind_weight"] & ~r[f"{p}_ind_diabetes"]
    exp = exp_col(agent)
    univ = exp & (diab_only | weight_only)
    res = ror(weight_only, r[EV], univ)
    rows.append({"exposure": agent, "weight_reports": res["a"] + res["b"], "weight_pancreatitis": res["a"],
                 "diabetes_reports": res["c"] + res["d"], "diabetes_pancreatitis": res["c"],
                 "ROR_weight_vs_diabetes": res["ROR"], "lower": res["lower"], "upper": res["upper"]})
pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "Indication_contrast.csv", index=False)

# ----------------------------------------------------------------------------- C. reporting composition
g = r[r["glp1_any"]].copy()
g["reporter_group"] = np.select([g["hcp"], g["reporter_type"].eq("CN"), g["reporter_type"].eq("LW")],
                                ["Health professional", "Consumer", "Lawyer"], "Unknown")
comp = pd.crosstab(g["report_year"], g["reporter_group"], normalize="index") * 100
comp["GLP-1 RA reports (n)"] = g.groupby("report_year").size()
comp["Pancreatitis (%)"] = g.groupby("report_year")[EV].mean() * 100
comp.round(2).to_csv(C.RESULTS_DIR / "Reporting_composition_by_year.csv")
fig, ax = plt.subplots(figsize=(7.2, 4))
order = [c for c in ["Health professional", "Consumer", "Lawyer", "Unknown"] if c in comp]
colors = {"Health professional": "#1f77b4", "Consumer": "#ff7f0e", "Lawyer": "#9467bd", "Unknown": "#c7c7c7"}
bottom = np.zeros(len(comp))
for c in order:
    ax.bar(comp.index, comp[c], bottom=bottom, color=colors[c], label=c, width=0.8)
    bottom += comp[c].values
ax.set_ylabel("Share of GLP-1 RA reports (%)")
ax.set_xlabel("Year of initial FDA receipt")
ax.set_xticks(comp.index)
ax2 = ax.twinx()
ax2.plot(comp.index, comp["Pancreatitis (%)"], "k-o", ms=4, label="Reports with pancreatitis (%)")
ax2.set_ylabel("Reports with pancreatitis (%)")
ax2.set_ylim(0, max(5, comp["Pancreatitis (%)"].max() * 1.2))
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False)
fig.tight_layout()
fig.savefig(C.FIG_DIR / "Fig_reporting_composition.png", dpi=C.FIG_DPI)
plt.close(fig)

# ----------------------------------------------------------------------------- D. case characteristics
def describe(mask, label, ind):
    """ind: prefix of the indication columns for the drug(s) described (glp1, an agent name, or dpp4)."""
    d = r[mask]
    n = len(d)
    q = d["age_years"].quantile([0.25, 0.5, 0.75])
    w = d["weight_kg"].quantile([0.25, 0.5, 0.75])
    pct = lambda s: 100 * s.sum() / n if n else np.nan
    return {"group": label, "n": n,
            "age_median": q[0.5], "age_q1": q[0.25], "age_q3": q[0.75], "age_missing_pct": 100 * d["age_years"].isna().mean(),
            "female_pct_of_known": 100 * (d["sex"] == "F").sum() / max(d["sex"].notna().sum(), 1),
            "weight_median": w[0.5], "weight_q1": w[0.25], "weight_q3": w[0.75],
            "health_professional_pct": pct(d["hcp"]), "consumer_pct": pct(d["reporter_type"].eq("CN")),
            "lawyer_pct": pct(d["reporter_type"].eq("LW")), "US_pct": pct(d["country"].eq("US")),
            "death_pct": pct(d["serious_death"]), "lifethreatening_pct": pct(d["serious_lifethreatening"]),
            "hospitalization_pct": pct(d["serious_hospitalization"]),
            "diabetes_indication_pct": pct(d[f"{ind}_ind_diabetes"]),
            "weight_indication_pct": pct(d[f"{ind}_ind_weight"]),
            "other_cause_pct": pct(d["other_cause"])}


rows = [describe(r["glp1_any"] & r[EV], "GLP-1 RA class", "glp1")]
rows += [describe(r[f"glp1_{a}"] & r[EV], a.title(), a) for a in C.GLP1_AGENTS]
rows += [describe(r["dpp4_only"] & r[EV], "DPP-4 inhibitors (no GLP-1 RA)", "dpp4")]
pd.DataFrame(rows).round(1).to_csv(C.RESULTS_DIR / "Case_characteristics.csv", index=False)

# Seriousness and report type of GLP-1 RA reports with vs without pancreatitis; median year of receipt
g = r[r["glp1_any"]]
misc = []
for lab, m in [("with pancreatitis", g[EV]), ("without pancreatitis", ~g[EV])]:
    misc.append((f"GLP-1 RA reports {lab}: n", int(m.sum())))
    misc.append((f"GLP-1 RA reports {lab}: hospitalization (%)", 100 * g.loc[m, "serious_hospitalization"].mean()))
    misc.append((f"GLP-1 RA reports {lab}: expedited report type (%)", 100 * g.loc[m, "report_type"].eq("EXP").mean()))
misc.append(("Median year of initial receipt, GLP-1 RA reports without DPP-4i", r.loc[r["glp1_any"] & ~r["dpp4"], "report_year"].median()))
misc.append(("Median year of initial receipt, DPP-4i reports without GLP-1 RA", r.loc[r["dpp4_only"], "report_year"].median()))
pd.DataFrame(misc, columns=["quantity", "value"]).to_csv(C.RESULTS_DIR / "Descriptive_numbers.csv", index=False)

# ----------------------------------------------------------------------------- E. time to onset
ev = r.loc[r["glp1_any"] & r[EV], ["primaryid", "event_dt"]]
ev = ev[ev["event_dt"].fillna("").astype(str).str.fullmatch(r"\d{8}")]
t = tto_raw.merge(ev, on="primaryid", how="inner")
t["start"] = pd.to_datetime(t["start_dt"], format="%Y%m%d", errors="coerce")
t["event"] = pd.to_datetime(t["event_dt"], format="%Y%m%d", errors="coerce")
t["tto_days"] = (t["event"] - t["start"]).dt.days
t = t[t["tto_days"].notna()]
# class: earliest start date of any GLP-1 RA in the report, chosen before the 0-3650 day window is applied
t_cls = t.loc[t.groupby("primaryid")["tto_days"].idxmax()]
t_cls = t_cls[t_cls["tto_days"].between(0, 3650)]
t = t[t["tto_days"].between(0, 3650)]
rng = np.random.default_rng(C.RANDOM_STATE)
rows = []
for agent in ["All GLP-1 RAs"] + C.GLP1_AGENTS:
    x = (t_cls if agent == "All GLP-1 RAs" else t[t["agent"] == agent])["tto_days"].values
    row = {"exposure": agent.title() if agent != "All GLP-1 RAs" else agent, "n": len(x)}
    if len(x) >= 10:
        row.update(median=np.median(x), q1=np.percentile(x, 25), q3=np.percentile(x, 75),
                   within_30d_pct=100 * np.mean(x <= 30), within_90d_pct=100 * np.mean(x <= 90),
                   within_365d_pct=100 * np.mean(x <= 365))
        xx = np.where(x == 0, 0.5, x)
        shape = stats.weibull_min.fit(xx, floc=0)[0]
        boots = [stats.weibull_min.fit(rng.choice(xx, len(xx)), floc=0)[0] for _ in range(200)]
        row.update(weibull_shape=shape, shape_lower=np.percentile(boots, 2.5), shape_upper=np.percentile(boots, 97.5))
    rows.append(row)
tto = pd.DataFrame(rows)
tto.round(2).to_csv(C.RESULTS_DIR / "Time_to_onset.csv", index=False)

plot_agents = [a for a in C.GLP1_AGENTS if (t["agent"] == a).sum() >= 10]
fig, ax = plt.subplots(figsize=(7.2, 4))
data = [np.where(t.loc[t.agent == a, "tto_days"].values == 0, 0.5, t.loc[t.agent == a, "tto_days"].values) for a in plot_agents]
bp = ax.boxplot(data, showfliers=False, patch_artist=True)
for patch, a in zip(bp["boxes"], plot_agents):
    patch.set_facecolor(C.AGENT_COLORS[a])
    patch.set_alpha(0.6)
ax.set_xticks(range(1, len(plot_agents) + 1))
ax.set_xticklabels([f"{a.title()}\n(n={(t.agent == a).sum()})" for a in plot_agents], fontsize=8)
ax.set_yscale("log")
ax.set_ylabel("Time to onset (days, log scale)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(C.FIG_DIR / "Fig_time_to_onset.png", dpi=C.FIG_DPI)
plt.close(fig)

pd.set_option("display.width", 200)
print(ladder[ladder.exposure == "GLP-1 RA class"][["step", "a", "ROR", "lower", "upper"]].round(2).to_string(index=False))
print(tto.round(1).to_string(index=False))
print("Done. All results are in the results folder.")
