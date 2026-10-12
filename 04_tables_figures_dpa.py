"""
04_tables_figures_dpa.py

Combines the frequentist results (02) with EBGM (03) and produces the manuscript tables and figures
for the disproportionality part.

Outputs (RESULTS_DIR)
  Table1_disproportionality_pancreatitis.csv
  TableS_disproportionality_secondary_outcomes.csv
  TableS_sensitivity_suspect_only.csv
  TableS_sensitivity_active_comparator_dpp4.csv
  figures/Fig1_forest_ROR_pancreatitis.png
  figures/Fig2_yearly_pancreatitis_counts.png
  figures/FigS_forest_sensitivity.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

main = pd.read_csv(C.RESULTS_DIR / "dpa_main.csv")
sus = pd.read_csv(C.RESULTS_DIR / "dpa_suspect_only.csv")
ac = pd.read_csv(C.RESULTS_DIR / "dpa_active_comparator_dpp4.csv")
eb_path = C.RESULTS_DIR / "ebgm_results.csv"
if eb_path.exists():
    eb = pd.read_csv(eb_path)[["exposure", "event", "EBGM", "EB05", "EB95"]]
    main = main.merge(eb, on=["exposure", "event"], how="left")
    main["EB05_signal"] = main["EB05"] > 1
else:
    warnings.warn("ebgm_results.csv not found - run 03_run_openebgm.R first. EBGM columns left empty.")
    for c in ["EBGM", "EB05", "EB95"]:
        main[c] = np.nan
    main["EB05_signal"] = np.nan

sig_cols = ["PRR_signal", "ROR_signal", "IC_signal"]
main["all_criteria_met"] = main[sig_cols].all(axis=1) & (main["EB05_signal"] == True)
main.to_csv(C.RESULTS_DIR / "dpa_main_with_ebgm.csv", index=False)


def f2(x):
    return "" if pd.isna(x) else f"{x:.2f}"


def fmt(df):
    out = pd.DataFrame({
        "Exposure": df["exposure"].str.title().str.replace("Glp-1 Ra Class", "GLP-1 RA class"),
        "Cases (n)": df["a"].map("{:,}".format),
        "PRR (95% CI)": [f"{f2(p)} ({f2(l)}–{f2(u)})" for p, l, u in zip(df.PRR, df.PRR_lower, df.PRR_upper)],
        "ROR (95% CI)": [f"{f2(p)} ({f2(l)}–{f2(u)})" for p, l, u in zip(df.ROR, df.ROR_lower, df.ROR_upper)],
        "IC (IC025–IC975)": [f"{f2(p)} ({f2(l)}–{f2(u)})" for p, l, u in zip(df.IC, df.IC025, df.IC975)],
    })
    if "EBGM" in df:
        out["EBGM"] = df["EBGM"].map(f2)
        out["EB05"] = df["EB05"].map(f2)
        out["EB95"] = df["EB95"].map(f2)
        out["All criteria met"] = df["all_criteria_met"].map({True: "Yes", False: "No"})
    return out


p = main[main.event == C.PRIMARY_OUTCOME]
fmt(p).to_csv(C.RESULTS_DIR / "Table1_disproportionality_pancreatitis.csv", index=False)
sec = main[main.event != C.PRIMARY_OUTCOME]
t = fmt(sec)
t.insert(0, "Event", sec["event"].map(C.OUTCOME_LABELS).values)
t.to_csv(C.RESULTS_DIR / "TableS_disproportionality_secondary_outcomes.csv", index=False)

s = sus[sus.event == C.PRIMARY_OUTCOME]
s_t = fmt(s.drop(columns=[c for c in ["EBGM"] if c in s]))
s_t.to_csv(C.RESULTS_DIR / "TableS_sensitivity_suspect_only.csv", index=False)

a = ac[ac.event == C.PRIMARY_OUTCOME]
pd.DataFrame({
    "Exposure": a["exposure"].str.title().str.replace("Glp-1 Ra Class", "GLP-1 RA class"),
    "GLP-1 RA reports with pancreatitis (a)": a["a"], "GLP-1 RA reports without (b)": a["b"],
    "DPP-4i reports with pancreatitis (c)": a["c"], "DPP-4i reports without (d)": a["d"],
    "ROR vs DPP-4i (95% CI)": [f"{f2(r)} ({f2(l)}–{f2(u)})" for r, l, u in zip(a.ROR, a.ROR_lower, a.ROR_upper)],
}).to_csv(C.RESULTS_DIR / "TableS_sensitivity_active_comparator_dpp4.csv", index=False)

# ---------------------------------------------------------------- Figure 1: forest plot (main ROR)
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
agents = C.GLP1_AGENTS
rows_f = agents + ["GLP-1 RA class"]
pa = p.set_index("exposure").loc[rows_f]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for i, ag in enumerate(rows_f[::-1]):
    row = pa.loc[ag]
    if pd.isna(row.ROR):
        continue
    cls = ag == "GLP-1 RA class"
    ax.errorbar(row.ROR, i, xerr=[[row.ROR - row.ROR_lower], [row.ROR_upper - row.ROR]], fmt="D" if cls else "s",
                color="black" if cls else "#444444", ecolor="black", capsize=3, ms=7)
    ax.text(row.ROR_upper * 1.12, i, f"{row.ROR:.2f} [{row.ROR_lower:.2f}–{row.ROR_upper:.2f}]", va="center", fontsize=9)
ax.axvline(1, ls="--", color="grey", lw=1)
ax.set_xscale("log")
ax.set_yticks(range(len(rows_f)))
ax.set_yticklabels([a.title() if a != "GLP-1 RA class" else "GLP-1 RA class" for a in rows_f[::-1]])
lo = max(0.1, np.nanmin(pa.ROR_lower) / 2)
ax.set_xlim(min(0.8, lo), np.nanmax(pa.ROR_upper) * 3.5)
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
ticks = [t for t in [0.5, 1, 2, 5, 10, 20, 50, 100] if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel("Reporting odds ratio (ROR) with 95% CI (log scale)")
fig.tight_layout()
fig.savefig(C.FIG_DIR / "Fig1_forest_ROR_pancreatitis.png", dpi=C.FIG_DPI)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: yearly counts
yc = pd.read_csv(C.RESULTS_DIR / "yearly_counts_pancreatitis.csv")
markers = ["s", "o", "^", "D", "v", "s", "p"]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for ag, mk in zip(agents, markers):
    ax.plot(yc.year, yc[ag], marker=mk, color=C.AGENT_COLORS[ag], label=ag.title(), lw=1.6, ms=5)
ax.plot(yc.year, yc["GLP-1 RA class"], marker="*", ls="--", color="grey", label="Total (class)", lw=2, ms=8)
ax.set_xticks(yc.year)
ax.set_xlabel("Year of initial FDA receipt")
ax.set_ylabel("Pancreatitis reports (n)")
ax.grid(alpha=0.3)
ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
fig.tight_layout()
fig.savefig(C.FIG_DIR / "Fig2_yearly_pancreatitis_counts.png", dpi=C.FIG_DPI)
plt.close(fig)

# ---------------------------------------------------------------- Supplementary: sensitivity forest plot
labels = ["Main (any role)", "Suspect only", "vs DPP-4i"]
cols = ["#333333", "#1f77b4", "#d62728"]
fig, ax = plt.subplots(figsize=(7.2, 5))
for j, (df, lab, col) in enumerate(zip([p, s, a], labels, cols)):
    d = df.set_index("exposure").reindex(agents + ["GLP-1 RA class"])
    y = np.arange(len(d))[::-1] + (1 - j) * 0.25
    ok = d.ROR.notna().values
    ax.errorbar(d.ROR.values[ok], y[ok], xerr=[(d.ROR - d.ROR_lower).values[ok], (d.ROR_upper - d.ROR).values[ok]],
                fmt="o", color=col, capsize=2, ms=4, label=lab)
ax.axvline(1, ls="--", color="grey", lw=1)
ax.set_xscale("log")
ax.set_yticks(np.arange(len(agents) + 1)[::-1])
ax.set_yticklabels([x.title() for x in agents] + ["GLP-1 RA class"])
ax.set_xlabel("ROR for pancreatitis (95% CI, log scale)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(C.FIG_DIR / "FigS_forest_sensitivity.png", dpi=C.FIG_DPI)
plt.close(fig)

print(fmt(p).to_string(index=False))
print("Done. Next: run 05_machine_learning.py")
