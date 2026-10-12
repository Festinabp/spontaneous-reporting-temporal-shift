"""
11_figure_comparator.py

Figure for the active-comparator results (run after 09):
  (a) proportion of reports including pancreatitis by year, GLP-1 RA vs DPP-4 inhibitor reports
  (b) GLP-1 RA class RORs for pancreatitis, crude, adjusted for year (primary) and fully adjusted, against all other
      reports, DPP-4 inhibitors (all, health-professional reports, recorded diabetes indication) and, if 09b has been
      run, SGLT2 inhibitors. Year-only estimates for the DPP-4 inhibitor subgroups come from 09d (run 09d before 11).
Outputs: results/Comparator_by_year.csv, figures/Fig_comparator_positive_control.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
import config as C

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
_csv = os.environ.get("FAERS_ML_INPUT")
r = pd.read_csv(_csv, low_memory=False, usecols=["glp1_any", "dpp4", "report_year", "pancreatitis"]) if _csv \
    else pd.read_pickle(C.WORK_DIR / "reports.pkl")[["glp1_any", "dpp4", "report_year", "pancreatitis"]]
g = r[r["glp1_any"] & ~r["dpp4"]]
d = r[r["dpp4"] & ~r["glp1_any"]]
tab = pd.DataFrame({
    "GLP-1 RA reports (n)": g.groupby("report_year").size(),
    "GLP-1 RA with pancreatitis (%)": g.groupby("report_year")["pancreatitis"].mean() * 100,
    "DPP-4i reports (n)": d.groupby("report_year").size(),
    "DPP-4i with pancreatitis (%)": d.groupby("report_year")["pancreatitis"].mean() * 100,
})
tab.round(2).to_csv(C.RESULTS_DIR / "Comparator_by_year.csv")

# panel b: GLP-1 RA class pancreatitis RORs, crude, year-adjusted (primary) and fully adjusted, under each comparison
pc = pd.read_csv(C.RESULTS_DIR / "Positive_control.csv")
ac = pd.read_csv(C.RESULTS_DIR / "Adjusted_comparator.csv")
cls = lambda df: df[(df.exposure == "GLP-1 RA class") & (df.event == "pancreatitis")].set_index("analysis")
pc, ac = cls(pc), cls(ac)
yo = pd.read_csv(C.RESULTS_DIR / "Year_only_subgroups_dpp4.csv")       # from 09d
yo = yo[yo.exposure == "GLP-1 RA class"].set_index("analysis")
rows_b = [("All other reports", pc.loc["1 Main analysis (all reports)"], None, None)]
for an, lab in [("vs DPP-4i (all)", "DPP-4i"), ("vs DPP-4i, health-professional reports", "DPP-4i,\nhealth professionals"),
                ("vs DPP-4i, recorded diabetes indication", "DPP-4i, recorded\ndiabetes indication")]:
    r_, y_ = ac.loc[an], yo.loc[an]
    rows_b.append((lab, r_, (y_["year_ROR"], y_["year_lower"], y_["year_upper"]), (r_["aROR"], r_["a_lower"], r_["a_upper"])))
sg_path = C.RESULTS_DIR / "SGLT2_comparator.csv"
if sg_path.exists():   # second active comparator (SGLT2 inhibitors), if 09b_sglt2_comparator.py has been run
    sg = cls(pd.read_csv(sg_path))
    for an, lab in [("vs SGLT2i (all)", "SGLT2i"), ("vs SGLT2i, health-professional reports", "SGLT2i,\nhealth professionals")]:
        r_ = sg.loc[an]
        rows_b.append((lab, r_, (r_["year_ROR"], r_["year_lower"], r_["year_upper"]), (r_["adj_ROR"], r_["adj_lower"], r_["adj_upper"])))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
a1.plot(tab.index, tab["GLP-1 RA with pancreatitis (%)"], "-o", color="#d62728", ms=4, label="GLP-1 RA, % with pancreatitis")
a1.plot(tab.index, tab["DPP-4i with pancreatitis (%)"], "-s", color="#555555", ms=4, label="DPP-4 inhibitor, % with pancreatitis")
sy_path = C.RESULTS_DIR / "SGLT2_by_year.csv"
if sy_path.exists():
    sy = pd.read_csv(sy_path, index_col=0)
    a1.plot(sy.index, sy["SGLT2i with pancreatitis (%)"], "--^", color="#2ca02c", ms=4, label="SGLT2 inhibitor, % with pancreatitis")
ax_n = a1.twinx()
if sy_path.exists():   # three bars per year: GLP-1 RA, DPP-4 inhibitor, SGLT2 inhibitor
    w, off = 0.28, [-0.28, 0, 0.28]
else:
    w, off = 0.4, [-0.2, 0.2, None]
ax_n.bar(tab.index + off[0], tab["GLP-1 RA reports (n)"] / 1000, width=w, color="#d62728", alpha=0.15, label="GLP-1 RA reports (n)")
ax_n.bar(tab.index + off[1], tab["DPP-4i reports (n)"] / 1000, width=w, color="#555555", alpha=0.15, label="DPP-4 inhibitor reports (n)")
if sy_path.exists():
    ax_n.bar(sy.index + off[2], sy["SGLT2i reports (n)"] / 1000, width=w, color="#2ca02c", alpha=0.18, label="SGLT2 inhibitor reports (n)")
ax_n.set_ylabel("Reports (thousands, bars)")
a1.set_zorder(ax_n.get_zorder() + 1)
a1.patch.set_visible(False)
a1.set(xlabel="Year of initial FDA receipt", ylabel="Reports including pancreatitis (%)", title="a")
a1.set_xticks(tab.index)
a1.set_xticklabels([str(y) for y in tab.index], rotation=45, ha="right")
h1, l1 = a1.get_legend_handles_labels()
h2, l2 = ax_n.get_legend_handles_labels()
a1.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
a1.grid(alpha=0.3)

y_b = np.arange(len(rows_b))[::-1]
for i, (lab, r_, yadj, adj) in zip(y_b, rows_b):
    a2.errorbar(r_["ROR"], i + 0.22, xerr=[[r_["ROR"] - r_["lower"]], [r_["upper"] - r_["ROR"]]], fmt="o", color="#d62728",
                capsize=2, ms=5, label="Crude" if i == y_b[0] else None)
    if yadj is not None:
        a2.errorbar(yadj[0], i, xerr=[[yadj[0] - yadj[1]], [yadj[2] - yadj[0]]], fmt="o", mfc="white", color="#d62728",
                    capsize=2, ms=5, label="Adjusted for year (primary)" if i == y_b[1] else None)
    if adj is not None:
        a2.errorbar(adj[0], i - 0.22, xerr=[[adj[0] - adj[1]], [adj[2] - adj[0]]], fmt="D", mfc="white", color="#d62728",
                    capsize=2, ms=5, label="Fully adjusted" if i == y_b[1] else None)
a2.axvline(1, ls="--", color="grey", lw=0.8)
a2.set_xscale("log")
a2.set_yticks(y_b)
a2.set_yticklabels([lab for lab, *_ in rows_b], fontsize=8)
a2.set(xlabel="GLP-1 RA class ROR for pancreatitis (95% CI, log scale)", title="b")
a2.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
a2.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4, 8]))
a2.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2", "4", "8"]))
a2.xaxis.set_minor_locator(NullLocator())
a2.set_xlim(0.5, 10)
a2.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(C.FIG_DIR / "Fig_comparator_positive_control.png", dpi=C.FIG_DPI)
plt.close(fig)
print(tab.round(2).to_string())
print("Done.")
