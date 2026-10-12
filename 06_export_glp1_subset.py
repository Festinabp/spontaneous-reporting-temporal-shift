"""
06_export_glp1_subset.py
Optional. Exports the GLP-1 RA- and DPP-4 inhibitor-exposed reports (one row per report, no free text) to a small compressed CSV
so the machine-learning results can be checked further. Output: work/glp1_reports.csv.gz
"""
import pandas as pd
import config as C

r = pd.read_pickle(C.WORK_DIR / "reports.pkl")
g = r[r["glp1_any"] | r["dpp4"]].copy()
g.to_csv(C.WORK_DIR / "glp1_reports.csv.gz", index=False, compression="gzip")
print(f"Exported {len(g):,} reports to {C.WORK_DIR / 'glp1_reports.csv.gz'}")
