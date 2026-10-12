"""
01_extract_faers_ascii.py

Reads the FAERS quarterly ASCII zip files (2015Q1-2025Q4) directly from the zips (no need to unzip),
applies FDA-recommended case-level deduplication, removes deleted cases, and builds one report-level
analysis table.

Steps
  Pass 1  Read DEMO (+ deleted-case lists) from every quarter.
          Remove deleted caseids; keep the latest version of each case (latest fda_dt, then highest
          primaryid); keep cases whose initial FDA receipt year (init_fda_dt) is 2015-2025.
  Pass 2  For each quarter, read DRUG, REAC and OUTC for the retained reports only and summarise them
          to one row per report (exposure flags, outcome flags, seriousness, counts).
          (The EBGM input is built separately by 01b_ebgm_counts.py.)

Outputs (WORK_DIR)
  reports.pkl                 one row per retained report (analysis table)
Outputs (RESULTS_DIR)
  data_flow.csv               record counts at each step (for the Methods / READUS-PV item 5b)
  quarter_inventory.csv       rows read per quarter (check that all 44 quarters are present)

Runtime: roughly 30-90 minutes for 44 quarters, depending on the computer. Each quarter is cached in
WORK_DIR, so if the script stops you can run it again and it continues where it left off.
"""
import re
import sys
import time
import zipfile
import numpy as np
import pandas as pd

import config as C

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def list_zips():
    zips = sorted(C.BASE_DIR.glob("faers_ascii_*.zip"), key=lambda p: p.name.lower())
    out = []
    for z in zips:
        m = re.search(r"(\d{4})q([1-4])", z.name.lower())
        if not m:
            continue
        q = f"{m.group(1)}q{m.group(2)}"
        if C.TEST_QUARTERS and q not in C.TEST_QUARTERS:
            continue
        out.append((q, z))
    if not out:
        sys.exit(f"No faers_ascii_*.zip files found in {C.BASE_DIR}")
    return out


def find_member(zf, table):
    """Return the zip member name for a FAERS table (DEMO, DRUG, REAC, OUTC, ...)."""
    pat = re.compile(rf"(^|/){table}\d{{2}}q[1-4](_new)?\.txt$", re.I)
    hits = [n for n in zf.namelist() if pat.search(n)]
    if len(hits) != 1:
        raise RuntimeError(f"{zf.filename}: expected one {table} file, found {hits}")
    return hits[0]


def read_table(zf, member, cols):
    """Read a $-delimited FAERS table, keeping only the requested columns (all as text)."""
    with zf.open(member) as fh:
        df = pd.read_csv(fh, sep="$", dtype=str, encoding="latin-1", quoting=3,
                         usecols=lambda c: c.strip().lower() in cols,
                         on_bad_lines="skip", keep_default_na=False, na_values=[""], engine="c")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def read_deleted(zf):
    ids = set()
    for n in zf.namelist():
        if re.search(r"delete", n, re.I) and n.lower().endswith(".txt"):
            with zf.open(n) as fh:
                for line in fh.read().decode("latin-1").splitlines():
                    s = line.strip().strip("$")
                    if s.isdigit():
                        ids.add(int(s))
    return ids


def to_int(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def name_pattern(names):
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b")


GLP1_PATTERNS = {a: name_pattern(n) for a, n in C.GLP1_MAP.items()}
DPP4_PATTERN = name_pattern([n for v in C.DPP4_MAP.values() for n in v])
EVENT_SETS = {k: set(v) for k, v in C.EVENT_TERMS.items()}
AGE_FACTORS = {"YR": 1.0, "DEC": 10.0, "MON": 1 / 12, "WK": 1 / 52.1775, "DY": 1 / 365.25, "HR": 1 / 8766}


def clean_text(s):
    return (s.fillna("").str.upper().str.replace(r"\s+", " ", regex=True).str.strip(" .,;"))


# ----------------------------------------------------------------------------
# Pass 1: DEMO and deleted cases
# ----------------------------------------------------------------------------
DEMO_COLS = {"primaryid", "caseid", "caseversion", "init_fda_dt", "fda_dt", "rept_cod", "age", "age_cod",
             "sex", "wt", "wt_cod", "occp_cod", "reporter_country", "occr_country"}

def compact_demo(d, q):
    """Convert DEMO columns to compact types so that 44 quarters fit in memory."""
    out = pd.DataFrame({
        "primaryid": to_int(d["primaryid"]),
        "caseid": to_int(d["caseid"]),
        "fda_dt_n": to_int(d["fda_dt"]),
        "init_fda_dt_n": to_int(d["init_fda_dt"]),
        "age": pd.to_numeric(d["age"], errors="coerce").astype("float32"),
        "wt": pd.to_numeric(d["wt"], errors="coerce").astype("float32"),
    })
    for c in ["age_cod", "sex", "wt_cod", "occp_cod", "rept_cod", "reporter_country", "occr_country"]:
        out[c] = d[c].str.upper().str.strip().astype("category")
    out["quarter"] = q
    out["quarter"] = out["quarter"].astype("category")
    return out


zips = list_zips()
log(f"Found {len(zips)} quarterly zip files in {C.BASE_DIR}")

inventory, deleted_all, demo_parts = [], set(), []
for q, zpath in zips:
    cache = C.WORK_DIR / f"demo_{q}.pkl"
    with zipfile.ZipFile(zpath) as zf:
        deleted = read_deleted(zf)
        deleted_all |= deleted
        if cache.exists():
            demo = pd.read_pickle(cache)
        else:
            demo = read_table(zf, find_member(zf, "DEMO"), DEMO_COLS)
            demo = compact_demo(demo, q)
            demo.to_pickle(cache)
    inventory.append({"quarter": q, "demo_rows": len(demo), "deleted_caseids_listed": len(deleted)})
    demo_parts.append(demo)
    log(f"Pass 1 {q}: DEMO rows {len(demo):,}; deleted caseids listed {len(deleted):,}")

for c in ["age_cod", "sex", "wt_cod", "occp_cod", "rept_cod", "reporter_country", "occr_country", "quarter"]:
    cats = sorted(set().union(*[set(p[c].cat.categories) for p in demo_parts]))
    for p in demo_parts:
        p[c] = p[c].cat.set_categories(cats)
demo = pd.concat(demo_parts, ignore_index=True)
del demo_parts
flow = [("DEMO rows read (all quarters)", len(demo))]
demo = demo.dropna(subset=["primaryid", "caseid"])

# identical primaryid appearing in more than one quarter file: keep the most recent quarter
demo = demo.sort_values(["primaryid", "quarter"]).drop_duplicates("primaryid", keep="last")
flow.append(("Unique report versions (primaryid)", len(demo)))

demo = demo[~demo["caseid"].isin(deleted_all)]
flow.append(("After removing FDA-deleted cases", len(demo)))

# FDA-recommended deduplication: one row per caseid, latest fda_dt, then highest primaryid
demo = (demo.sort_values(["caseid", "fda_dt_n", "primaryid"], ascending=[True, False, False])
            .drop_duplicates("caseid", keep="first"))
flow.append(("After case-level deduplication (latest version per caseid)", len(demo)))

init = demo["init_fda_dt_n"].fillna(demo["fda_dt_n"])
demo["report_year"] = (init // 10000).astype("Int64")
demo = demo[demo["report_year"].between(C.YEAR_START, C.YEAR_END)].copy()
flow.append((f"Initial FDA receipt {C.YEAR_START}-{C.YEAR_END} (analytic dataset)", len(demo)))
log(f"Retained reports: {len(demo):,}")

# Derived demographic variables
for c in ["age_cod", "sex", "wt_cod", "occp_cod", "rept_cod", "reporter_country", "occr_country", "quarter"]:
    demo[c] = demo[c].astype(object)
age = demo["age"].astype(float) * demo["age_cod"].map(AGE_FACTORS).astype(float)
demo["age_years"] = age.where(age.between(0, 120))
wt = demo["wt"].astype(float)
wt = np.where(demo["wt_cod"].eq("LBS"), wt * 0.453592, np.where(demo["wt_cod"].eq("KG"), wt, np.nan))
demo["weight_kg"] = pd.Series(wt, index=demo.index).where(lambda s: s.between(0.5, 400))
demo["sex"] = demo["sex"].where(demo["sex"].isin(["F", "M"]))
demo["reporter_type"] = demo["occp_cod"].where(demo["occp_cod"].isin(["MD", "PH", "OT", "LW", "CN", "HP"]))
demo["country"] = demo["reporter_country"].fillna(demo["occr_country"])
demo["report_type"] = demo["rept_cod"]
demo = demo[["primaryid", "caseid", "quarter", "report_year", "fda_dt_n", "init_fda_dt_n", "age_years",
             "weight_kg", "sex", "reporter_type", "country", "report_type"]]

# ----------------------------------------------------------------------------
# Pass 2: DRUG, REAC, OUTC summarised per report
# ----------------------------------------------------------------------------
report_parts = []

for (q, zpath), inv in zip(zips, inventory):
    cache = C.WORK_DIR / f"reports_{q}.pkl"
    keep = demo.loc[demo["quarter"] == q, "primaryid"]
    keep_set = set(keep.astype("int64"))
    if cache.exists():
        rep = pd.read_pickle(cache)
    else:
        with zipfile.ZipFile(zpath) as zf:
            # ---- REAC
            reac = read_table(zf, find_member(zf, "REAC"), {"primaryid", "pt"})
            inv["reac_rows"] = len(reac)
            reac["primaryid"] = pd.to_numeric(reac["primaryid"], errors="coerce")
            reac = reac[reac["primaryid"].isin(keep_set)]
            reac["pt"] = clean_text(reac["pt"])
            reac = reac[reac["pt"] != ""].drop_duplicates()
            rep = pd.DataFrame(index=pd.Index(sorted(keep_set), name="primaryid"))
            rep["n_reactions"] = reac.groupby("primaryid").size()
            for ev, terms in EVENT_SETS.items():
                hit = reac[reac["pt"].isin(terms)].groupby("primaryid").size()
                rep[f"n_terms_{ev}"] = hit
                rep[ev] = hit.gt(0)
            del reac

            # ---- OUTC
            outc = read_table(zf, find_member(zf, "OUTC"), {"primaryid", "outc_cod"})
            outc["primaryid"] = pd.to_numeric(outc["primaryid"], errors="coerce")
            outc = outc[outc["primaryid"].isin(keep_set)]
            outc["outc_cod"] = outc["outc_cod"].str.upper().str.strip()
            for code, name in C.OUTC_CODES.items():
                rep[f"serious_{name}"] = outc.loc[outc["outc_cod"] == code].groupby("primaryid").size().gt(0)
            del outc

            # ---- DRUG
            drug = read_table(zf, find_member(zf, "DRUG"), {"primaryid", "role_cod", "drugname", "prod_ai"})
            inv["drug_rows"] = len(drug)
            drug["primaryid"] = pd.to_numeric(drug["primaryid"], errors="coerce")
            drug = drug[drug["primaryid"].isin(keep_set)].copy()
            drug["drugname"] = clean_text(drug["drugname"])
            drug["prod_ai"] = clean_text(drug["prod_ai"])
            drug["role_cod"] = drug["role_cod"].str.upper().str.strip()
            drug["suspect"] = drug["role_cod"].isin(C.SUSPECT_ROLES)
            text = drug["drugname"] + " | " + drug["prod_ai"]
            rep["n_drugs"] = drug.groupby("primaryid").size()
            for agent, pat in GLP1_PATTERNS.items():
                m = text.str.contains(pat, na=False)
                rep[f"glp1_{agent}"] = drug.loc[m].groupby("primaryid").size().gt(0)
                rep[f"glp1_{agent}_suspect"] = drug.loc[m & drug["suspect"]].groupby("primaryid").size().gt(0)
            m = text.str.contains(DPP4_PATTERN, na=False)
            rep["dpp4"] = drug.loc[m].groupby("primaryid").size().gt(0)
            rep["dpp4_suspect"] = drug.loc[m & drug["suspect"]].groupby("primaryid").size().gt(0)
            del drug, text

        bool_cols = [c for c in rep.columns if c in C.OUTCOMES or c.startswith(("serious_", "glp1_", "dpp4"))]
        rep[bool_cols] = rep[bool_cols].fillna(False).astype(bool)
        cnt_cols = ["n_reactions", "n_drugs"] + [f"n_terms_{e}" for e in C.OUTCOMES]
        rep[cnt_cols] = rep[cnt_cols].fillna(0).astype("int32")
        rep = rep.reset_index()
        rep.to_pickle(cache)
    report_parts.append(rep)
    log(f"Pass 2 {q}: {len(rep):,} retained reports summarised")

reports = pd.concat(report_parts, ignore_index=True)
reports["primaryid"] = reports["primaryid"].astype("int64")
demo["primaryid"] = demo["primaryid"].astype("int64")
reports = demo.merge(reports, on="primaryid", how="left", validate="one_to_one")
flag_cols = [c for c in reports.columns if c in C.OUTCOMES or c.startswith(("serious_", "glp1_", "dpp4"))]
reports[flag_cols] = reports[flag_cols].fillna(False).astype(bool)
reports["glp1_any"] = reports[[f"glp1_{a}" for a in C.GLP1_AGENTS]].any(axis=1)
reports["glp1_any_suspect"] = reports[[f"glp1_{a}_suspect" for a in C.GLP1_AGENTS]].any(axis=1)
reports["n_glp1_agents"] = reports[[f"glp1_{a}" for a in C.GLP1_AGENTS]].sum(axis=1)
reports.to_pickle(C.WORK_DIR / "reports.pkl")

flow.append(("Reports with any drug record", int((reports["n_drugs"] > 0).sum())))
flow.append(("Reports with any reaction term", int((reports["n_reactions"] > 0).sum())))
flow.append(("GLP-1 RA-exposed reports (any role)", int(reports["glp1_any"].sum())))
flow.append(("GLP-1 RA-exposed reports with pancreatitis", int((reports["glp1_any"] & reports["pancreatitis"]).sum())))
pd.DataFrame(flow, columns=["step", "n"]).to_csv(C.RESULTS_DIR / "data_flow.csv", index=False)
pd.DataFrame(inventory).to_csv(C.RESULTS_DIR / "quarter_inventory.csv", index=False)

log("Data flow:")
for s, n in flow:
    log(f"  {s}: {n:,}")
log("Done. Next: run 01b_ebgm_counts.py")
