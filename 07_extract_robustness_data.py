"""
07_extract_robustness_data.py

Second pass over the FAERS ASCII zips (retained reports from step 01 only) to collect the extra information
needed for the robustness, time-to-onset and case-description analyses:
  - co-reported reaction terms indicating other common causes of acute pancreatitis
    (gallstone disease, alcohol-related terms, hypertriglyceridaemia, hypercalcaemia)        -> all reports
  - indication of the GLP-1 RA / DPP-4 inhibitor (INDI table, linked by drug sequence)      -> GLP-1 RA / DPP-4i reports
  - therapy start date of each GLP-1 RA (THER table) and event date (DEMO)                  -> GLP-1 RA reports

Outputs (WORK_DIR)
  extra_reports.pkl     one row per GLP-1 RA or DPP-4i report (indication categories by class and by GLP-1 RA,
                        event date)
  extra_tto.pkl         one row per GLP-1 RA report x agent with the earliest therapy start date
  riskfactor_ids.pkl    set of primaryids (all reports) with a co-reported risk-factor term, by category
Outputs (RESULTS_DIR)
  check_riskfactor_terms.csv   every reaction term matched as a risk factor, with counts (please review)
  check_indication_terms.csv   the most frequent indication terms for GLP-1 RAs / DPP-4i and their category

Runtime: roughly 15-40 minutes. Each quarter is cached, so the script can be restarted.
"""
import re
import sys
import time
import zipfile
import pandas as pd
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


def read_table(zf, member, cols):
    with zf.open(member) as fh:
        df = pd.read_csv(fh, sep="$", dtype=str, encoding="latin-1", quoting=3,
                         usecols=lambda c: c.strip().lower() in cols,
                         on_bad_lines="skip", keep_default_na=False, na_values=[""], engine="c")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def clean(s):
    return s.fillna("").str.upper().str.replace(r"\s+", " ", regex=True).str.strip(" .,;")


def pattern(names):
    return re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b")


GLP1_PAT = {a: pattern(n) for a, n in C.GLP1_MAP.items()}
DPP4_PAT = pattern([n for v in C.DPP4_MAP.values() for n in v])

# Reaction-term patterns for other common causes of acute pancreatitis (matched on upper-case MedDRA PTs).
# All matched terms are written to check_riskfactor_terms.csv so the list can be reviewed.
RISK_PATTERNS = {
    "gallstone": r"CHOLELITHIASIS|CHOLEDOCHOLITHIASIS|GALLSTONE|BILIARY COLIC|BILE DUCT STONE|CALCULUS OF GALLBLADDER",
    "alcohol": r"(?<!NON-)(?<!NON)ALCOHOL",
    "hypertriglyceridaemia": r"HYPERTRIGLYCERIDAEMIA|HYPERTRIGLYCERIDEMIA|BLOOD TRIGLYCERIDES INCREASED",
    "hypercalcaemia": r"HYPERCALCAEMIA|HYPERCALCEMIA|BLOOD CALCIUM INCREASED",
}


# Terms that contain a pattern above but do not indicate the risk factor
# (negative or unspecified test results, maternal or foetal exposure, social drinking and labelling or medication-error
# terms are not counted as evidence of an alcohol-related cause)
RISK_EXCLUDE = (r"INTOLERANCE|ABSTAINS|EX-ALCOHOL|SEPTAL ABLATION|FALSE POSITIVE|FALSE NEGATIVE|FOETAL|FETAL|MATERNAL|"
                r"NEGATIVE|SOCIAL ALCOHOL DRINKER|LABELLED DRUG-ALCOHOL|ALCOHOL TEST$|^BLOOD ALCOHOL$|DECREASED")


def indication_category(pt):
    """Map an indication PT to diabetes (excluding type 1), weight/obesity, unknown, or other."""
    if pt == "" or "UNKNOWN INDICATION" in pt:
        return "unknown"
    if "DIABET" in pt and "TYPE 1" not in pt and "GESTATIONAL" not in pt and "INSIPIDUS" not in pt:
        return "diabetes"
    if re.search(r"OBES|OVERWEIGHT|WEIGHT", pt):
        return "weight"
    return "other"


reports = pd.read_pickle(C.WORK_DIR / "reports.pkl")
keep_cols = ["primaryid", "quarter", "glp1_any", "dpp4"]
reports = reports[keep_cols]
quarters = sorted(reports["quarter"].unique())
zips = {}
for z in C.BASE_DIR.glob("faers_ascii_*.zip"):
    m = re.search(r"(\d{4})q([1-4])", z.name.lower())
    if m:
        zips[f"{m.group(1)}q{m.group(2)}"] = z

rf_ids = {k: set() for k in RISK_PATTERNS}
rf_terms = {}
ind_terms = {}
extra_parts, tto_parts = [], []

for q in quarters:
    cache = C.WORK_DIR / f"extra_{q}.pkl"
    rq = reports[reports["quarter"] == q]
    all_ids = set(rq["primaryid"].astype("int64"))
    target = set(rq.loc[rq["glp1_any"] | rq["dpp4"], "primaryid"].astype("int64"))
    if cache.exists():
        rf_q, rft_q, indt_q, ex_q, tto_q = pd.read_pickle(cache)
    else:
        with zipfile.ZipFile(zips[q]) as zf:
            # ---- risk-factor reaction terms (all retained reports)
            reac = read_table(zf, find_member(zf, "REAC"), {"primaryid", "pt"})
            reac["primaryid"] = pd.to_numeric(reac["primaryid"], errors="coerce")
            reac = reac[reac["primaryid"].isin(all_ids)]
            reac["pt"] = clean(reac["pt"])
            rf_q, rft_q = {}, {}
            for k, pat in RISK_PATTERNS.items():
                hit = reac[reac["pt"].str.contains(pat, regex=True) & ~reac["pt"].str.contains(RISK_EXCLUDE, regex=True)]
                rf_q[k] = set(hit["primaryid"].astype("int64"))
                for t, n in hit["pt"].value_counts().items():
                    rft_q[(k, t)] = rft_q.get((k, t), 0) + int(n)
            del reac

            # ---- GLP-1 RA / DPP-4i drug rows (target reports only)
            drug = read_table(zf, find_member(zf, "DRUG"), {"primaryid", "drug_seq", "drugname", "prod_ai"})
            drug["primaryid"] = pd.to_numeric(drug["primaryid"], errors="coerce")
            drug = drug[drug["primaryid"].isin(target)].copy()
            text = clean(drug["drugname"]) + " | " + clean(drug["prod_ai"])
            drug["agent"] = pd.NA
            for a, pat in GLP1_PAT.items():
                drug.loc[text.str.contains(pat) & drug["agent"].isna(), "agent"] = a
            drug.loc[text.str.contains(DPP4_PAT) & drug["agent"].isna(), "agent"] = "DPP4"
            drug = drug[drug["agent"].notna()][["primaryid", "drug_seq", "agent"]]

            # ---- indications for those drug rows
            indi = read_table(zf, find_member(zf, "INDI"), {"primaryid", "indi_drug_seq", "indi_pt"})
            indi["primaryid"] = pd.to_numeric(indi["primaryid"], errors="coerce")
            indi = indi[indi["primaryid"].isin(target)]
            indi["indi_pt"] = clean(indi["indi_pt"])
            ind = drug.merge(indi.rename(columns={"indi_drug_seq": "drug_seq"}), on=["primaryid", "drug_seq"], how="left")
            ind["indi_pt"] = ind["indi_pt"].fillna("")
            ind["cat"] = ind["indi_pt"].map(indication_category)
            ind["cls"] = ind["agent"].where(ind["agent"] == "DPP4", "GLP1")
            indt_q = ind.groupby(["cls", "indi_pt", "cat"]).size().to_dict()
            ex_q = pd.DataFrame(index=pd.Index(sorted(target), name="primaryid"))
            for cls in ["GLP1", "DPP4"]:
                for cat in ["diabetes", "weight", "other", "unknown"]:
                    ex_q[f"{cls.lower()}_ind_{cat}"] = ind[(ind["cls"] == cls) & (ind["cat"] == cat)].groupby("primaryid").size().gt(0)
            # indication recorded for each GLP-1 RA's own drug rows
            for a in C.GLP1_AGENTS:
                for cat in ["diabetes", "weight"]:
                    ex_q[f"{a}_ind_{cat}"] = ind[(ind["agent"] == a) & (ind["cat"] == cat)].groupby("primaryid").size().gt(0)

            # ---- therapy start dates for GLP-1 RA rows
            ther = read_table(zf, find_member(zf, "THER"), {"primaryid", "dsg_drug_seq", "start_dt"})
            ther["primaryid"] = pd.to_numeric(ther["primaryid"], errors="coerce")
            ther = ther[ther["primaryid"].isin(target)]
            t = drug[drug["agent"] != "DPP4"].merge(ther.rename(columns={"dsg_drug_seq": "drug_seq"}), on=["primaryid", "drug_seq"], how="inner")
            t = t[t["start_dt"].fillna("").str.fullmatch(r"\d{8}")]
            tto_q = t.groupby(["primaryid", "agent"])["start_dt"].min().reset_index()

            # ---- event date
            demo = read_table(zf, find_member(zf, "DEMO"), {"primaryid", "event_dt"})
            demo["primaryid"] = pd.to_numeric(demo["primaryid"], errors="coerce")
            demo = demo[demo["primaryid"].isin(target)].drop_duplicates("primaryid").set_index("primaryid")
            ex_q["event_dt"] = demo["event_dt"]
            ex_q = ex_q.reset_index()
        pd.to_pickle((rf_q, rft_q, indt_q, ex_q, tto_q), cache)

    for k in rf_ids:
        rf_ids[k] |= rf_q[k]
    for key, n in rft_q.items():
        rf_terms[key] = rf_terms.get(key, 0) + n
    for key, n in indt_q.items():
        ind_terms[key] = ind_terms.get(key, 0) + n
    extra_parts.append(ex_q)
    tto_parts.append(tto_q)
    log(f"{q}: done ({len(target):,} GLP-1 RA/DPP-4i reports)")

extra = pd.concat(extra_parts, ignore_index=True)
flag_cols = [c for c in extra.columns if "_ind_" in c]
extra[flag_cols] = extra[flag_cols].fillna(False).astype(bool)
extra.to_pickle(C.WORK_DIR / "extra_reports.pkl")
pd.concat(tto_parts, ignore_index=True).to_pickle(C.WORK_DIR / "extra_tto.pkl")
pd.to_pickle(rf_ids, C.WORK_DIR / "riskfactor_ids.pkl")

pd.DataFrame([(k, t, n) for (k, t), n in rf_terms.items()], columns=["category", "reaction_term", "n_rows"]) \
  .sort_values(["category", "n_rows"], ascending=[True, False]).to_csv(C.RESULTS_DIR / "check_riskfactor_terms.csv", index=False)
it = pd.DataFrame([(c, t, k, n) for (c, t, k), n in ind_terms.items()], columns=["class", "indication_term", "category", "n_rows"])
it.sort_values(["class", "n_rows"], ascending=[True, False]).groupby("class").head(100) \
  .to_csv(C.RESULTS_DIR / "check_indication_terms.csv", index=False)
log({k: len(v) for k, v in rf_ids.items()})
log("Done. Next: run 08_robustness_tto_cases.py")
