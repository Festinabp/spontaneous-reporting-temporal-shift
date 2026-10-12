"""
01b_ebgm_counts.py

Input for the empirical Bayes (EBGM) analysis, built at the level of individual MedDRA preferred terms (PTs).
Run after 01_extract_faers_ascii.py (it uses the retained reports in work/reports.pkl).

For every retained report the active ingredients are taken from prod_ai (split on "\\"); when prod_ai is
empty the drug name is used. Tokens naming a GLP-1 RA or a DPP-4 inhibitor (generic name, salt form or brand
name, including fixed-dose combinations) are mapped to the canonical agent name, so that, for example,
"SITAGLIPTIN PHOSPHATE" and "JANUMET" both count as SITAGLIPTIN. Each report counts once per ingredient,
once per PT and once per ingredient x PT pair.

Outputs (WORK_DIR)
  ebgm_pt_input.csv      ingredient x PT pairs with N >= 1 (var1, var2, N, E) - used to estimate the prior
  ebgm_group_input.csv   ingredient x outcome (term set) pairs (var1, var2, N, E) for the GLP-1 RAs and
                         DPP-4 inhibitors - posterior EBGM, EB05 and EB95 are computed for these with the prior
  E = (reports with the ingredient) x (reports with the PT or term set) / (all retained reports)

Runtime: roughly 30-60 minutes (one more pass over the REAC and DRUG tables). Each quarter is cached in
WORK_DIR (ebgmq_*.pkl), so the script can be restarted.
"""
import re
import time
import zipfile
import numpy as np
import pandas as pd
import config as C

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.0f}s] {m}", flush=True)


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


AGENT_PAT = {a: pattern(n) for a, n in {**C.GLP1_MAP, **C.DPP4_MAP}.items()}
EVENT_SETS = {k: set(v) for k, v in C.EVENT_TERMS.items()}
TARGET = list(C.GLP1_MAP) + list(C.DPP4_MAP)

reports = pd.read_pickle(C.WORK_DIR / "reports.pkl")[["primaryid", "quarter"]]
N_ALL = len(reports)
zips = {}
for z in C.BASE_DIR.glob("faers_ascii_*.zip"):
    m = re.search(r"(\d{4})q([1-4])", z.name.lower())
    if m:
        zips[f"{m.group(1)}q{m.group(2)}"] = z

ing_tot, pt_tot, grp_parts = pd.Series(dtype="int64"), pd.Series(dtype="int64"), []
grp_n = {ev: 0 for ev in C.OUTCOMES}
ING_ID, PT_ID = {}, {}          # global integer ids for ingredients and PTs
KEY = 1_000_000                 # pair key = ingredient id * KEY + PT id (fewer than 1,000,000 distinct PTs)
pair_parts, pair_tot = [], pd.Series(dtype="int64")


def ids(values, table):
    for v in pd.unique(values):
        if v not in table:
            table[v] = len(table)
    return values.map(table).astype("int64")


def compact():
    global pair_tot, pair_parts
    if pair_parts:
        pair_tot = pd.concat([pair_tot] + pair_parts).groupby(level=0).sum()
        pair_parts = []
for q in sorted(reports["quarter"].unique()):
    cache = C.WORK_DIR / f"ebgmq_{q}.pkl"
    if cache.exists():
        ing_q, pt_q, pair_q, grp_q, grpn_q = pd.read_pickle(cache)
    else:
        keep = set(reports.loc[reports["quarter"] == q, "primaryid"].astype("int64"))
        with zipfile.ZipFile(zips[q]) as zf:
            reac = read_table(zf, find_member(zf, "REAC"), {"primaryid", "pt"})
            drug = read_table(zf, find_member(zf, "DRUG"), {"primaryid", "drugname", "prod_ai"})
        reac["primaryid"] = pd.to_numeric(reac["primaryid"], errors="coerce")
        reac = reac[reac["primaryid"].isin(keep)]
        reac["pt"] = clean(reac["pt"])
        reac = reac[reac["pt"] != ""].drop_duplicates(["primaryid", "pt"])
        drug["primaryid"] = pd.to_numeric(drug["primaryid"], errors="coerce")
        drug = drug[drug["primaryid"].isin(keep)]
        prod, name = clean(drug["prod_ai"]), clean(drug["drugname"])
        ing = prod.where(prod != "", name).str.split("\\", regex=False)
        ing = pd.DataFrame({"primaryid": drug["primaryid"].values, "ingredient": ing.values}).explode("ingredient")
        ing["ingredient"] = ing["ingredient"].fillna("").str.strip(" .,;")
        for a, pat in AGENT_PAT.items():
            ing.loc[ing["ingredient"].str.contains(pat, na=False), "ingredient"] = a
        ing = ing[ing["ingredient"] != ""].drop_duplicates(["primaryid", "ingredient"])
        del drug, prod, name
        ing_q = ing["ingredient"].value_counts()
        pt_q = reac["pt"].value_counts()
        # ingredient x PT pairs (integer codes keep the merge small)
        ing["ic"], ing_lev = pd.factorize(ing["ingredient"])
        reac["pc"], pt_lev = pd.factorize(reac["pt"])
        pr = ing[["primaryid", "ic"]].merge(reac[["primaryid", "pc"]], on="primaryid")
        vc = pr.groupby(["ic", "pc"]).size()
        pair_q = pd.DataFrame({"ingredient": ing_lev[vc.index.get_level_values(0)],
                               "pt": pt_lev[vc.index.get_level_values(1)], "n": vc.values})
        del pr, vc
        # ingredient x outcome term set (one count per report), for the agents of interest
        grp_rows, grpn_q = [], {}
        ing_t = ing[ing["ingredient"].isin(TARGET)]
        for ev, terms in EVENT_SETS.items():
            pos = set(reac.loc[reac["pt"].isin(terms), "primaryid"])
            grpn_q[ev] = len(pos)
            v = ing_t.loc[ing_t["primaryid"].isin(pos), "ingredient"].value_counts()
            grp_rows.append(pd.DataFrame({"ingredient": v.index, "event": ev, "n": v.values}))
        grp_q = pd.concat(grp_rows, ignore_index=True)
        del ing, reac
        pd.to_pickle((ing_q, pt_q, pair_q, grp_q, grpn_q), cache)
    ing_tot = ing_tot.add(ing_q, fill_value=0)
    pt_tot = pt_tot.add(pt_q, fill_value=0)
    key = ids(pair_q["ingredient"], ING_ID).values * KEY + ids(pair_q["pt"], PT_ID).values
    pair_parts.append(pd.Series(pair_q["n"].values, index=key))
    if len(pair_parts) >= 6:
        compact()
    grp_parts.append(grp_q)
    for ev in grp_n:
        grp_n[ev] += grpn_q[ev]
    log(f"{q}: done")

compact()
ing_name = np.array(list(ING_ID), dtype=object)
pt_name = np.array(list(PT_ID), dtype=object)
pairs = pd.DataFrame({"var1": ing_name[pair_tot.index.values // KEY], "var2": pt_name[pair_tot.index.values % KEY],
                      "N": pair_tot.values.astype("int64")})
pairs["E"] = pairs["var1"].map(ing_tot).values * pairs["var2"].map(pt_tot).values / N_ALL
pairs.to_csv(C.WORK_DIR / "ebgm_pt_input.csv", index=False)
grp = pd.concat(grp_parts).groupby(["ingredient", "event"])["n"].sum().astype("int64").rename("N").reset_index()
grp["E"] = grp["ingredient"].map(ing_tot).values * grp["event"].map(grp_n).values / N_ALL
grp.rename(columns={"ingredient": "var1", "event": "var2"})[["var1", "var2", "N", "E"]] \
   .to_csv(C.WORK_DIR / "ebgm_group_input.csv", index=False)
log(f"Ingredient-PT pairs: {len(pairs):,}; ingredients: {len(ing_tot):,}; PTs: {len(pt_tot):,}")
log(grp[grp["event"] == C.PRIMARY_OUTCOME].sort_values("N", ascending=False).to_string(index=False))
log("Done. Next: run 02_disproportionality.py, then 03_run_openebgm.R")
