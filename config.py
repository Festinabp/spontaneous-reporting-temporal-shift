"""
config.py - shared settings for the FAERS GLP-1 RA pancreatitis re-analysis (ASCII, 2015-2025).

Edit BASE_DIR if your FAERS ASCII zip files are somewhere else.
All other scripts import this file, so keep it in the same folder as the scripts.
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# Folder that contains faers_ascii_2015q1.zip ... faers_ascii_2025Q4.zip
BASE_DIR = Path(os.environ.get("FAERS_BASE_DIR", Path.home() / "Desktop" / "ASCII2015-2025"))
WORK_DIR = BASE_DIR / "work"          # intermediate files (can be deleted and regenerated)
RESULTS_DIR = BASE_DIR / "results"    # tables (CSV) for the manuscript
FIG_DIR = RESULTS_DIR / "figures"     # figures (PNG, 600 dpi)
for _d in (WORK_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Study period and validation design
# ----------------------------------------------------------------------------
YEAR_START, YEAR_END = 2015, 2025          # by year of initial FDA receipt (init_fda_dt)
RANDOM_STATE = 42

TEMPORAL_TRAIN = (2015, 2022)              # temporal validation: train
TEMPORAL_TEST = (2023, 2025)               # temporal validation: test
RECAL_TRAIN = (2015, 2021)                 # recalibration: base-model training
RECAL_CALIB = (2022, 2022)                 # recalibration: calibration set
RECAL_TEST = (2023, 2025)                  # recalibration: test

MIN_CASES = 3                              # minimum co-reports (a >= 3) for PRR/ROR/IC signals
BOOTSTRAP_N = 200
PERMUTATION_N_REPEATS = 5
PERMUTATION_MAX_ROWS = 50000               # subsample of the test set for permutation importance
FIG_DPI = 600

# Optional: restrict to a few quarters for a quick test run, e.g. "2023q1,2023q2"
TEST_QUARTERS = [q.strip().lower() for q in os.environ.get("FAERS_TEST_QUARTERS", "").split(",") if q.strip()]

# ----------------------------------------------------------------------------
# Drug definitions (matched against prod_ai ingredients and drugname, upper case)
# ----------------------------------------------------------------------------
GLP1_MAP = {
    "EXENATIDE":    ["EXENATIDE", "BYETTA", "BYDUREON"],
    "LIXISENATIDE": ["LIXISENATIDE", "ADLYXIN", "LYXUMIA", "SOLIQUA", "SULIQUA"],
    "LIRAGLUTIDE":  ["LIRAGLUTIDE", "VICTOZA", "SAXENDA", "XULTOPHY"],
    "ALBIGLUTIDE":  ["ALBIGLUTIDE", "TANZEUM", "EPERZAN"],
    "SEMAGLUTIDE":  ["SEMAGLUTIDE", "OZEMPIC", "WEGOVY", "RYBELSUS"],
    "DULAGLUTIDE":  ["DULAGLUTIDE", "TRULICITY"],
    "TIRZEPATIDE":  ["TIRZEPATIDE", "MOUNJARO", "ZEPBOUND"],
}
GLP1_AGENTS = list(GLP1_MAP.keys())

# Active comparator: DPP-4 inhibitors (including fixed-dose combinations)
DPP4_MAP = {
    "SITAGLIPTIN": ["SITAGLIPTIN", "JANUVIA", "JANUMET", "STEGLUJAN", "ZITUVIO"],
    "SAXAGLIPTIN": ["SAXAGLIPTIN", "ONGLYZA", "KOMBIGLYZE", "QTERN"],
    "LINAGLIPTIN": ["LINAGLIPTIN", "TRADJENTA", "TRAJENTA", "JENTADUETO", "GLYXAMBI", "TRIJARDY"],
    "ALOGLIPTIN":  ["ALOGLIPTIN", "NESINA", "KAZANO", "OSENI", "VIPIDIA"],
    "VILDAGLIPTIN": ["VILDAGLIPTIN", "GALVUS", "EUCREAS"],
}

# Second active comparator: SGLT2 inhibitors (including fixed-dose combinations)
SGLT2_MAP = {
    "CANAGLIFLOZIN": ["CANAGLIFLOZIN", "INVOKANA", "INVOKAMET", "VOKANAMET"],
    "DAPAGLIFLOZIN": ["DAPAGLIFLOZIN", "FARXIGA", "FORXIGA", "XIGDUO", "QTERN", "QTERNMET"],
    "EMPAGLIFLOZIN": ["EMPAGLIFLOZIN", "JARDIANCE", "SYNJARDY", "GLYXAMBI", "TRIJARDY"],
    "ERTUGLIFLOZIN": ["ERTUGLIFLOZIN", "STEGLATRO", "SEGLUROMET", "STEGLUJAN"],
    "BEXAGLIFLOZIN": ["BEXAGLIFLOZIN", "BRENZAVVY"],
}

SUSPECT_ROLES = {"PS", "SS"}   # primary suspect, secondary suspect

# ----------------------------------------------------------------------------
# Event definitions (MedDRA preferred terms, upper case; MedDRA 29.1)
# ----------------------------------------------------------------------------
EVENT_TERMS = {
    "pancreatitis": [
        "CULLEN'S SIGN", "GREY TURNER'S SIGN", "HAEMORRHAGIC NECROTIC PANCREATITIS",
        "HEREDITARY PANCREATITIS", "IDIOPATHIC PANCREATITIS", "IMMUNE-MEDIATED PANCREATITIS",
        "ISCHAEMIC PANCREATITIS", "OEDEMATOUS PANCREATITIS", "PANCREATIC ABSCESS", "PANCREATIC NECROSIS",
        "PANCREATIC CYST DRAINAGE", "PANCREATIC HAEMORRHAGE", "PANCREATIC PHLEGMON",
        "PANCREATIC PSEUDOANEURYSM", "PANCREATIC PSEUDOCYST", "PANCREATIC PSEUDOCYST DRAINAGE",
        "PANCREATIC PSEUDOCYST HAEMORRHAGE", "PANCREATIC PSEUDOCYST RUPTURE", "PANCREATITIS",
        "PANCREATITIS ACUTE", "PANCREATITIS HAEMORRHAGIC", "PANCREATITIS NECROTISING",
        "PANCREATITIS RELAPSING", "PANCREATORENAL SYNDROME", "SUBACUTE PANCREATITIS",
        "WALLED-OFF PANCREATIC NECROSIS",
    ],
    "gallbladder_biliary": [
        "CHOLELITHIASIS", "CHOLECYSTITIS", "CHOLECYSTITIS ACUTE", "CHOLECYSTITIS CHRONIC",
        "CHOLECYSTITIS INFECTIVE", "CHOLECYSTITIS OBSTRUCTIVE", "EMPHYSEMATOUS CHOLECYSTITIS",
        "BILE DUCT STONE", "BILIARY COLIC",
    ],
}
OUTCOMES = list(EVENT_TERMS.keys())
PRIMARY_OUTCOME = "pancreatitis"
OUTCOME_LABELS = {"pancreatitis": "Pancreatitis", "gallbladder_biliary": "Gallbladder/biliary"}

# FAERS seriousness outcome codes (OUTC table)
OUTC_CODES = {"DE": "death", "LT": "lifethreatening", "HO": "hospitalization",
              "DS": "disability", "CA": "congenital_anomaly", "RI": "required_intervention", "OT": "other_serious"}

AGENT_COLORS = {"EXENATIDE": "#1f77b4", "LIXISENATIDE": "#ff7f0e", "LIRAGLUTIDE": "#2ca02c",
                "ALBIGLUTIDE": "#d62728", "SEMAGLUTIDE": "#9467bd", "DULAGLUTIDE": "#8c564b",
                "TIRZEPATIDE": "#e377c2"}
