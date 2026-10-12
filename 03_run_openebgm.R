# 03_run_openebgm.R
#
# Empirical Bayes geometric mean (EBGM) with the openEBGM package (DuMouchel gamma-Poisson shrinker).
# Inputs (written by 01b_ebgm_counts.py):
#   work/ebgm_pt_input.csv     active ingredient x MedDRA preferred term pairs (var1, var2, N, E), N >= 1
#   work/ebgm_group_input.csv  active ingredient x outcome term set pairs for the GLP-1 RAs and DPP-4 inhibitors
# The prior (hyperparameters) is estimated from all ingredient x preferred term pairs in the database; posterior
# EBGM, EB05 and EB95 are then computed with this prior for each agent and outcome term set.
# Outputs: results/ebgm_results.csv, results/ebgm_hyperparameters.csv
#
# Run in RStudio or in Terminal:  Rscript 03_run_openebgm.R
# First time only:  install.packages("remotes"); remotes::install_version("openEBGM", version = "0.10.0")

library(openEBGM)

base_dir <- Sys.getenv("FAERS_BASE_DIR", file.path(path.expand("~"), "Desktop", "ASCII2015-2025"))
work_dir <- file.path(base_dir, "work")
res_dir  <- file.path(base_dir, "results")

# data.table reads the large pair file much faster than read.csv (falls back to read.csv if not installed)
pt_file <- file.path(work_dir, "ebgm_pt_input.csv")
dat <- if (requireNamespace("data.table", quietly = TRUE)) as.data.frame(data.table::fread(pt_file)) else
  read.csv(pt_file, stringsAsFactors = FALSE)
dat <- dat[dat$N >= 1 & dat$E > 0, ]
cat("Ingredient-preferred term pairs:", nrow(dat), "\n")

# Data squashing (N = 1 and N = 2) to make hyperparameter estimation feasible
sq <- squashData(dat)
sq <- squashData(sq, count = 2, bin_size = 10, keep_pts = 50)

# Hyperparameter estimation; several starting points, first one that converges is used
theta_init <- data.frame(
  alpha1 = c(0.2, 0.1, 0.3, 0.5, 0.2),
  beta1  = c(0.1, 0.1, 0.5, 0.3, 0.2),
  alpha2 = c(2,   10,  6,   12,  5),
  beta2  = c(4,   10,  6,   12,  5),
  p      = c(1/3, 0.2, 0.5, 0.8, 0.4)
)
set.seed(42)
hyper <- autoHyper(sq, theta_init = theta_init, squashed = TRUE, zeroes = FALSE,
                   N_star = 1, max_pts = max(20000, nrow(sq)))
theta <- hyper$estimates
print(hyper)
write.csv(data.frame(parameter = c(names(theta), "n_pairs"), estimate = c(as.numeric(theta), nrow(dat))),
          file.path(res_dir, "ebgm_hyperparameters.csv"), row.names = FALSE)

# Posterior summaries for the drugs of interest and the outcome term sets
glp1 <- c("EXENATIDE", "LIXISENATIDE", "LIRAGLUTIDE", "ALBIGLUTIDE", "SEMAGLUTIDE", "DULAGLUTIDE", "TIRZEPATIDE")
dpp4 <- c("SITAGLIPTIN", "SAXAGLIPTIN", "LINAGLIPTIN", "ALOGLIPTIN", "VILDAGLIPTIN")
sub <- read.csv(file.path(work_dir, "ebgm_group_input.csv"), stringsAsFactors = FALSE)
sub <- sub[sub$var1 %in% c(glp1, dpp4) & sub$N >= 1 & sub$E > 0, ]
qn  <- Qn(theta, N = sub$N, E = sub$E)
sub$EBGM <- ebgm(theta, N = sub$N, E = sub$E, qn = qn)
sub$EB05 <- quantBisect(5,  theta_hat = theta, N = sub$N, E = sub$E, qn = qn)
sub$EB95 <- quantBisect(95, theta_hat = theta, N = sub$N, E = sub$E, qn = qn)
sub$EB05_signal <- sub$EB05 > 1
names(sub)[names(sub) == "var1"] <- "exposure"
names(sub)[names(sub) == "var2"] <- "event"
write.csv(sub, file.path(res_dir, "ebgm_results.csv"), row.names = FALSE)
print(sub[sub$event == "pancreatitis", ])

cat("\nR version: ", R.version.string, "\n")
cat("openEBGM version: ", as.character(packageVersion("openEBGM")), "\n")
cat("Done. Next: run 04_tables_figures_dpa.py\n")
