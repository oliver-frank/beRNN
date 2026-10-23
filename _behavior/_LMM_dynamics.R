# ==============================================================================
# M3 | Dynamics markers | Gaussian LMM (lmer), random intercepts only
# ==============================================================================
library(tidyverse)
library(lme4)
library(lmerTest)   # load after lme4

# ==============================================================================
# SHARED BLOCK - identical in all four scripts. Do not edit per model.
# ==============================================================================
out_dir     <- "W:\\group_csp\\analyses\\oliver.frank\\Data\\supplement_tables\\M3_dynamics_original"   # all CSVs / PNGs land here
RUN_POSTHOC <- FALSE                 # TRUE = additional Tukey contrasts over the time factor
dir.create(out_dir, showWarnings = FALSE)

summarise_model <- function(m, model_id, response, family_label,
                            bt_label = NA_character_,  # e.g. "Odds ratio"; NA = no back-transformation
                            poisson  = FALSE,          # TRUE adds the DHARMa zero-inflation test
                            sd_ci    = FALSE,          # TRUE adds profile CIs for random-effect SDs (slow)
                            n_sim    = 500) {
  is_glmm <- inherits(m, "glmerMod")
  ng      <- ngrps(m)
  n_obs   <- nobs(m)
  bt      <- !is.na(bt_label)
  
  # ---- Fixed effects ----------------------------------------------------------
  cf     <- as.data.frame(coef(summary(m)))
  has_df <- "df" %in% names(cf)                  # TRUE for lmerTest (Satterthwaite), FALSE for glmer
  zc     <- qnorm(0.975)
  fixed  <- tibble(
    Model = model_id, Response = response, Family = family_label,
    Term = rownames(cf), Estimate = cf$Estimate, SE = cf$`Std. Error`,
    df = if (has_df) cf$df else NA_real_,
    `Statistic type` = if (has_df) "t" else "z",
    Statistic = if (has_df) cf$`t value` else cf$`z value`,
    p = cf[[ncol(cf)]]
  )
  keep <- fixed$Term != "(Intercept)"
  fixed$`p (Holm)` <- NA_real_
  fixed$`p (Holm)`[keep] <- p.adjust(fixed$p[keep], method = "holm")   # within model, intercept excluded
  fixed <- fixed %>% transmute(
    Model, Response, Family, Term, Estimate, SE, df, `Statistic type`, Statistic, p, `p (Holm)`,
    `95% CI lower` = Estimate - zc * SE,                                 # Wald CI
    `95% CI upper` = Estimate + zc * SE,
    `Back-transformed type`       = if (bt) bt_label else NA_character_,
    `Back-transformed effect`     = if (bt) exp(Estimate) else NA_real_,
    `95% CI lower (back-transf.)` = if (bt) exp(Estimate - zc * SE) else NA_real_,
    `95% CI upper (back-transf.)` = if (bt) exp(Estimate + zc * SE) else NA_real_,
    `N obs` = n_obs
  )
  
  # ---- Random effects + model-level information ---------------------------------
  vc    <- as.data.frame(VarCorr(m))
  denom <- if (is_glmm) {                         # adjusted ICC denominator (random + distribution variance)
    v <- tryCatch(performance::get_variance(m, verbose = FALSE), error = function(e) NULL)
    if (is.null(v)) NA_real_ else v$var.random + v$var.residual
  } else sum(vc$vcov)
  
  sd_lo <- sd_hi <- rep(NA_real_, nrow(vc))
  if (sd_ci) {
    ci <- tryCatch(suppressMessages(confint(m, method = "profile", oldNames = FALSE)),
                   error = function(e) NULL)
    if (!is.null(ci)) for (i in seq_len(nrow(vc))) {
      nm <- if (vc$grp[i] == "Residual") "sigma" else paste0("sd_(Intercept)|", vc$grp[i])
      if (nm %in% rownames(ci)) { sd_lo[i] <- ci[nm, 1]; sd_hi[i] <- ci[nm, 2] }
    }
  }
  
  r2n  <- tryCatch(suppressWarnings(suppressMessages(performance::r2_nakagawa(m))), error = function(e) NULL)
  r2m  <- if (is.null(r2n)) NA_real_ else as.numeric(r2n$R2_marginal)
  r2c  <- if (is.null(r2n)) NA_real_ else as.numeric(r2n$R2_conditional)
  msgs <- c(unlist(m@optinfo$conv$lme4$messages), unlist(m@optinfo$warnings))
  conv <- paste0(paste(unique(m@optinfo$optimizer), collapse = "/"), "; ",
                 if (length(msgs)) paste(msgs, collapse = " | ") else "no warnings")
  est  <- if (is_glmm) "Laplace (nAGQ = 1)" else if (isREML(m)) "REML" else "ML"
  disp <- if (is_glmm) sum(residuals(m, type = "pearson")^2) / df.residual(m) else NA_real_
  res_var <- if (is_glmm) NA_real_ else sigma(m)^2
  
  random <- vc %>% transmute(
    Model = model_id,
    `Grouping factor` = grp,
    Term = ifelse(grp == "Residual", "Residual", var1),
    `N groups` = ifelse(grp == "Residual", n_obs, as.integer(ng[grp])),
    `Variance (σ²)` = vcov,
    `SD (σ)` = sdcor,
    `SD 95% CI lower (profile)` = sd_lo,
    `SD 95% CI upper (profile)` = sd_hi,
    ICC = ifelse(grp == "Residual", NA_real_, vcov / denom),
    `Residual variance` = res_var,
    Dispersion = disp,
    `Singular fit` = ifelse(isSingular(m), "Yes", "No"),
    Convergence = conv,
    `R2 marginal` = r2m,
    `R2 conditional` = r2c,
    logLik = as.numeric(logLik(m)),
    AIC = AIC(m),
    BIC = BIC(m),
    Estimation = est,
    `N obs` = n_obs
  )
  
  # ---- Model overview (formula, reference levels, software) ---------------------
  mf   <- model.frame(m)
  facs <- names(Filter(is.factor, mf))
  overview <- tibble(
    Model = model_id, Response = response, Family = family_label,
    Formula = paste(deparse(formula(m), width.cutoff = 500), collapse = " "),
    `Reference levels` = paste0(facs, " = ", vapply(mf[facs], function(x) levels(x)[1], ""), collapse = "; "),
    `N obs` = n_obs,
    `N groups` = paste0(names(ng), " = ", ng, collapse = "; "),
    Software = paste0("R ", getRversion(), "; lme4 ", packageVersion("lme4"),
                      if (!is_glmm) paste0("; lmerTest ", packageVersion("lmerTest")) else "")
  )
  
  # ---- DHARMa diagnostics ---------------------------------------------------------
  set.seed(42)
  sim <- DHARMa::simulateResiduals(m, n = n_sim, plot = FALSE)
  png(file.path(out_dir, paste0(model_id, "_DHARMa.png")), width = 1600, height = 800, res = 150)
  plot(sim); dev.off()
  tu <- DHARMa::testUniformity(sim, plot = FALSE)
  td <- DHARMa::testDispersion(sim, plot = FALSE)
  diagnostics <- tibble(
    Model = model_id,
    Test = c("KS uniformity", "Dispersion"),
    Statistic = c(as.numeric(tu$statistic), as.numeric(td$statistic)),
    p = c(tu$p.value, td$p.value)
  )
  if (poisson) {
    tz <- DHARMa::testZeroInflation(sim, plot = FALSE)
    diagnostics <- bind_rows(diagnostics, tibble(Model = model_id, Test = "Zero-inflation",
                                                 Statistic = as.numeric(tz$statistic), p = tz$p.value))
  }
  
  list(fixed = fixed, random = random, overview = overview, diagnostics = diagnostics)
}

posthoc_month <- function(m, model_id, var = "Month") {
  emm <- emmeans::emmeans(m, stats::as.formula(paste("pairwise ~", var)),
                          lmer.df = "satterthwaite", pbkrtest.limit = 1e5, lmerTest.limit = 1e5)
  as_tibble(summary(emm$contrasts, infer = TRUE, adjust = "tukey")) %>%
    mutate(Model = model_id, .before = 1)
}

write_supplement <- function(results, prefix) {
  for (part in c("fixed", "random", "overview", "diagnostics")) {
    out <- as_tibble(bind_rows(lapply(results, `[[`, part)))
    write_excel_csv2(out, file.path(out_dir, paste0(prefix, "_", part, ".csv")))
    print(out, n = Inf, width = Inf)
  }
  writeLines(capture.output(sessionInfo()), file.path(out_dir, paste0(prefix, "_sessionInfo.txt")))
}
# ==============================================================================
# END SHARED BLOCK
# ==============================================================================


# ---- 1. Data (wide -> long over markers) ---------------------------------------
target_dir <- "C:/Users/oliver.frank/Desktop/PyProjects/beRNN/data/"
markers    <- c("Marker_1", "Marker_2", "Marker_3")

df <- list.files(target_dir, pattern = "trajectory_data_.*highDim.*\\.csv$", full.names = TRUE) %>%
  map_df(~ read_csv(.x, show_col_types = FALSE)) %>%
  mutate(
    Month_chr = as.character(Month),
    Month     = factor(Month_chr, levels = unique(Month_chr[order(parse_number(Month_chr))])),
    Subject   = factor(Subject),
    Model_UID = factor(Model_ID)
  ) %>%
  pivot_longer(all_of(markers), names_to = "Marker", values_to = "Value") %>%
  mutate(Marker = factor(Marker, levels = markers))

# ---- 2. Model ------------------------------------------------------------------
m <- lmer(Value ~ Marker + Month + (1 | Subject) + (1 | Subject:Month), # excluded + (1 | Model_UID) for singularity issues
          data = df,
          control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)))

# ---- 3. Supplement tables --------------------------------------------------------
results <- list()
results[["M3_markers"]] <- summarise_model(
  m, model_id = "M3_markers",
  response = "Marker value (Marker_1-3, long format)",
  family_label = "Gaussian (identity)",
  bt_label = NA_character_)
write_supplement(results, "M3_markers")

if (RUN_POSTHOC) write_excel_csv2(posthoc_month(m, "M3_markers", "Month"),
                                  file.path(out_dir, "M3_markers_posthoc_Month.csv"))