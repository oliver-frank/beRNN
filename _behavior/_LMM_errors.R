# ==============================================================================
# M4 | Error profile | beta-binomial GLMM (glmmTMB, logit link), random intercepts only
# ==============================================================================
library(tidyverse)
library(glmmTMB)

# ==============================================================================
# HELPER BLOCK - glmmTMB version
# ==============================================================================
out_dir     <- "W:\\group_csp\\analyses\\oliver.frank\\Data\\supplement_tables\\M4_errors"
RUN_POSTHOC <- FALSE                 # TRUE = additional Tukey contrasts over the time factor
dir.create(out_dir, showWarnings = FALSE)

summarise_model <- function(m, model_id, response, family_label,
                            bt_label = NA_character_,  # e.g. "Odds ratio"; NA = no back-transformation
                            n_sim    = 500) {
  n_obs <- nobs(m)
  bt    <- !is.na(bt_label)
  mf    <- m$frame
  
  # ---- Fixed effects ----------------------------------------------------------
  cf    <- as.data.frame(coef(summary(m))$cond)
  zc    <- qnorm(0.975)
  fixed <- tibble(
    Model = model_id, Response = response, Family = family_label,
    Term = rownames(cf), Estimate = cf$Estimate, SE = cf$`Std. Error`,
    df = NA_real_, `Statistic type` = "z",
    Statistic = cf$`z value`, p = cf$`Pr(>|z|)`
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
  
  # ---- DHARMa diagnostics (simulated residuals; valid for models with their own dispersion) ----
  set.seed(42)
  sim <- DHARMa::simulateResiduals(m, n = n_sim, plot = FALSE)
  tu     <- DHARMa::testUniformity(sim, plot = FALSE)
  td     <- DHARMa::testDispersion(sim, plot = FALSE)
  to     <- DHARMa::testOutliers(sim, plot = FALSE)
  fmt_p  <- function(p) ifelse(p < 0.001, "< 0.001", sprintf("%.3f", p))
  
  # Test results go into a header ABOVE the panels, so plotted points can never cover them
  png(file.path(out_dir, paste0(model_id, "_DHARMa.png")), width = 2000, height = 1000, res = 200)
  par(mfrow = c(1, 2), oma = c(0, 0, 5, 0))
  pt <- list(pch = 16, cex = 0.35, col = grDevices::adjustcolor("#2C7FB8", alpha.f = 0.35))   # blue dots, quantile lines stay red/black
  do.call(DHARMa::plotQQunif, c(list(sim, testUniformity = FALSE, testOutliers = FALSE,
                                     testDispersion = FALSE), pt))
  do.call(DHARMa::plotResiduals, c(list(sim), pt))
  mtext(paste0(model_id, "  |  n = ", nobs(m)), outer = TRUE, line = 3, font = 2, cex = 1)
  mtext(sprintf("KS uniformity: D = %.3f, p %s   |   Dispersion: ratio = %.3f, p %s   |   Outliers: p %s",
                as.numeric(tu$statistic), ifelse(tu$p.value < 0.001, "< 0.001", paste("=", fmt_p(tu$p.value))),
                as.numeric(td$statistic), ifelse(td$p.value < 0.001, "< 0.001", paste("=", fmt_p(td$p.value))),
                ifelse(to$p.value < 0.001, "< 0.001", paste("=", fmt_p(to$p.value)))),
        outer = TRUE, line = 1, cex = 0.8)
  dev.off()
  disp   <- as.numeric(td$statistic)
  disp_p <- td$p.value
  
  # ---- Random effects + model-level information ---------------------------------
  vcc <- VarCorr(m)$cond
  vc  <- tibble(grp   = names(vcc),
                var1  = "(Intercept)",
                sdcor = vapply(vcc, function(x) attr(x, "stddev")[[1]], numeric(1))) %>%
    mutate(vcov = sdcor^2)
  ng <- vapply(vc$grp, function(g) nrow(unique(mf[strsplit(g, ":")[[1]]])), integer(1))
  names(ng) <- vc$grp
  
  # Latent-scale (logit) ICC: sigma_g^2 / (sum of random variances + pi^2/3)
  dist_var <- pi^2 / 3
  denom    <- sum(vc$vcov) + dist_var
  
  # Latent-scale R2 (Nakagawa-style): fixed-effect variance from the linear predictor
  var_f <- tryCatch(var(as.numeric(predict(m, re.form = NA, type = "link"))), error = function(e) NA_real_)
  var_r <- sum(vc$vcov)
  r2m   <- var_f / (var_f + var_r + dist_var)
  r2c   <- (var_f + var_r) / (var_f + var_r + dist_var)
  
  pdh  <- isTRUE(m$sdr$pdHess)
  conv <- paste0("nlminb; convergence code = ", m$fit$convergence, "; pdHess = ", pdh,
                 if (!pdh) " (WARNING: Hessian not positive-definite)" else "")
  
  random <- vc %>% transmute(
    Model = model_id,
    `Grouping factor` = grp,
    Term = var1,
    `N groups` = as.integer(ng[grp]),
    `Variance (σ²)` = vcov,
    `SD (σ)` = sdcor,
    `SD 95% CI lower (profile)` = NA_real_,      # not implemented for glmmTMB: confint(m, parm = "theta_")
    `SD 95% CI upper (profile)` = NA_real_,
    ICC = vcov / denom,
    `Residual variance` = NA_real_,
    `Dispersion (DHARMa)` = disp,
    `Dispersion p (DHARMa)` = disp_p,
    `Dispersion parameter (φ)` = sigma(m),       # beta-binomial dispersion
    `Singular fit` = ifelse(any(sdcor < 1e-4), "Yes", "No"),
    Convergence = conv,
    `R2 marginal` = r2m,
    `R2 conditional` = r2c,
    logLik = as.numeric(logLik(m)),
    AIC = AIC(m),
    BIC = BIC(m),
    Estimation = "Laplace (glmmTMB)",
    `N obs` = n_obs
  )
  
  # ---- Model overview (formula, reference levels, software) ---------------------
  facs <- names(Filter(is.factor, mf))
  overview <- tibble(
    Model = model_id, Response = response, Family = family_label,
    Formula = paste(deparse(formula(m), width.cutoff = 500), collapse = " "),
    `Reference levels` = paste0(facs, " = ", vapply(mf[facs], function(x) levels(x)[1], ""), collapse = "; "),
    `N obs` = n_obs,
    `N groups` = paste0(names(ng), " = ", ng, collapse = "; "),
    Software = paste0("R ", getRversion(), "; glmmTMB ", packageVersion("glmmTMB"))
  )
  
  # ---- Diagnostics table ----------------------------------------------------------
  diagnostics <- tibble(
    Model = model_id,
    Test = c("KS uniformity", "Dispersion"),
    Statistic = c(as.numeric(tu$statistic), as.numeric(td$statistic)),
    p = c(tu$p.value, td$p.value)
  )
  
  list(fixed = fixed, random = random, overview = overview, diagnostics = diagnostics)
}

posthoc_month <- function(m, model_id, var = "Month") {
  emm <- emmeans::emmeans(m, stats::as.formula(paste("pairwise ~", var)))
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
# END HELPER BLOCK
# ==============================================================================


# ---- 1. Data -------------------------------------------------------------------
data_path <- "C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\all_tasks_flat_counts_monthWise.csv"  
all_data  <- read.csv(data_path, stringsAsFactors = FALSE)

# ---- 2. Model (one per task) + 3. Supplement tables ------------------------------
results <- list()
posthoc <- list()

for (current_task in unique(all_data$task)) {
  message("Task: ", current_task)
  model_id <- paste0("M4_errors_", current_task)
  
  d <- all_data %>%
    filter(task == current_task) %>%
    group_by(error_category) %>% filter(sum(count) > 0) %>% ungroup() %>%   # drop empty categories
    group_by(subject, month) %>% mutate(total = sum(count)) %>% ungroup() %>%
    filter(total > 0) %>%                                                   # drop months without errors
    mutate(subject = factor(subject), month = factor(month), category = factor(error_category))
  
  tryCatch({
    # Share of errors falling into a category. The subject-level and subject-by-month intercepts
    # are dropped on purpose: shares within a subject-month sum to 1, so a shift that applies to
    # all categories at once cannot be estimated.
    m <- glmmTMB(cbind(count, total - count) ~ category + month + (1 | subject:category),
                 data = d, family = betabinomial(link = "logit"),
                 control = glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4)))
    
    # LRT 
    m_noPart_Cat  <- update(m, . ~ . - (1 | subject:category))
    print(anova(m, m_noPart_Cat))
    
    results[[model_id]] <- summarise_model(
      m, model_id = model_id,
      response = "Share of errors in category (errors in category out of all errors per subject x month)",
      family_label = "Beta-binomial (logit)", bt_label = "Odds ratio")
    
    if (RUN_POSTHOC) posthoc[[model_id]] <- posthoc_month(m, model_id, "month")
  }, error = function(e) message("  failed: ", conditionMessage(e)))
}

write_supplement(results, "M4_errors")
if (RUN_POSTHOC) write_excel_csv2(bind_rows(posthoc), file.path(out_dir, "M4_errors_posthoc_month.csv"))