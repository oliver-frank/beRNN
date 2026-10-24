# ==============================================================================
# M1 | Accuracy | beta-binomial GLMM (glmmTMB, logit link), random intercepts only
#      Beta-binomial = binomial with an extra dispersion parameter (phi) that absorbs
#      session-to-session variance beyond what the binomial allows (overdispersion).
# ==============================================================================
library(jsonlite)
library(tidyverse)
library(glmmTMB)

# ==============================================================================
# HELPER BLOCK - glmmTMB version
# ==============================================================================
out_dir     <- "W:\\group_csp\\analyses\\oliver.frank\\Data\\supplement_tables\\M1_performance" 
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
  disp <- sum(residuals(m, type = "pearson")^2) / df.residual(m)   # Pearson chi2 / df (should be ~1)
  
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
    Dispersion = disp,
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
process_json_robust <- function(filepath, value_name) {
  raw_data <- fromJSON(filepath, simplifyVector = FALSE)
  enframe(raw_data, name = "Participant", value = "Months") %>%
    unnest_longer(Months, indices_to = "Month") %>%
    unnest_longer(Months, indices_to = "Task") %>%
    mutate(Values = map(Months, as.numeric)) %>%
    select(-Months) %>%
    unnest_longer(Values, indices_to = "Session") %>%
    rename(!!value_name := Values)
}

df_rt   <- process_json_robust("W:/group_csp/analyses/oliver.frank/Data/participants_dict_reactionTime.json", "ReactionTime")
df_acc  <- process_json_robust("W:/group_csp/analyses/oliver.frank/Data/participants_dict_accuracy.json",     "Accuracy")
df_comp <- process_json_robust("W:/group_csp/analyses/oliver.frank/Data/participants_dict_complexities.json", "Complexity")

df_final <- df_rt %>%
  left_join(df_acc,  by = c("Participant", "Month", "Task", "Session")) %>%
  left_join(df_comp, by = c("Participant", "Month", "Task", "Session")) %>%
  mutate(Month = as.numeric(Month)) %>%
  group_by(Participant, Month, Task) %>%
  fill(Complexity, .direction = "down") %>%
  ungroup() %>%
  mutate(Month = as.factor(Month), Complexity = as.factor(Complexity))

# Accuracy (%) -> successes / failures out of 40 trials per session
df_final <- df_final %>%
  mutate(n_trials  = 40,
         n_correct = round(Accuracy / 100 * n_trials),
         n_wrong   = n_trials - n_correct)

# ---- 2. Model ------------------------------------------------------------------
m <- glmmTMB(cbind(n_correct, n_wrong) ~ Month + Complexity + Task +
               (1 | Participant) + (1 | Participant:Month),
             data = df_final, family = betabinomial(link = "logit"),
             control = glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4)))

# LRT 
m_noPart  <- update(m, . ~ . - (1 | Participant))
m_noPM    <- update(m, . ~ . - (1 | Participant:Month))
anova(m, m_noPart); anova(m, m_noPM)

# ---- 3. Supplement tables --------------------------------------------------------
results <- list()
results[["M1_accuracy"]] <- summarise_model(
  m, model_id = "M1_accuracy",
  response = "Accuracy (correct/incorrect of 40 trials)",
  family_label = "Beta-binomial (logit)", bt_label = "Odds ratio")
write_supplement(results, "M1_accuracy")

if (RUN_POSTHOC) write_excel_csv2(posthoc_month(m, "M1_accuracy", "Month"),
                                  file.path(out_dir, "M1_accuracy_posthoc_Month.csv"))