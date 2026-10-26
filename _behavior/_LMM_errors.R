# ==============================================================================
# M4 | Error profile (all tasks, one model) | binomial GLMM (glmmTMB, logit link), random intercepts only
#      Categories are task-specific, so every category gets its own
#      ID (Task::Category); the participant effect is category-specific: (1 | Participant:Category).
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
data_path <- "C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\all_tasks_flat_counts_monthWise.csv"
all_data  <- read.csv(data_path, stringsAsFactors = FALSE)

# One rule for all tasks: a category stays separate only if EVERY participant has >= MIN_COUNT_SUBJECT
# counts in it; all other categories are summed into OTHER_RARE. Tasks with fewer than MIN_CATS such
# categories are left out.
MIN_COUNT_SUBJECT <- 10
MIN_CATS          <- 2
OTHER_LABEL       <- "OTHER_RARE"

prep_task <- function(dt, task_name) {
  subj_mat <- xtabs(count ~ error_category + subject, data = dt)
  keep     <- rownames(subj_mat)[apply(subj_mat, 1, min) >= MIN_COUNT_SUBJECT]
  if (length(keep) < MIN_CATS) { message("Task ", task_name, " left out: < ", MIN_CATS, " categories"); return(NULL) }
  cats <- if (nrow(subj_mat) > length(keep)) c(keep, OTHER_LABEL) else keep
  dt %>%
    mutate(error_category = ifelse(error_category %in% keep, error_category, OTHER_LABEL)) %>%
    group_by(subject, month, error_category) %>%
    summarise(count = sum(count), .groups = "drop") %>%
    complete(nesting(subject, month), error_category = cats, fill = list(count = 0)) %>%
    group_by(subject, month) %>% mutate(total = sum(count)) %>% ungroup() %>%
    filter(total > 0) %>%                                   # drop months without errors
    mutate(task = task_name)
}

df_final <- bind_rows(lapply(unique(all_data$task),
                             function(tk) prep_task(filter(all_data, task == tk), tk))) %>%
  transmute(Participant = factor(subject),
            Month       = factor(month),
            Task        = factor(task),
            Category    = factor(paste(task, error_category, sep = "::")),   # task-specific category ID
            count, total)

# How many counts end up in OTHER_RARE per task?
df_final %>% group_by(Task) %>%
  summarise(n_categories = n_distinct(Category),
            pct_other    = round(100 * sum(count[grepl(paste0("::", OTHER_LABEL, "$"), Category)]) / sum(count), 2)) %>%
  as.data.frame() %>% print(row.names = FALSE)

# ---- 2. Model ------------------------------------------------------------------
# Participant / Participant:Month intercepts of the accuracy model are not estimable here (shares within a
# participant-month sum to 1), so the participant effect is category-specific.
# Binomial (no extra dispersion term): a beta-binomial / observation-level term had no overdispersion left
# to explain (variance ~ 0) and made the Hessian non-positive-definite.
m <- glmmTMB(cbind(count, total - count) ~ Month + Task +
               (1 | Category) + (1 | Participant:Category),
             data = df_final, family = binomial(link = "logit"),
             control = glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4)))
print(summary(m))
cat("pdHess:", isTRUE(m$sdr$pdHess), "\n")
if (!isTRUE(m$sdr$pdHess)) warning("Hessian not positive-definite: do NOT interpret the LRTs below.")

# Overdispersion check (should be ~1; if clearly > 1, add (1 | obs) with df_final$obs <- factor(seq_len(nrow(df_final))))
cat("Pearson chi2 / df:", sum(residuals(m, type = "pearson")^2) / df.residual(m), "\n")

# LRT
m_noPart <- update(m, . ~ . - (1 | Participant:Category))   # do participants differ in their error profile?
m_noCat  <- update(m, . ~ . - (1 | Category))
anova(m, m_noPart); anova(m, m_noCat)

# ---- 3. Supplement tables --------------------------------------------------------
results <- list()
results[["M4_errors"]] <- summarise_model(
  m, model_id = "M4_errors",
  response = "Share of errors in category (errors in category out of all errors per participant x month x task)",
  family_label = "Binomial (logit)", bt_label = "Odds ratio")
write_supplement(results, "M4_errors")

if (RUN_POSTHOC) write_excel_csv2(posthoc_month(m, "M4_errors", "Month"),
                                  file.path(out_dir, "M4_errors_posthoc_Month.csv"))