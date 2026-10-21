library(tidyverse)
library(lme4)
library(lmerTest)

target_dir <- "C:/Users/oliver.frank/Desktop/PyProjects/beRNN/data/"
markers    <- c("Marker_1", "Marker_2", "Marker_3")

# ---- Load data ---------------------------------------------------------------
df <- list.files(target_dir, pattern = "trajectory_data_.*highDim.*\\.csv$",
                 full.names = TRUE) %>%
  map_df(~ read_csv(.x, show_col_types = FALSE)) %>%
  mutate(
    Month_chr = as.character(Month),
    Month     = factor(Month_chr, levels = unique(Month_chr[order(parse_number(Month_chr))])),
    Subject   = factor(Subject),
    Model_UID = interaction(Subject, Model_ID, drop = TRUE, sep = "_")  # unique per subject
  )

# ---- Fit one model per marker ------------------------------------------------
analyse <- function(mk) {
  ################################################################################
  ################################################################################
  ################################################################################
  m <- lmer(
    as.formula(paste(mk, "~ Month + (1|Subject) + (1|Subject:Month) + (1|Model_UID)")),
    data = df,
    control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  )
  ################################################################################
  ################################################################################
  ################################################################################
  
  # Variance components
  vc        <- as.data.frame(VarCorr(m))
  var_fixed <- var(as.numeric(model.matrix(m) %*% fixef(m)))   # variance explained by Month
  var_total <- var_fixed + sum(vc$vcov)
  
  # Fixed effect: F-test (Satterthwaite), partial eta^2, share of total variance
  a <- anova(m)
  fixed <- tibble(
    Marker = mk, Type = "Fixed", Term = "Month",
    Stat = a$`F value`, p = a$`Pr(>F)`,
    Effect_size = a$`F value` * a$NumDF / (a$`F value` * a$NumDF + a$DenDF),  # partial eta^2
    Share = var_fixed / var_total
  )
  
  # Random effects: LRT (p halved because variance is tested at the boundary 0)
  r <- as.data.frame(ranova(m)) %>%
    rownames_to_column("Term") %>%
    filter(Term != "<none>") %>%
    mutate(Term = gsub("^\\(1 \\| (.*)\\)$", "\\1", Term))
  
  random <- vc %>%
    transmute(Term = grp, Share = vcov / var_total, SD = sdcor) %>%
    left_join(transmute(r, Term, Stat = LRT, p = `Pr(>Chisq)` / 2), by = "Term") %>%
    mutate(Marker = mk, Type = ifelse(Term == "Residual", "Residual", "Random"),
           Effect_size = SD) %>%                      # SD of the random effect, in marker units
    select(Marker, Type, Term, Stat, p, Effect_size, Share)
  
  bind_rows(fixed, random)
}

results <- map_df(markers, analyse) %>%
  group_by(Type, Term) %>%
  mutate(p_holm = ifelse(is.na(p), NA, p.adjust(p, method = "holm"))) %>%  # across markers
  ungroup() %>%
  mutate(Significant = case_when(is.na(p_holm) ~ NA_character_,
                                 p_holm < 0.05 ~ "yes", TRUE ~ "no"))

options(pillar.sigfig = 3, width = 140)
print(results, n = Inf)

# Compact: only the significant effects
cat("\n==== Significant effects (Holm-adjusted p < 0.05) ====\n")
print(filter(results, Significant == "yes") %>% arrange(Marker, desc(Share)), n = Inf)

