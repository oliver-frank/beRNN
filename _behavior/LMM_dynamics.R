# ======================================================================================
# Do subjects differ in their marker trajectories?
#
# Design: Subject (5) -> Model_ID (20 per subject) -> Month (12)
#         -> 27 validation steps per model and month
#
#   Marker ~ Month + (1 | Subject) + (1 | Subject:Month)
#                  + (1 | Model_UID) 
#
#   (1 | Subject)          overall level differences between subjects
#   (1 | Subject:Month)    deviation of ALL models of a subject from the average
#                          trajectory at a month = trajectory differences between
#                          subjects (the key term)
#   (1 | Model_UID)        model-specific deviation at a month (the cluster of the
#                          27 validation steps)
#   residual               step-to-step variability within a model and month
#
# Model_UID = Subject x Model_ID, so models with the same ID in different subjects
# are never treated as the same model.
# ======================================================================================
library(tidyverse)
library(lme4)
library(lmerTest)

target_dir <- "C:/Users/oliver.frank/Desktop/PyProjects/beRNN/data/"

file_list <- list.files(
  path = target_dir,
  pattern = "trajectory_data_.*highDim_correctOnly.*\\.csv$",
  full.names = TRUE
)

full_dataset <- map_df(file_list, ~ read_csv(.x, show_col_types = FALSE) %>%
                         mutate(Source = basename(.x))) %>%
  mutate(
    Month_chr = as.character(Month),
    Month_num = parse_number(Month_chr) - min(parse_number(Month_chr)),  # 0 = first month
    Subject   = as.factor(Subject),
    # chronological factor levels (Month_1, Month_2, ..., Month_12)
    Month     = factor(Month_chr, levels = unique(Month_chr[order(parse_number(Month_chr))])),
    Model_ID  = as.factor(Model_ID),
    Model_UID = interaction(Subject, Model_ID, drop = TRUE, sep = "_")  # unique per subject
  )

# ---- Sanity checks -------------------------------------------------------------------
print(count(full_dataset, Source))
print(with(full_dataset, table(Subject, Month)))
print(full_dataset %>% distinct(Subject, Model_ID) %>% count(Subject, name = "n_models"))
cat("Number of subjects:", nlevels(full_dataset$Subject), "\n")
cat("Number of unique models:", nlevels(full_dataset$Model_UID), "\n")

# Rows per model-month (should be 27 validation steps)
print(full_dataset %>% count(Model_UID, Month) %>% count(n, name = "n_cells"))

# Optional: if you have a step column (name assumed "Step"), check for drift within
# model-months, and consider adding "+ Step" to the fixed effects below.
# print(full_dataset %>% group_by(Month) %>% summarise(across(all_of(markers), mean)))

markers <- c("Marker_1", "Marker_2", "Marker_3")

# Helper: tidy variance components with a share of total variance
tidy_vc <- function(m, mk, model_name) {
  as.data.frame(VarCorr(m)) %>%
    filter(is.na(var2)) %>%                          # variances only, drop correlations
    transmute(Marker = mk, Model = model_name, Term = grp, Effect = var1,
              Variance = vcov, SD = sdcor, Share = vcov / sum(vcov))
}

# Helper: fit all markers on a dataset, test each random term, Holm across markers
fit_model_A <- function(data, label) {
  
  map_df(markers, function(mk) {
    
    m <- lmer(
      as.formula(paste(
        mk, "~ Month + (1 | Subject) + (1 | Subject:Month) +",
        "(1 | Model_UID)"
      )),
      data = data,
      control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )
    
    # LRT per random term (term dropped one at a time, REML)
    lrt <- as.data.frame(ranova(m)) %>%
      rownames_to_column("Term") %>%
      filter(Term != "<none>") %>%
      mutate(Term = gsub("^\\(1 \\| (.*)\\)$", "\\1", Term)) %>%
      transmute(Term, LRT, p = `Pr(>Chisq)`,
                p_boundary = `Pr(>Chisq)` / 2)     # variance tested at the boundary (0)
    
    tidy_vc(m, mk, label) %>%
      left_join(lrt, by = "Term") %>%
      mutate(Singular = isSingular(m))
  }) %>%
    group_by(Term) %>%
    mutate(p_boundary_holm = p.adjust(p_boundary, method = "holm")) %>%  # across markers
    ungroup()
}

# ======================================================================================
# MODEL A: factor Month, random intercepts (main analysis, all data)
# ======================================================================================
results <- fit_model_A(full_dataset, "A: Month factor")

cat("\n==== Model A: variance components and LRTs ====\n")
print(results, n = Inf)
# Key rows:
#   Term == "Subject:Month"      -> do subjects differ in their trajectories?
#   Term == "Subject"            -> do subjects differ in overall level?
#   Term == "Model_UID:Month"    -> how much do individual models deviate at a month?
#   Term == "Model_UID"          -> stable model-to-model differences
#   Term == "Residual"           -> step-to-step variability

# ======================================================================================
# SENSITIVITY: beRNN_04 only has Months 1-7, so the design is unbalanced.
#   S1: restrict all subjects to Months 1-7
#   S2: drop beRNN_04 and use all months
# ======================================================================================
data_s1 <- full_dataset %>%
  filter(Month_num <= 6) %>%
  droplevels()

data_s2 <- full_dataset %>%
  filter(Subject != "beRNN_04") %>%
  droplevels()

results_s1 <- fit_model_A(data_s1, "S1: Months 1-7, all subjects")
results_s2 <- fit_model_A(data_s2, "S2: without beRNN_04, all months")

cat("\n==== Sensitivity S1: Months 1-7 ====\n")
print(results_s1, n = Inf)
cat("\n==== Sensitivity S2: without beRNN_04 ====\n")
print(results_s2, n = Inf)

# Compact comparison of variance shares across analyses
share_comparison <- bind_rows(results, results_s1, results_s2) %>%
  select(Marker, Model, Term, Share) %>%
  pivot_wider(names_from = Model, values_from = Share)

cat("\n==== Variance shares across analyses ====\n")
print(share_comparison, n = Inf)