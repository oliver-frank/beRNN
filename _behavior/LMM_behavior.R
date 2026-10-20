# ======================================================================================
# Linear Mixed Model to compare behavioral data
# ======================================================================================
library(jsonlite)
library(tidyverse)
library(lme4)
library(lmerTest) 
library(emmeans)
library(dplyr)
library(ggplot2)
library(performance)

# function to transform json file 
process_json_robust <- function(filepath, value_name) {
  raw_data <- fromJSON(filepath, simplifyVector = FALSE)
  
  # numeric df preprocessing 
  df <- enframe(raw_data, name = "Participant", value = "Months") %>%
    unnest_longer(Months, indices_to = "Month") %>%
    unnest_longer(Months, indices_to = "Task") %>%
    # secure flat list of values 
    mutate(Values = map(Months, as.numeric)) %>% 
    select(-Months) %>%
    # safely unpack
    unnest_longer(Values, indices_to = "Session") %>%
    rename(!!value_name := Values)
  
  return(df)
}

df_rt <- process_json_robust("W:/group_csp/analyses/oliver.frank/Data/participants_dict_reactionTime.json", "ReactionTime")
df_acc <- process_json_robust("W:/group_csp/analyses/oliver.frank/Data/participants_dict_accuracy.json", "Accuracy")
df_comp <- process_json_robust("W:/group_csp/analyses/oliver.frank/Data/participants_dict_complexities.json", "Complexity")

df_final <- df_rt %>%
  left_join(df_acc, by = c("Participant", "Month", "Task", "Session")) %>%
  left_join(df_comp, by = c("Participant", "Month", "Task", "Session"))

df_final <- df_final %>%
  mutate(Month = as.numeric(Month))

df_final <- df_final %>%
  group_by(Participant, Month, Task) %>% 
  fill(Complexity, .direction = "down") %>% 
  ungroup() %>%
  mutate(
    Month = as.factor(Month),
    Complexity = as.factor(Complexity)
  )


# ======================================================================================
# Calculate contrasts
# ======================================================================================
# REACTION TIME ################################################################
model_rt <- lmer(log(ReactionTime) ~ Month + Complexity + Task + (1|Participant), data = df_final) 
# check results RT
summary(model_rt)
anova(model_rt)
ranova(model_rt)
r2(model_rt)
r2_nakagawa(model_rt)
# check performance
performance::r2(model_rt)
performance::icc(model_rt)
# Post-hoc-Tests (Tukey)
posthoc_rt <- emmeans(model_rt, pairwise ~ Month, pbkrtest.limit = 5140)
# show contrast results
print(posthoc_rt$contrasts)
# Potentially add confidence intervall to express imprecise estimation of random effects due to small cohort
confint(model_rt, parm = "sd_(Intercept)|Participant", method = "profile")


# ACCURACY #####################################################################
# Needs gLMM because of relative outcome variable accuracy
df_final <- df_final %>%
  mutate(
    n_trials  = 40,
    n_correct = round(Accuracy / 100 * n_trials),
    n_wrong   = n_trials - n_correct,
    dev       = Accuracy / 100 * n_trials - n_correct   # deviation in trials
  )
model_acc <- glmer(
  cbind(n_correct, n_wrong) ~ Month + Complexity + Task + (1 | Participant),
  data    = df_final,
  family  = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa")
)
# check results ACC
summary(model_acc)
anova(model_acc)
# Fixed effect
m_glmm <- glmer(cbind(n_correct, n_wrong) ~ Month + Complexity + Task + (1 | Participant),
                data = df_final, family = binomial,
                control = glmerControl(optimizer = "bobyqa"))
m_glm  <- glm(cbind(n_correct, n_wrong) ~ Month + Complexity + Task,
              data = df_final, family = binomial)
lrt <- as.numeric(2 * (logLik(m_glmm) - logLik(m_glm)))
c(LRT = lrt, p = 0.5 * pchisq(lrt, df = 1, lower.tail = FALSE))
r2(model_acc)
r2_nakagawa(model_acc)
# check performance
performance::r2(model_acc)
performance::icc(model_acc)
# Post-hoc-Tests (Tukey)
posthoc_acc <- emmeans(model_acc, pairwise ~ Month, pbkrtest.limit = 5140)
# show contrast results
print(posthoc_acc$contrasts)
# Potentially add confidence intervall to express imprecise estimation of random effects due to small cohort
confint(model_acc, parm = "sd_(Intercept)|Participant", method = "profile")


# ======================================================================================
# Calculate emmeans and plot them over months and complexity levels for rt and acc
# ======================================================================================
# Estimated Marginal Means (acc) ###############################################
emm_acc_month <- emmeans(model_acc, "Month")
emm_acc_comp  <- emmeans(model_acc, "Complexity")
emm_acc_task  <- emmeans(model_acc, "Task")

pairs_month <- pwpm(emm_acc_month) 
pairs_comp  <- pwpm(emm_acc_comp)
pairs_task  <- pwpm(emm_acc_task)

plot(emm_acc_month, comparisons = TRUE) + 
  theme_minimal() +
  labs(title = "Influence of months on acc (cleaned from complexity)",
       x = "Estimated mean (EMMean)",
       y = "Month")

plot(emm_acc_comp, comparisons = TRUE) + 
  theme_minimal() +
  labs(title = "Influence of complexity on acc (cleaned from months)",
       x = "EMMean",
       y = "Complexity level")

plot(emm_acc_task, comparisons = TRUE) + 
  theme_minimal() +
  labs(title = "Influence of task on acc (cleaned from months)",
       x = "EMMean",
       y = "Complexity level")


# Estimated Marginal Means (rt) ###############################################
emm_rt_month <- emmeans(model_rt, "Month")
emm_rt_comp  <- emmeans(model_rt, "Complexity")
emm_rt_task  <- emmeans(model_rt, "Task")

pairs_month <- pwpm(emm_rt_month) 
pairs_comp  <- pwpm(emm_rt_comp)
pairs_task  <- pwpm(emm_rt_task)

plot(emm_rt_month, comparisons = TRUE) + 
  theme_minimal() +
  labs(title = "Influence of months on rt (cleaned from complexity)",
       x = "Estimated mean (EMMean)",
       y = "Month")

plot(emm_rt_comp, comparisons = TRUE) + 
  theme_minimal() +
  labs(title = "Influence of complexity on rt (cleaned from months)",
       x = "EMMean",
       y = "Complexity level")

plot(emm_rt_task, comparisons = TRUE) + 
  theme_minimal() +
  labs(title = "Influence of task on rt (cleaned from months)",
       x = "EMMean",
       y = "Complexity level")


################################################################################
# Final Evaluation: Stability of RT, ACC, and Complexity (3-Month Window)
################################################################################
# Ensure column names are standardized so 'emmean' exists
emm_rt_df  <- as.data.frame(emmeans(model_rt, ~ Month))
emm_acc_df <- as.data.frame(emmeans(model_acc, ~ Month))

# descriptive data
comp_stability <- df_final %>%
  mutate(Comp_num = as.numeric(as.character(Complexity))) %>%
  group_by(Month) %>%
  summarise(
    mean_comp = mean(Comp_num, na.rm = TRUE),
    totalTrials = n(),
    .groups = 'drop'
  ) %>%
  mutate(Month_num = as.numeric(as.character(Month))) # Ensure Month is numeric

# preparation for the Loop
final_eval_results <- data.frame()
# Use numeric values of months for the loop
months_vec <- sort(unique(as.numeric(as.character(emm_rt_df$Month))))

for(i in 1:(length(months_vec) - 2)) {
  # Time window (3 consecutive months from the list)
  current_window <- months_vec[i:(i+2)]
  group_label <- paste0("M", current_window[1], "-", current_window[3])
  
  # Filter data for the current window
  # Convert Month in DFs to numeric for comparison
  w_rt   <- emm_rt_df %>% filter(as.numeric(as.character(Month)) %in% current_window)
  w_acc  <- emm_acc_df %>% filter(as.numeric(as.character(Month)) %in% current_window)
  w_comp <- comp_stability %>% filter(Month_num %in% current_window)
  
  # Calculate metrics (SD across the emmean column)
  final_eval_results <- rbind(final_eval_results, data.frame(
    Gruppe      = group_label,
    # Explicitly use the 'emmean' column
    RT_SD       = sd(w_rt$emmean),    # Stability of RT (lower = better)
    Acc_SD      = sd(w_acc$emmean),   # Stability of Accuracy (lower = better)
    Comp_SD     = sd(w_comp$mean_comp), # Design stability (lower = better)
    TotalTrials = sum(w_comp$totalTrials) # Data volume (higher = better)
  ))
}

# utility function for normalization (0 to 1 score)
rescale_utility <- function(x, direction = "high_is_better") {
  if(max(x) == min(x)) return(rep(1, length(x))) # Return 1 if all values are identical
  if(direction == "high_is_better") {
    return((x - min(x)) / (max(x) - min(x))) # Highest value gets highest score
  } else {
    return((max(x) - x) / (max(x) - min(x))) # Lowest value gets highest score
  }
}

# calculate Scoring
final_eval_results <- final_eval_results %>%
  mutate(
    score_rt    = rescale_utility(RT_SD, "low_is_better"),
    score_acc   = rescale_utility(Acc_SD, "low_is_better"),
    score_comp  = rescale_utility(Comp_SD, "low_is_better"),
    score_trials = rescale_utility(TotalTrials, "high_is_better"),
    # Total Score (Average of all 4 factors)
    TotalScore  = (score_rt + score_acc + score_comp + score_trials) / 4
  ) %>%
  arrange(desc(TotalScore))

# Display results
print("Ranking of the most stable 3-month phases:")
print(final_eval_results)

# Mark the best window
cat("\nThe objectively most stable window is:", final_eval_results$Gruppe[1], 
    "with a score of", round(final_eval_results$TotalScore[1], 3))


################################################################################
# Visualization of Design Stability (Complexity over Months)
################################################################################
# Enhance descriptive data with Standard Deviation (SD) for error bars
comp_stability_plot <- df_final %>%
  mutate(Comp_num = as.numeric(as.character(Complexity))) %>%
  group_by(Month) %>%
  summarise(
    mean_comp = mean(Comp_num, na.rm = TRUE),
    sd_comp   = sd(Comp_num, na.rm = TRUE), # Measure of spread within the month
    .groups = 'drop'
  )

# Create plot analogous to the EMMeans style
plot_comp_stability <- ggplot(comp_stability_plot, aes(x = mean_comp, y = Month)) +
  # Add Error Bars (Standard Deviation) to show distribution within months
  geom_errorbarh(aes(xmin = mean_comp - sd_comp, xmax = mean_comp + sd_comp), 
                 height = 0.2, color = "blue", alpha = 0.5) +
  # Plot Mean points
  geom_point(color = "blue", size = 3) +
  theme_minimal() +
  labs(title = "Design Stability: Mean Complexity per Month",
       subtitle = "Blue bars represent Standard Deviation of task levels",
       x = "Mean Complexity Level",
       y = "Month") +
  # Center the view on your actual complexity range
  theme(panel.grid.minor = element_blank())

# Display plot
print(plot_comp_stability)


################################################################################
# Visualization of counts over all participants 
################################################################################
# Count for Reaction Time (RT)
# Filter NAs to ensure only actual measurements are counted
df_count_rt <- df_final %>%
  filter(!is.na(ReactionTime)) %>%
  group_by(Month) %>%
  summarise(count = n(), .groups = 'drop')

plot_rt <- ggplot(df_count_rt, aes(x = Month, y = count)) +
  geom_col(fill = "steelblue") +
  theme_minimal() +
  labs(title = "Number of Entries: Reaction Time", 
       x = "Month", y = "Count (N)")

# Count for Accuracy (acc)
df_count_acc <- df_final %>%
  filter(!is.na(Accuracy)) %>%
  group_by(Month) %>%
  summarise(count = n(), .groups = 'drop')

plot_acc <- ggplot(df_count_acc, aes(x = Month, y = count)) +
  geom_col(fill = "darkorange") +
  theme_minimal() +
  labs(title = "Number of Entries: Accuracy", 
       x = "Month", y = "Count (N)")

# Display plots sequentially
print(plot_rt)
print(plot_acc)


################################################################################
# Visualization of counts for each participant respectively  
################################################################################
library(dplyr)
library(ggplot2)

# Prepare data for RT (Count per Participant and Month)
df_counts_rt <- df_final %>%
  filter(!is.na(ReactionTime)) %>%
  group_by(Participant, Month) %>%
  summarise(count = n(), .groups = 'drop')

# Plot for Reaction Time (RT) by Participant
plot_rt_facet <- ggplot(df_counts_rt, aes(x = Month, y = count, fill = Participant)) +
  geom_col(fill = "steelblue") +
  facet_grid(Participant ~ .) +  # Arrange participants vertically
  theme_minimal() +
  labs(title = "RT: Number of Entries per Month & Participant", 
       x = "Month", y = "N") +
  theme(legend.position = "none") # Remove legend as labels are on the left

# Prepare data for Accuracy (acc)
df_counts_acc <- df_final %>%
  filter(!is.na(Accuracy)) %>%
  group_by(Participant, Month) %>%
  summarise(count = n(), .groups = 'drop')

# Plot for Accuracy by Participant
plot_acc_facet <- ggplot(df_counts_acc, aes(x = Month, y = count, fill = Participant)) +
  geom_col(fill = "darkorange") +
  facet_grid(Participant ~ .) +
  theme_minimal() +
  labs(title = "Accuracy: Number of Entries per Month & Participant", 
       x = "Month", y = "N") +
  theme(legend.position = "none")

# Display plots
print(plot_rt_facet)
print(plot_acc_facet)


df_summary <- df_counts_acc %>%
  group_by(Participant) %>%
  summarise(total_count = sum(count, na.rm = TRUE), .groups = "drop")

print(df_summary)

