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
################################################################################
################################################################################
################################################################################
model_rt <- lmer(log(ReactionTime) ~ Month + Complexity + Task + (1|Participant), data = df_final) 
################################################################################
################################################################################
################################################################################
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
################################################################################
################################################################################
################################################################################
model_acc <- glmer(
  cbind(n_correct, n_wrong) ~ Month + Complexity + Task + (1 | Participant),
  data    = df_final,
  family  = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa")
)
################################################################################
################################################################################
################################################################################
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


