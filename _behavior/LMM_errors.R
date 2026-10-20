library(dplyr)
library(tidyr)
library(glmmTMB)
library(jsonlite)

data_path <- "C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\all_tasks_flat_counts.csv"
json_path <- "W:\\group_csp\\analyses\\oliver.frank\\Data\\participants_dict_reactionTime.json"  

all_data <- read.csv(data_path, stringsAsFactors = FALSE)
all_data$subject        <- as.factor(all_data$subject)
all_data$error_category <- as.factor(all_data$error_category)

# --- n_trials per subject x task (summed over all months) ----------------------------
j <- fromJSON(json_path, simplifyVector = FALSE)
trials <- bind_rows(lapply(names(j), function(s)
  bind_rows(lapply(j[[s]], function(m)
    data.frame(subject = s, task = names(m), n = lengths(m)))))) %>%
  group_by(subject, task) %>%
  summarise(n_trials = sum(n), .groups = "drop")

for (current_task in unique(all_data$task)) {
  cat("\n========================================================\n")
  cat("ANALYSIS FOR TASK:", current_task, "\n")
  cat("========================================================\n")
  
  task_data <- all_data %>%
    filter(task == current_task) %>%
    group_by(error_category) %>% filter(sum(count) > 0) %>% ungroup() %>%
    complete(subject, error_category, fill = list(count = 0)) %>%
    mutate(task = current_task) %>%
    left_join(trials, by = c("subject", "task")) %>%               # exposure, also for filled-in rows
    filter(!is.na(n_trials), n_trials > 0) %>%                     # log(0) / NA would break the offset
    group_by(subject) %>% filter(sum(count) > 0) %>% ungroup() %>%
    droplevels()
  
  if (nlevels(task_data$error_category) < 2) {
    cat("Fewer than 2 error categories, skipping.\n")
    next
  }
  
  tryCatch({
    # nbinom takes care of overdispersion in m_0 by very different error_category counts 
    # (random interaction effect in m_1 takes care of that)
    m_0 <- glmmTMB(count ~ error_category + offset(log(n_trials)),
                   family = nbinom2, data = task_data)
    
    m_1 <- glmmTMB(count ~ error_category + (1 | subject) + offset(log(n_trials)),
                   family = nbinom2, data = task_data)
    
    res <- anova(m_0, m_1)
    print(res)
    cat("Boundary-corrected p-value:", res$`Pr(>Chisq)`[2] / 2, "\n")
    print(VarCorr(m_1))
  }, error = function(e) cat("Model failed:", conditionMessage(e), "\n"))
}

