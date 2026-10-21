library(dplyr)
library(glmmTMB)

data_path <- "C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\all_tasks_flat_counts_monthWise.csv"  # <- adjust

all_data <- read.csv(data_path, stringsAsFactors = FALSE)

fmt_p <- function(p) ifelse(is.na(p), "NA", ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))

# LRT of a reduced vs. the full model; random effects get the boundary correction (p / 2)
lrt_test <- function(m_reduced, m_full, effect, random = FALSE) {
  a   <- anova(m_reduced, m_full)
  chi <- a$Chisq[2]
  df  <- a$Df[2]
  p   <- pchisq(chi, df, lower.tail = FALSE)
  if (random) p <- 0.5 * p
  data.frame(Effect = effect, Type = ifelse(random, "random", "fixed"),
             Chisq = chi, df = df, p = p)
}

results <- list()

for (current_task in unique(all_data$task)) {
  message("Task: ", current_task)
  
  d <- all_data %>%
    filter(task == current_task) %>%
    group_by(error_category) %>% filter(sum(count) > 0) %>% ungroup() %>%   # drop empty categories
    group_by(subject, month) %>% mutate(total = sum(count)) %>% ungroup() %>%
    filter(total > 0) %>%                                                  # drop months without errors
    mutate(subject = factor(subject), month = factor(month), category = factor(error_category))
  
  tryCatch({
    m_full <- glmmTMB(count ~ category + month + offset(log(total)) +
                        (1 | subject) + (1 | subject:category) + (1 | subject:month),
                      data = d, family = poisson)
    
    tests <- bind_rows(
      # fixed effects
      lrt_test(update(m_full, . ~ . - category), m_full, "Category"),
      lrt_test(update(m_full, . ~ . - month),    m_full, "Month"),
      # random effects
      lrt_test(update(m_full, . ~ . - (1 | subject)),          m_full, "Subject",          random = TRUE),
      lrt_test(update(m_full, . ~ . - (1 | subject:category)), m_full, "Subject:Category", random = TRUE),
      lrt_test(update(m_full, . ~ . - (1 | subject:month)),    m_full, "Subject:Month",    random = TRUE)
    )
    
    vc <- VarCorr(m_full)$cond
    sds <- sapply(vc, function(x) attr(x, "stddev")[[1]])
    tests$SD <- c(NA, NA, sds["subject"], sds["subject:category"], sds["subject:month"])
    
    results[[current_task]] <- tests %>%
      mutate(Task = current_task, Hessian_OK = isTRUE(m_full$sdr$pdHess))
  }, error = function(e) message("  failed: ", conditionMessage(e)))
}

final_table <- bind_rows(results) %>%
  mutate(p_holm = p.adjust(p, "holm")) %>%      # correction across all tests
  transmute(Task, Effect, Type,
            SD = round(SD, 3), `LRT Chi-square` = round(Chisq, 2), df,
            `p (LRT)` = fmt_p(p), `p Holm` = fmt_p(p_holm), Hessian_OK)

print(final_table, row.names = FALSE)
# write.csv(final_table, "supplement_error_profile_glmm.csv", row.names = FALSE)