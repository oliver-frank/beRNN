library(dplyr)

data_path <- "C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\all_tasks_flat_counts.csv"

all_data <- read.csv(data_path, stringsAsFactors = FALSE)
all_data$subject        <- as.character(all_data$subject)
all_data$error_category <- as.character(all_data$error_category)  # characters, so no unused factor levels

results <- list()

for (current_task in unique(all_data$task)) {
  
  tab <- all_data %>%
    filter(task == current_task) %>%
    xtabs(count ~ subject + error_category, data = .)
  
  # drop empty categories (columns) and empty subjects (rows)
  tab <- tab[rowSums(tab) > 0, colSums(tab) > 0, drop = FALSE]
  
  if (nrow(tab) < 2 || ncol(tab) < 2) next
  
  tryCatch({
    set.seed(1)
    chi  <- chisq.test(tab, simulate.p.value = TRUE, B = 1e5)
    fish <- tryCatch(fisher.test(tab, simulate.p.value = TRUE, B = 1e5),
                     error = function(e) NULL)
    
    n <- sum(tab); k <- min(dim(tab)) - 1
    
    results[[current_task]] <- data.frame(
      Task       = current_task,
      Subjects   = nrow(tab),
      Categories = ncol(tab),
      Errors     = n,
      Chisq      = as.numeric(chi$statistic),
      p_chisq    = chi$p.value,
      p_fisher   = if (is.null(fish)) NA_real_ else fish$p.value,
      Cramers_V  = sqrt(as.numeric(chi$statistic) / (n * k))
    )
  }, error = function(e) message(current_task, ": ", conditionMessage(e)))
}

fmt_p <- function(p) ifelse(is.na(p), "NA", ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))

final_table <- bind_rows(results) %>%
  mutate(p_chisq_holm  = p.adjust(p_chisq, "holm"),
         p_fisher_holm = p.adjust(p_fisher, "holm")) %>%
  transmute(Task, Subjects, Categories, Errors,
            `Chi-square`    = round(Chisq, 2),
            `p (MC)`        = fmt_p(p_chisq),
            `p Holm`        = fmt_p(p_chisq_holm),
            `p Fisher (MC)` = fmt_p(p_fisher),
            `p Fisher Holm` = fmt_p(p_fisher_holm),
            `Cramer's V`    = round(Cramers_V, 3))

print(final_table, row.names = FALSE)
# write.csv(final_table, "supplement_error_pattern_tests.csv", row.names = FALSE)

