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

