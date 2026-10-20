# ======================================================================================
# Permutation tests: are datasets more similar within participants than between participants?
#   Q1: closeness of values (same participant & month vs. shuffled participants)
#   Q2: similarity of the month-to-month course (per-month group mean removed)
# One test per density x marker x dataset pair, BH-corrected across densities.
# ======================================================================================
library(jsonlite)
library(dplyr)
library(tidyr)
library(ggplot2)

# --- Configuration ---
markers   <- c("avg_clustering", "mod_value_sparse", "participation_coefficient")
densities <- c("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0")
folder    <- "C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels/__topologicalMarker_pValue_lists"

patterns <- c(
  highDim             = "topologicalMarker_dict_beRNN_compare_4task_beRNN_01_highDim_128_hp8_",
  highDim_correctOnly = "topologicalMarker_dict_beRNN_compare_4task_beRNN_01_highDim_correctOnly_128_hp8",
  brain               = "topologicalMarker_dict_brain_"
)

pair_list <- list(c("brain", "highDim"),
                  c("brain", "highDim_correctOnly"))
B <- 5000
set.seed(1)

# --- Read one JSON file into a long data frame (Pair_ID = model month 1..5, by position) ---
read_markers <- function(density, group) {
  files <- list.files(folder, full.names = TRUE)
  f <- files[grep(paste0(patterns[[group]], ".*", density, "\\.json$"), files)]
  if (length(f) == 0) { message("No file: ", group, " density ", density); return(NULL) }
  data <- fromJSON(f[1])
  bind_rows(lapply(names(data), function(p) {
    bind_rows(lapply(intersect(names(data[[p]]), markers), function(m) {
      v <- data[[p]][[m]][1:5]
      data.frame(Participant = p, Group = group, Pair_ID = seq_along(v),
                 Marker = m, Value = v)
    }))
  }))
}

read_all <- function(density) {
  bind_rows(lapply(names(patterns), function(g) read_markers(density, g)))
}

# --- participants x months matrix for one dataset (rows named by participant) ---
to_mat <- function(d, g) {
  w <- d %>% filter(Group == g) %>%
    select(Participant, Pair_ID, Value) %>%
    pivot_wider(names_from = Pair_ID, values_from = Value) %>%
    as.data.frame()
  rownames(w) <- as.character(w$Participant)
  as.matrix(w[, -1, drop = FALSE])
}

# --- Main loop ---
sim <- data.frame()

for (density in densities) {
  df <- read_all(density)
  if (is.null(df) || nrow(df) == 0) next
  
  for (m in markers) {
    d <- filter(df, Marker == m)
    
    for (pr in pair_list) {
      if (!all(pr %in% d$Group)) next
      A  <- to_mat(d, pr[1])
      Bm <- to_mat(d, pr[2])
      
      # same participants in both datasets, no missing values
      common <- intersect(rownames(A), rownames(Bm))
      A <- A[common, , drop = FALSE]; Bm <- Bm[common, , drop = FALSE]
      ok <- complete.cases(A) & complete.cases(Bm)
      A <- A[ok, , drop = FALSE]; Bm <- Bm[ok, , drop = FALSE]
      n <- nrow(A); id <- seq_len(n)
      if (n < 3) {
        message("Skipping ", density, " ", m, " ", paste(pr, collapse = " vs "), ": n = ", n)
        next
      }
      
      Ac <- sweep(A,  2, colMeans(A))    # remove the shared month trend
      Bc <- sweep(Bm, 2, colMeans(Bm))
      
      s_dist <- function(p) mean(abs(A - Bm[p, , drop = FALSE]))        # lower = more similar
      s_cor  <- function(p) mean(sapply(id, function(i) cor(Ac[i, ], Bc[p[i], ])), na.rm = TRUE)
      
      perms  <- replicate(B, sample(n), simplify = FALSE)
      n_dist <- sapply(perms, s_dist)
      n_cor  <- sapply(perms, s_cor)
      o_dist <- s_dist(id)
      o_cor  <- s_cor(id)
      
      sim <- rbind(sim, data.frame(
        Density = as.numeric(density), Marker = m, Comparison = paste(pr, collapse = " vs "),
        N = n,
        Dist_within = o_dist, Dist_between = mean(n_dist),
        P_dist = (sum(n_dist <= o_dist) + 1) / (B + 1),
        Cor_within  = o_cor,  Cor_between  = mean(n_cor, na.rm = TRUE),
        P_cor  = (sum(n_cor >= o_cor, na.rm = TRUE) + 1) / (B + 1),
        row.names = NULL))
    }
  }
  message("Density ", density, " done")
}

# --- Diagnostics ---
cat("\nRows:", nrow(sim), "(expected 90)\n")
print(colSums(is.na(sim)))
print(table(sim$Density))
print(table(sim$N))

# --- Multiple-testing correction across the 10 densities ---
sim <- sim %>%
  group_by(Marker, Comparison) %>%
  mutate(P_dist_adj = p.adjust(P_dist, "BH"),
         P_cor_adj  = p.adjust(P_cor,  "BH")) %>%
  ungroup()

print(sim, n = Inf)
write.csv(sim, "similarity_permutation_results.csv", row.names = FALSE)

# --- Plot: BH-adjusted p per density ---
sim %>%
  pivot_longer(c(P_dist_adj, P_cor_adj), names_to = "Question", values_to = "P_adj") %>%
  mutate(Question = recode(Question,
                           P_dist_adj = "Q1: within-participant closeness",
                           P_cor_adj  = "Q2: similar month course")) %>%
  ggplot(aes(Density, P_adj, color = Comparison, group = Comparison)) +
  geom_hline(yintercept = 0.05, linetype = "dashed") +
  geom_line() + geom_point() +
  scale_x_continuous(breaks = seq(0.1, 1, 0.1)) +
  facet_grid(Marker ~ Question) +
  labs(y = "BH-adjusted p (permutation)") +
  theme_bw()

