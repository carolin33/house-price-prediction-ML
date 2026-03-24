# =============================================================================
#  wykresy.R — Wybrane wykresy eksperymentów MLP (6 wykresów: 3 regresja + 3 klasyfikacja)
# =============================================================================

library(ggplot2)
library(dplyr)
library(readr)
library(tidyr)

# ---------- Wczytanie danych -------------------------------------------------
df <- read_csv("results_summary.csv", show_col_types = FALSE)
dir.create("wykresy", showWarnings = FALSE)

# ---------- Globalna paleta i motyw -------------------------------------------
kolor_train <- "#6C5CE7"
kolor_test  <- "#00B894"

motyw <- theme_minimal(base_size = 13) +
  theme(
    text             = element_text(family = "sans"),
    plot.title       = element_text(face = "bold", size = 15, hjust = 0.5,
                                    margin = margin(b = 4)),
    plot.subtitle    = element_text(color = "grey40", size = 10.5, hjust = 0.5,
                                    margin = margin(b = 12)),
    legend.position  = "bottom",
    legend.title     = element_blank(),
    legend.text      = element_text(size = 11),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    axis.title       = element_text(size = 12),
    axis.text        = element_text(size = 10),
    axis.text.x      = element_text(margin = margin(t = 4)),
    plot.margin      = margin(16, 20, 10, 16)
  )


# ═══════════════════════════════════════════════════════════════════════════════
#  REGRESJA — 3 wykresy (R²)
# ═══════════════════════════════════════════════════════════════════════════════

# ---- Funkcja pomocnicza: zgrabny wykres słupkowy Train vs Test ---------------
make_reg_plot <- function(param, tytul, etykieta_x, dodge = 0.75, bar_w = 0.65) {

  d <- df %>%
    filter(problem == "regression", param_name == param) %>%
    select(param_value, avg_train_r2, avg_test_r2) %>%
    pivot_longer(c(avg_train_r2, avg_test_r2),
                 names_to = "Zbior", values_to = "R2") %>%
    mutate(
      Zbior = ifelse(Zbior == "avg_train_r2",
                     "Train R\u00B2", "Test R\u00B2"),
      param_value = factor(param_value,
                           levels = unique(df$param_value[df$param_name == param &
                                                            df$problem == "regression"]))
    ) %>%
    filter(!is.na(R2))

  ggplot(d, aes(x = param_value, y = R2, fill = Zbior)) +
    geom_col(position = position_dodge(width = dodge), width = bar_w,
             color = "white", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.3f", R2)),
              position = position_dodge(width = dodge),
              vjust = -0.6, size = 3.3, fontface = "bold") +
    scale_fill_manual(values = c("Train R\u00B2" = kolor_train,
                                 "Test R\u00B2"  = kolor_test)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12)),
                       limits = c(0, NA)) +
    motyw +
    labs(
      title    = paste0("Regresja \u2014 ", tytul),
      subtitle = "Wsp\u00F3\u0142czynnik determinacji R\u00B2 (train vs test)",
      x = etykieta_x,
      y = expression(R^2)
    )
}

# 1) Regresja: Epochs
p1 <- make_reg_plot("epochs", "Wp\u0142yw liczby epok", "Liczba epok")
ggsave("wykresy/reg_epochs.png", p1, width = 9, height = 5.5, dpi = 300, bg = "white")

# 2) Regresja: Hidden layers
p2 <- make_reg_plot("hidden_layers", "Wp\u0142yw architektury sieci",
                    "Warstwy ukryte") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 9))
ggsave("wykresy/reg_hidden_layers.png", p2, width = 10, height = 5.5, dpi = 300, bg = "white")

# 3) Regresja: Neurons
p3 <- make_reg_plot("neurons", "Wp\u0142yw liczby neuron\u00F3w", "Neurony na warstw\u0119")
ggsave("wykresy/reg_neurons.png", p3, width = 9, height = 5.5, dpi = 300, bg = "white")


# ═══════════════════════════════════════════════════════════════════════════════
#  KLASYFIKACJA — 3 wykresy (Accuracy)
# ═══════════════════════════════════════════════════════════════════════════════

make_cls_plot <- function(param, tytul, etykieta_x, dodge = 0.75, bar_w = 0.65) {

  d <- df %>%
    filter(problem == "classification", param_name == param) %>%
    select(param_value, avg_train_acc, avg_test_acc) %>%
    pivot_longer(c(avg_train_acc, avg_test_acc),
                 names_to = "Zbior", values_to = "ACC") %>%
    mutate(
      Zbior = ifelse(Zbior == "avg_train_acc",
                     "Train Accuracy", "Test Accuracy"),
      param_value = factor(param_value,
                           levels = unique(df$param_value[df$param_name == param &
                                                            df$problem == "classification"]))
    ) %>%
    filter(!is.na(ACC))

  ggplot(d, aes(x = param_value, y = ACC, fill = Zbior)) +
    geom_col(position = position_dodge(width = dodge), width = bar_w,
             color = "white", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.2f%%", ACC * 100)),
              position = position_dodge(width = dodge),
              vjust = -0.6, size = 3.3, fontface = "bold") +
    scale_fill_manual(values = c("Train Accuracy" = "#E17055",
                                 "Test Accuracy"  = "#0984E3")) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12)),
                       labels = scales::percent_format(accuracy = 1),
                       limits = c(0, NA)) +
    motyw +
    labs(
      title    = paste0("Klasyfikacja \u2014 ", tytul),
      subtitle = "Dok\u0142adno\u015B\u0107 klasyfikacji (train vs test)",
      x = etykieta_x,
      y = "Accuracy"
    )
}

# 4) Klasyfikacja: Epochs
p4 <- make_cls_plot("epochs", "Wp\u0142yw liczby epok", "Liczba epok")
ggsave("wykresy/cls_epochs.png", p4, width = 9, height = 5.5, dpi = 300, bg = "white")

# 5) Klasyfikacja: Hidden layers
p5 <- make_cls_plot("hidden_layers", "Wp\u0142yw architektury sieci",
                    "Warstwy ukryte") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 9))
ggsave("wykresy/cls_hidden_layers.png", p5, width = 10, height = 5.5, dpi = 300, bg = "white")

# 6) Klasyfikacja: Activation
p6 <- make_cls_plot("activation", "Wp\u0142yw funkcji aktywacji", "Funkcja aktywacji")
ggsave("wykresy/cls_activation.png", p6, width = 9, height = 5.5, dpi = 300, bg = "white")


cat("\n\u2705 Gotowe! 6 wykres\u00F3w zapisano w folderze 'wykresy/':\n")
cat("   \u2022 reg_epochs.png\n")
cat("   \u2022 reg_hidden_layers.png\n")
cat("   \u2022 reg_neurons.png\n")
cat("   \u2022 cls_epochs.png\n")
cat("   \u2022 cls_hidden_layers.png\n")
cat("   \u2022 cls_activation.png\n")