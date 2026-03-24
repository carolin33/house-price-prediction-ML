# =============================================================================
#  tabelki.R — 8 tabel porównawczych (PNG) dla każdego hiperparametru
# =============================================================================

library(ggplot2)
library(dplyr)
library(readr)
library(gridExtra)
library(grid)

# ---------- Wczytanie danych -------------------------------------------------
df <- read_csv("results_summary.csv", show_col_types = FALSE)
dir.create("wykresy", showWarnings = FALSE)

# ---------- Kolory motywu -----------------------------------------------------
kolor_naglowek_bg   <- "#2D3436"
kolor_naglowek_txt  <- "#FFFFFF"
kolor_wiersz_1      <- "#F5F6FA"
kolor_wiersz_2      <- "#DFE6E9"
kolor_najlepszy     <- "#00B894"
kolor_tytul         <- "#2D3436"

# ---------- Funkcja budująca tabelę ------------------------------------------

generuj_tabele <- function(param, tytul, col_param, col_reg, col_cls,
                           format_reg = "%.3f", format_cls = "%.2f%%",
                           mnoz_cls = 100) {

  # Dane regresji
  d_reg <- df %>%
    filter(problem == "regression", param_name == param) %>%
    select(param_value, avg_test_r2) %>%
    rename(val = param_value, reg = avg_test_r2)

  # Dane klasyfikacji
  d_cls <- df %>%
    filter(problem == "classification", param_name == param) %>%
    select(param_value, avg_test_acc) %>%
    rename(val = param_value, cls = avg_test_acc)

  # Połączenie
  d <- merge(d_reg, d_cls, by = "val", all = TRUE)

  # Zachowanie oryginalnej kolejności z CSV
  orig_order <- unique(df$param_value[df$param_name == param])
  d$val <- factor(d$val, levels = orig_order)
  d <- d[order(d$val), ]

  # Najlepsze wartości
  best_reg <- which.max(d$reg)
  best_cls <- which.max(d$cls)

  # Formatowanie tekstów
  n <- nrow(d)
  tab_text <- matrix("", nrow = n, ncol = 3)
  for (i in 1:n) {
    tab_text[i, 1] <- as.character(d$val[i])
    tab_text[i, 2] <- if (!is.na(d$reg[i])) sprintf(format_reg, d$reg[i]) else "—"
    tab_text[i, 3] <- if (!is.na(d$cls[i])) sprintf(format_cls, d$cls[i] * mnoz_cls) else "—"
  }

  # -- Budowa tabeli za pomocą tableGrob --

  kolumny <- c(col_param, col_reg, col_cls)

  tt <- tableGrob(
    tab_text,
    cols = kolumny,
    rows = NULL,
    theme = ttheme_minimal(
      core = list(
        fg_params = list(fontsize = 12, fontface = "plain", col = "#2D3436"),
        bg_params = list(fill = rep(c(kolor_wiersz_1, kolor_wiersz_2), length.out = n))
      ),
      colhead = list(
        fg_params = list(fontsize = 13, fontface = "bold", col = kolor_naglowek_txt),
        bg_params = list(fill = kolor_naglowek_bg)
      )
    )
  )

  # Pogrubienie najlepszych wyników
  for (j in 2:3) {
    best_row <- if (j == 2) best_reg else best_cls
    if (!is.na(best_row) && length(best_row) > 0) {
      idx <- tt$layout$t == (best_row + 1) & tt$layout$l == j
      if (any(idx)) {
        tt$grobs[idx][[1]]$gp$font <- 2L  # 2 = bold w grid
        tt$grobs[idx][[1]]$gp$col  <- kolor_najlepszy
      }
    }
  }

  # Tytuł
  tytul_grob <- textGrob(
    tytul,
    gp = gpar(fontsize = 15, fontface = "bold", col = kolor_tytul),
    just = "center"
  )

  # Padding
  padding <- unit(0.8, "lines")

  wynik <- arrangeGrob(
    tytul_grob,
    tt,
    ncol = 1,
    heights = unit.c(unit(2, "lines"), unit(1, "null")),
    padding = padding
  )

  wynik
}

# ---------- Generowanie 8 tabel ----------------------------------------------

tabele <- list(
  list(param = "epochs",
       tytul = "1. Wpływ liczby epok (epochs)",
       col_param = "Epochs",
       col_reg = "Regresja Test R²",
       col_cls = "Klasyfikacja Test ACC"),

  list(param = "lr",
       tytul = "2. Wpływ współczynnika uczenia (learning rate)",
       col_param = "Learning Rate",
       col_reg = "Regresja Test R²",
       col_cls = "Klasyfikacja Test ACC"),

  list(param = "hidden_layers",
       tytul = "3. Wpływ architektury warstw ukrytych",
       col_param = "Warstwy ukryte",
       col_reg = "Regresja Test R²",
       col_cls = "Klasyfikacja Test ACC"),

  list(param = "neurons",
       tytul = "4. Wpływ liczby neuronów",
       col_param = "Neurony",
       col_reg = "Regresja Test R²",
       col_cls = "Klasyfikacja Test ACC"),

  list(param = "activation",
       tytul = "5. Wpływ funkcji aktywacji",
       col_param = "Aktywacja",
       col_reg = "Regresja Test R²",
       col_cls = "Klasyfikacja Test ACC"),

  list(param = "test_ratio",
       tytul = "6. Wpływ proporcji zbioru testowego",
       col_param = "Test Ratio",
       col_reg = "Regresja Test R²",
       col_cls = "Klasyfikacja Test ACC"),

  list(param = "weight_init_scale",
       tytul = "7. Wpływ skali inicjalizacji wag",
       col_param = "Skala wag",
       col_reg = "Regresja Test R²",
       col_cls = "Klasyfikacja Test ACC"),

  list(param = "repeats",
       tytul = "8. Wpływ liczby powtórzeń",
       col_param = "Repeats",
       col_reg = "Regresja Test R²",
       col_cls = "Klasyfikacja Test ACC")
)

for (t in tabele) {
  tab_grob <- generuj_tabele(
    param     = t$param,
    tytul     = t$tytul,
    col_param = t$col_param,
    col_reg   = t$col_reg,
    col_cls   = t$col_cls
  )

  n_wierszy <- nrow(df %>% filter(problem == "regression", param_name == t$param))
  wys <- 1.2 + n_wierszy * 0.45  # dynamiczna wysokość

  plik <- paste0("wykresy/tab_", t$param, ".png")
  ggsave(plik, plot = tab_grob, width = 8, height = wys, dpi = 300, bg = "white")
  cat(paste0("  ✓ ", plik, "\n"))
}

cat("\n✅ Gotowe! 8 tabel PNG zapisano w folderze 'wykresy/'.\n")
