# Installa i pacchetti necessari se non sono già installati
if (!requireNamespace("mlbench", quietly = TRUE)) install.packages("mlbench")
if (!requireNamespace("evtree", quietly = TRUE)) install.packages("evtree")
if (!requireNamespace("rpart", quietly = TRUE)) install.packages("rpart")
if (!requireNamespace("partykit", quietly = TRUE)) install.packages("partykit")

# Carica i pacchetti necessari
library(mlbench)
library(evtree)
library(rpart)
library(partykit)

# Carica i dati
data("BostonHousing", package = "mlbench")

# Rimuovi eventuali NA
Boston <- na.omit(BostonHousing)

# Albero rpart senza restrizione di profondità
rp <- as.party(rpart(medv ~ ., data = Boston, minbucket = 10), model = TRUE)

# Albero ctree senza restrizione di profondità
ct <- ctree(medv ~ ., data = Boston, minbucket = 10, mincriterion = 0.99)

# Visualizza e salva gli alberi
if (!dir.exists("graph_boston")) dir.create("graph_boston")

png("graph_boston/boston_rp.png", width = 1200, height = 600)
plot(rp)
dev.off()

png("graph_boston/boston_ct.png", width = 1200, height = 600)
plot(ct)
dev.off()

# Albero evtree con profondità massima 5
set.seed(1090)
ev <- evtree(medv ~ ., data = Boston, minbucket = 10, maxdepth = 5)

# Visualizza e salva l'albero evtree
png("graph_boston/boston_ev.png")
plot(ev)
dev.off()

print(ev)

# Funzione per calcolare la evaluation function per alberi di regressione
eval_regression_tree <- function(tree, data, alpha = 1) {
  pred <- predict(tree, data)
  RMSE <- sqrt(mean((data$medv - pred)^2))
  N <- nrow(data)
  if (inherits(tree, "party")) {
    M <- sum(nodeids(tree, terminal = TRUE) %in% nodeids(tree))
  } else {
    stop("Tree type not supported")
  }
  loss <- N * RMSE
  comp <- alpha * M * log(N)
  eval <- loss + comp
  return(list(loss = loss, comp = comp, eval = eval, RMSE = RMSE, M = M))
}

# Calcola e stampa la evaluation function per ogni albero
cat("rp eval:", eval_regression_tree(rp, Boston)$eval, "\n")
cat("ct eval:", eval_regression_tree(ct, Boston)$eval, "\n")
cat("ev eval:", eval_regression_tree(ev, Boston)$eval, "\n")

# Calcola i risultati per ogni albero
res_rp   <- eval_regression_tree(rp, Boston)
res_ct   <- eval_regression_tree(ct, Boston)
res_ev   <- eval_regression_tree(ev, Boston)

# Funzione per calcolare la RMSE OOB tramite bootstrap
bootstrap_oob_rmse <- function(data, model_fun, n_iter = 250, ...) {
  rmse <- numeric(n_iter)
  comp <- numeric(n_iter)
  set.seed(42)
  for (i in seq_len(n_iter)) {
    cat("Processing bootstrap sample", i, "of", n_iter, "\n")
    idx <- sample(seq_len(nrow(data)), replace = TRUE)
    oob_idx <- setdiff(seq_len(nrow(data)), unique(idx))
    train <- data[idx, ]
    test <- data[oob_idx, ]
    # Se non ci sono dati OOB, non si può calcolare la RMSE/complessità
    if (nrow(test) == 0) {
      rmse[i] <- NA
      comp[i] <- NA
      next
    }
    model <- model_fun(train, ...)
    pred <- predict(model, test)
    rmse[i] <- sqrt(mean((test$medv - pred)^2))
    # Calcola la complessità sul modello addestrato
    N <- nrow(train)
    if (inherits(model, "party")) {
      M <- sum(nodeids(model, terminal = TRUE) %in% nodeids(model))
    } else {
      comp[i] <- NA
      next
    }
    comp[i] <- 1 * M * log(N) # alpha=1
  }
  list(rmse = rmse[!is.na(rmse)], comp = comp[!is.na(comp)])
}

# Wrapper per i modelli
fit_rpart <- function(data, ...) as.party(rpart(medv ~ ., data = data, minbucket = 10), model = TRUE)
fit_ctree <- function(data, ...) ctree(medv ~ ., data = data, minbucket = 10, mincriterion = 0.99)
fit_evtree <- function(data, ...) evtree(medv ~ ., data = data, minbucket = 10, maxdepth = 5)

# Calcola RMSE OOB e complessità per ciascun modello
res_rp_boot <- bootstrap_oob_rmse(Boston, fit_rpart)
res_ct_boot <- bootstrap_oob_rmse(Boston, fit_ctree)
res_ev_boot <- bootstrap_oob_rmse(Boston, fit_evtree)

rmse_rp <- res_rp_boot$rmse
rmse_ct <- res_ct_boot$rmse
rmse_ev <- res_ev_boot$rmse

comp_rp <- res_rp_boot$comp
comp_ct <- res_ct_boot$comp
comp_ev <- res_ev_boot$comp

# Calcola media e intervallo di confidenza
summary_stats <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) < 2) {
    return(c(Mean = mean(x), `2.5%` = NA, `97.5%` = NA))
  }
  c(
    Mean = mean(x),
    `2.5%` = as.numeric(quantile(x, 0.025)),
    `97.5%` = as.numeric(quantile(x, 0.975))
  )
}

stats_rp <- summary_stats(rmse_rp)
stats_ct <- summary_stats(rmse_ct)
stats_ev <- summary_stats(rmse_ev)

# Tabella riassuntiva
results <- data.frame(
  Model = c("evtree", "rpart", "ctree"),
  MeanRMSE = c(stats_ev["Mean"], stats_rp["Mean"], stats_ct["Mean"]),
  CI_Lower = c(stats_ev["2.5%"], stats_rp["2.5%"], stats_ct["2.5%"]),
  CI_Upper = c(stats_ev["97.5%"], stats_rp["97.5%"], stats_ct["97.5%"])
)

print(results, row.names = FALSE)

# Salva boxplot delle RMSE
png("graph_boston/boston_rmse_boxplot.png", width = 900, height = 600)
boxplot(list(evtree = rmse_ev, rpart = rmse_rp, ctree = rmse_ct),
        ylab = "OOB RMSE", main = "OOB RMSE (Bootstrap, n=250)")
dev.off()

# Salva boxplot della complessità
png("graph_boston/boston_complexity_boxplot.png", width = 900, height = 600)
boxplot(list(evtree = comp_ev, rpart = comp_rp, ctree = comp_ct),
        ylab = "Complexity (alpha * M * log(N))", main = "Model Complexity (Bootstrap, n=250)")
dev.off()
