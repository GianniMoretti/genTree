# Installa i pacchetti necessari se non sono già installati
if (!requireNamespace("evtree", quietly = TRUE)) install.packages("evtree")
if (!requireNamespace("rpart", quietly = TRUE)) install.packages("rpart")
if (!requireNamespace("partykit", quietly = TRUE)) install.packages("partykit")

# Carica i pacchetti necessari
library(evtree)
library(rpart)
library(partykit)

# Carica i dati
df <- read.csv("data/chessboard.csv")
df$class <- as.factor(df$class)

# Albero rpart senza restrizione di profondità
rp <- as.party(rpart(class ~ ., data = df, minbucket = 10), model = TRUE)

# Albero ctree senza restrizione di profondità
ct <- ctree(class ~ ., data = df, minbucket = 10, mincriterion = 0.99)

# Visualizza e salva gli alberi
if (!dir.exists("graph_chessboard")) dir.create("graph_chessboard")

png("graph_chessboard/chessboard_rp.png", width = 1200, height = 600)
plot(rp)
dev.off()

png("graph_chessboard/chessboard_ct.png", width = 1200, height = 600)
plot(ct)
dev.off()

set.seed(1090)
# Albero evtree con profondità massima 5
ev <- evtree(class ~ ., data = df, minbucket = 10, maxdepth = 4)

# Visualizza e salva l'albero evtree
png("graph_chessboard/chessboard_ev.png", width = 1200, height = 600)
plot(ev)
dev.off()

print(ev)

# Funzione per calcolare la evaluation function per alberi di classificazione
eval_classification_tree <- function(tree, data, alpha = 1) {
  pred <- predict(tree, data, type = "response")
  MC <- mean(data$class != pred)
  N <- nrow(data)
  if (inherits(tree, "party")) {
    M <- sum(nodeids(tree, terminal = TRUE) %in% nodeids(tree))
  } else {
    stop("Tree type not supported")
  }
  loss <- 2 * N * MC
  comp <- alpha * M * log(N)
  eval <- loss + comp
  return(list(loss = loss, comp = comp, eval = eval, MC = MC, M = M))
}

# Calcola e stampa la evaluation function per ogni albero
cat("rp eval:", eval_classification_tree(rp, df)$eval, "\n")
cat("ct eval:", eval_classification_tree(ct, df)$eval, "\n")
cat("ev eval:", eval_classification_tree(ev, df)$eval, "\n")

# Calcola i risultati per ogni albero
res_rp   <- eval_classification_tree(rp, df)
res_ct   <- eval_classification_tree(ct, df)
res_ev   <- eval_classification_tree(ev, df)

# Funzione per calcolare l'accuracy OOB tramite bootstrap
bootstrap_oob_accuracy <- function(data, model_fun, n_iter = 100, ...) {
  acc <- numeric(n_iter)
  comp <- numeric(n_iter)
  set.seed(42)
  for (i in seq_len(n_iter)) {
    cat("Processing bootstrap sample", i, "of", n_iter, "\n")
    idx <- sample(seq_len(nrow(data)), replace = TRUE)
    oob_idx <- setdiff(seq_len(nrow(data)), unique(idx))
    train <- data[idx, ]
    test <- data[oob_idx, ]
    if (nrow(test) == 0) {
      acc[i] <- NA
      comp[i] <- NA
      next
    }
    model <- model_fun(train, ...)
    pred <- predict(model, test, type = "response")
    acc[i] <- mean(pred == test$class)
    N <- nrow(train)
    if (inherits(model, "party")) {
      M <- sum(nodeids(model, terminal = TRUE) %in% nodeids(model))
    } else {
      comp[i] <- NA
      next
    }
    comp[i] <- 1 * M * log(N) # alpha=1
  }
  list(acc = acc[!is.na(acc)], comp = comp[!is.na(comp)])
}

# Wrapper per i modelli
fit_rpart <- function(data, ...) as.party(rpart(class ~ ., data = data, minbucket = 10), model = TRUE)
fit_ctree <- function(data, ...) ctree(class ~ ., data = data, minbucket = 10, mincriterion = 0.99)
fit_evtree <- function(data, ...) evtree(class ~ ., data = data, minbucket = 10, maxdepth = 2)

# Calcola accuracy OOB e complessità per ciascun modello
res_rp_boot <- bootstrap_oob_accuracy(df, fit_rpart)
res_ct_boot <- bootstrap_oob_accuracy(df, fit_ctree)
res_ev_boot <- bootstrap_oob_accuracy(df, fit_evtree)

acc_rp <- res_rp_boot$acc
acc_ct <- res_ct_boot$acc
acc_ev <- res_ev_boot$acc

comp_rp <- res_rp_boot$comp
comp_ct <- res_ct_boot$comp
comp_ev <- res_ev_boot$comp

# Calcola media e intervallo di confidenza
summary_stats <- function(acc) {
  acc <- acc[!is.na(acc)]
  if (length(acc) < 2) {
    return(c(Mean = mean(acc), `2.5%` = NA, `97.5%` = NA))
  }
  c(
    Mean = mean(acc),
    `2.5%` = as.numeric(quantile(acc, 0.025)),
    `97.5%` = as.numeric(quantile(acc, 0.975))
  )
}

stats_rp <- summary_stats(acc_rp)
stats_ct <- summary_stats(acc_ct)
stats_ev <- summary_stats(acc_ev)

# Tabella riassuntiva
results <- data.frame(
  Model = c("evtree", "rpart", "ctree"),
  MeanAccuracy = c(stats_ev["Mean"], stats_rp["Mean"], stats_ct["Mean"]),
  CI_Lower = c(stats_ev["2.5%"], stats_rp["2.5%"], stats_ct["2.5%"]),
  CI_Upper = c(stats_ev["97.5%"], stats_rp["97.5%"], stats_ct["97.5%"])
)

print(results, row.names = FALSE)

# Calcola media e intervallo di confidenza per la complessità
stats_comp_ev <- summary_stats(comp_ev)
stats_comp_rp <- summary_stats(comp_rp)
stats_comp_ct <- summary_stats(comp_ct)

comp_results <- data.frame(
  Model = c("evtree", "rpart", "ctree"),
  MeanComplexity = c(stats_comp_ev["Mean"], stats_comp_rp["Mean"], stats_comp_ct["Mean"]),
  CI_Lower = c(stats_comp_ev["2.5%"], stats_comp_rp["2.5%"], stats_comp_ct["2.5%"]),
  CI_Upper = c(stats_comp_ev["97.5%"], stats_comp_rp["97.5%"], stats_comp_ct["97.5%"])
)

print(comp_results, row.names = FALSE)

# Salva boxplot delle accuracy
png("graph_chessboard/chessboard_accuracy_boxplot.png", width = 900, height = 600)
boxplot(list(evtree = acc_ev, rpart = acc_rp, ctree = acc_ct),
        ylab = "OOB Accuracy", main = "OOB Accuracy (Bootstrap, n=100)")
dev.off()

# Salva boxplot della complessità
png("graph_chessboard/chessboard_complexity_boxplot.png", width = 900, height = 600)
boxplot(list(evtree = comp_ev, rpart = comp_rp, ctree = comp_ct),
        ylab = "Complexity (alpha * M * log(N))", main = "Model Complexity (Bootstrap, n=100)")
dev.off()
