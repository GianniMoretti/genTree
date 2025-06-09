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
data("Glass", package = "mlbench")

# Rimuovi eventuali NA
Glass <- na.omit(Glass)

# Albero rpart senza restrizione di profondità
rp <- as.party(rpart(Type ~ ., data = Glass, minbucket = 10), model = TRUE)

# Albero ctree senza restrizione di profondità
ct <- ctree(Type ~ ., data = Glass, minbucket = 10, mincriterion = 0.99)

# Visualizza e salva gli alberi
if (!dir.exists("graphs")) dir.create("graphs")

png("graph_glass/glass_rp.png", width = 1200, height = 600)
plot(rp)
dev.off()

png("graph_glass/glass_ct.png", width = 1200, height = 600)
plot(ct)
dev.off()

# Albero evtree con profondità massima 2
set.seed(1090)
ev <- evtree(Type ~ ., data = Glass, minbucket = 10, maxdepth = 2)

# Visualizza e salva l'albero evtree
png("graph_glass/glass_ev.png")
plot(ev)
dev.off()

print(ev)

# Funzione per calcolare la evaluation function per alberi di classificazione
eval_classification_tree <- function(tree, data, alpha = 1) {
  pred <- predict(tree, data, type = "response")
  MC <- mean(data$Type != pred)
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
cat("rp eval:", eval_classification_tree(rp, Glass)$eval, "\n")
cat("ct eval:", eval_classification_tree(ct, Glass)$eval, "\n")
cat("ev eval:", eval_classification_tree(ev, Glass)$eval, "\n")

# Calcola i risultati per ogni albero
res_rp   <- eval_classification_tree(rp, Glass)
res_ct   <- eval_classification_tree(ct, Glass)
res_ev   <- eval_classification_tree(ev, Glass)

# Tabella riassuntiva
results <- data.frame(
  Model = c("evtree", "rpart", "ctree"),
  Misclassification = c(res_ev$MC, res_rp$MC, res_ct$MC),
  EvaluationFunction = c(res_ev$eval, res_rp$eval, res_ct$eval)
)

print(results, row.names = FALSE)
