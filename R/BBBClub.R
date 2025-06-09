# Installa i pacchetti necessari se non sono già installati
if (!requireNamespace("evtree", quietly = TRUE)) install.packages("evtree")
if (!requireNamespace("rpart", quietly = TRUE)) install.packages("rpart")
if (!requireNamespace("partykit", quietly = TRUE)) install.packages("partykit")

# Carica i pacchetti necessari
library(evtree)
library(rpart)
library(partykit)

# Carica i dati
data("BBBClub", package = "evtree")

# Albero rpart senza restrizione di profondità
rp <- as.party(rpart(choice ~ ., data = BBBClub, minbucket = 10), model = TRUE)

# Albero rpart con profondità massima 2
rp2 <- as.party(rpart(choice ~ ., data = BBBClub, minbucket = 10, maxdepth = 2), model = TRUE)

# Albero ctree senza restrizione di profondità
ct <- ctree(choice ~ ., data = BBBClub, minbucket = 10, mincriterion = 0.99)

# Albero ctree con profondità massima 2
ct2 <- ctree(choice ~ ., data = BBBClub, minbucket = 10, mincriterion = 0.99, maxdepth = 2)

# Visualizza e salva gli alberi
png("graph_BBBC/rp.png")
plot(rp)
dev.off()

png("graph_BBBC/ct.png", width = 1200, height = 600)
plot(ct)
dev.off()

# Albero evtree con profondità massima 2
set.seed(1090)
# evtree senza restrizione di profondità e massimo 2000 iterazioni
ev <- evtree(choice ~ ., data = BBBClub, minbucket = 10, maxdepth = 2)

# Visualizza e salva l'albero evtree
png("graph_BBBC/ev.png")
plot(ev)
dev.off()

print(ev)

# Funzione per calcolare la evaluation function per alberi di classificazione
eval_classification_tree <- function(tree, data, alpha = 1) {
  # Predizioni
  pred <- predict(tree, data, type = "response")
  # Misclassification count
  MC <- mean(data$choice != pred)
  N <- nrow(data)
  # Numero nodi terminali
  if (inherits(tree, "party")) {
    M <- sum(nodeids(tree, terminal = TRUE) %in% nodeids(tree))
  } else {
    stop("Tree type not supported")
  }
  # Loss e complessità
  loss <- 2 * N * MC
  comp <- alpha * M * log(N)
  eval <- loss + comp
  return(list(loss = loss, comp = comp, eval = eval, MC = MC, M = M))
}

# Calcola e stampa la evaluation function per ogni albero
cat("rp eval:", eval_classification_tree(rp, BBBClub)$eval, "\n")
cat("rp2 eval:", eval_classification_tree(rp2, BBBClub)$eval, "\n")
cat("ct eval:", eval_classification_tree(ct, BBBClub)$eval, "\n")
cat("ct2 eval:", eval_classification_tree(ct2, BBBClub)$eval, "\n")
cat("ev eval:", eval_classification_tree(ev, BBBClub)$eval, "\n")

# Calcola i risultati per ogni albero
res_rp   <- eval_classification_tree(rp, BBBClub)
res_rp2  <- eval_classification_tree(rp2, BBBClub)
res_ct   <- eval_classification_tree(ct, BBBClub)
res_ct2  <- eval_classification_tree(ct2, BBBClub)
res_ev   <- eval_classification_tree(ev, BBBClub)

# Tabella riassuntiva
results <- data.frame(
  Model = c("evtree", "rpart", "ctree", "rpart2", "ctree2"),
  Misclassification = c(res_ev$MC, res_rp$MC, res_ct$MC, res_rp2$MC, res_ct2$MC),
  EvaluationFunction = c(res_ev$eval, res_rp$eval, res_ct$eval, res_rp2$eval, res_ct2$eval)
)

print(results, row.names = FALSE)