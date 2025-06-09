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

# Split train/test
train_idx <- 1:2000
test_idx <- 2001:4000
train <- df[train_idx, ]
test <- df[test_idx, ]

# Albero rpart senza restrizione di profondità
rp <- as.party(rpart(class ~ ., data = train, minbucket = 10), model = TRUE)

# Albero ctree senza restrizione di profondità
ct <- ctree(class ~ ., data = train, minbucket = 10, mincriterion = 0.99)

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
ev <- evtree(class ~ ., data = train, minbucket = 10, maxdepth = 5)

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

# Calcola e stampa la evaluation function, accuracy e complexity per ogni albero su train e test
models <- list(evtree = ev, rpart = rp, ctree = ct)
model_names <- names(models)

results <- data.frame(
  Model = character(),
  Set = character(),
  Accuracy = numeric(),
  Complexity = numeric(),
  Eval = numeric(),
  stringsAsFactors = FALSE
)

for (model_name in model_names) {
  model <- models[[model_name]]
  # Train set
  res_train <- eval_classification_tree(model, train)
  # Test set
  res_test <- eval_classification_tree(model, test)
  results <- rbind(
    results,
    data.frame(
      Model = model_name,
      Set = "Train",
      Accuracy = 1 - res_train$MC,
      Complexity = res_train$comp,
      Eval = res_train$eval
    ),
    data.frame(
      Model = model_name,
      Set = "Test",
      Accuracy = 1 - res_test$MC,
      Complexity = res_test$comp,
      Eval = res_test$eval
    )
  )
}

print(results, row.names = FALSE)
