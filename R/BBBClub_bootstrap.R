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

# Funzione per calcolare accuracy OOB su un bootstrap sample
bootstrap_accuracy <- function(data, model_fun, model_args = list(), n_boot = 250, seed = 42) {
  set.seed(seed)
  acc <- numeric(n_boot)
  for (i in 1:n_boot) {
    idx <- sample(seq_len(nrow(data)), replace = TRUE)
    train <- data[idx, ]
    oob_idx <- setdiff(seq_len(nrow(data)), unique(idx))
    if (length(oob_idx) == 0) {
      acc[i] <- NA
      next
    }
    test <- data[oob_idx, ]
    fit <- do.call(model_fun, c(list(data = train), model_args))
    pred <- predict(fit, test, type = "response")
    acc[i] <- mean(test$choice == pred)
  }
  acc
}

# Wrapper per evtree
evtree_fun <- function(data, ...) {
  evtree(choice ~ ., data = data, ...)
}
# Wrapper per rpart
rpart_fun <- function(data, ...) {
  as.party(rpart(choice ~ ., data = data, ...), model = TRUE)
}
# Wrapper per ctree
ctree_fun <- function(data, ...) {
  ctree(choice ~ ., data = data, ...)
}

# Calcola accuracy bootstrap per ogni modello
acc_ev   <- bootstrap_accuracy(BBBClub, evtree_fun, list(minbucket=10, maxdepth=2))
acc_rp   <- bootstrap_accuracy(BBBClub, rpart_fun, list(minbucket=10))
acc_rp2  <- bootstrap_accuracy(BBBClub, rpart_fun, list(minbucket=10, maxdepth=2))
acc_ct   <- bootstrap_accuracy(BBBClub, ctree_fun, list(minbucket=10, mincriterion=0.99))
acc_ct2  <- bootstrap_accuracy(BBBClub, ctree_fun, list(minbucket=10, mincriterion=0.99, maxdepth=2))

# Tabella riassuntiva con media e intervallo
results <- data.frame(
  Model = c("evtree", "rpart", "ctree", "rpart2", "ctree2"),
  MeanAccuracy = c(mean(acc_ev, na.rm=TRUE), mean(acc_rp, na.rm=TRUE), mean(acc_ct, na.rm=TRUE), mean(acc_rp2, na.rm=TRUE), mean(acc_ct2, na.rm=TRUE)),
  MinAccuracy = c(min(acc_ev, na.rm=TRUE), min(acc_rp, na.rm=TRUE), min(acc_ct, na.rm=TRUE), min(acc_rp2, na.rm=TRUE), min(acc_ct2, na.rm=TRUE)),
  MaxAccuracy = c(max(acc_ev, na.rm=TRUE), max(acc_rp, na.rm=TRUE), max(acc_ct, na.rm=TRUE), max(acc_rp2, na.rm=TRUE), max(acc_ct2, na.rm=TRUE))
)
print(results, row.names = FALSE)

# Boxplot delle accuracy
acc_df <- data.frame(
  Accuracy = c(acc_ev, acc_rp, acc_ct, acc_rp2, acc_ct2),
  Model = factor(rep(c("evtree", "rpart", "ctree", "rpart2", "ctree2"), each=length(acc_ev)))
)
png("graph_BBBC/boxplot_accuracy.png", width=900, height=600)
boxplot(Accuracy ~ Model, data=acc_df, main="OOB Accuracy (250 bootstrap)", ylab="Accuracy", col="lightblue")
dev.off()