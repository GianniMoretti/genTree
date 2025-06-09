# Carica il pacchetto evtree se necessario
if (!requireNamespace("evtree", quietly = TRUE)) install.packages("evtree")
library(evtree)

# Carica i dati
data("BBBClub", package = "evtree")

# Salva i dati in un file CSV
write.csv(BBBClub, file = "BBBClub.csv", row.names = FALSE)
