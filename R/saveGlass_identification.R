if (!requireNamespace("mlbench", quietly = TRUE)) install.packages("mlbench")
library(mlbench)

data("Glass", package = "mlbench")
Glass <- na.omit(Glass)

write.csv(Glass, file = "data/glass_identification.csv", row.names = FALSE)
