library(readr)
library(dplyr)
library(tidyr)
library(corrplot)


df <- read.csv('C:/Users/User/Desktop/ML/ML_Project/train.csv')
df <- data.frame(df)
View(df)
sum(is.na(df))
colSums(is.na(df))

################################################
## MISSING VALUES ##

# Lot.Frontage: 362
# Mas.Vnr.Area: 22
# BsmtFin.SF.1: 1
# BsmtFin.SF.2: 1
# BsmtFin.Unf.SF: 1
# Total.BsmtFin.SF: 1
# Bsmt.Full.Bath: 1
# Bsmt.Half.Bath: 1
# Garage.Yr.Blt: 122
# Garage.Cars: 1
# Garage.Area: 1

#################################################

cor_matrix <- cor(df[sapply(df, is.numeric)], use = "complete.obs")

# Drop columns (and rows) that have NA values in the correlation matrix
cor_matrix <- cor_matrix[!apply(is.na(cor_matrix), 1, any), !apply(is.na(cor_matrix), 2, any)]

corrplot(cor_matrix, method = "color")
