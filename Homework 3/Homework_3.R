library (keras)
library (dplyr)
library (rsample)
library (recipes)
library (tidyverse)

Mydata_raw <- read.csv(file="E:/DeepLearningDataSets/supersym.csv", header=TRUE, sep=",")

#Mydata <- Mydata_raw %>%
 # select(median_house_value, everything())

plot_missing(Mydata_raw)