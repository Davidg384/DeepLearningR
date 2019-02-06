library (keras)
library (dplyr)
library (rsample)
library (recipes)
library (tidyverse)
library(DataExplorer)

Mydata_raw <- read.csv(file="E:/DeepLearningDataSets/supersym.csv", header=TRUE, sep=",")

#Mydata <- Mydata_raw %>%
 # select(median_house_value, everything())

plot_missing(Mydata_raw)

recipeObject <- recipe(Churn~ ., data = Mydata_raw)%>%
  step_bagimpute(all_predictors(), -all_outcomes())%>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  step_dummy(all_predictors(), -all_outcomes(), one_hot = TRUE) %>%
  prep(data = Mydata_raw)

recipeObject