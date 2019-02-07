library (keras)
library (dplyr)
library (rsample)
library(DataExplorer)
library(tidyverse)
library (recipes)

Mydata_raw <- read.csv(file="E:/DeepLearningDataSets/supersym.csv", header=TRUE, sep=",")
#                            ^ ^ ^ link to supersym.csv

glimpse(Mydata[,1])

Mydata <- Mydata_raw %>%
  select(target, everything())

#plot_missing(Mydata_raw)

#(Train_1)

rec_obj <- recipe(target ~ ., data = Mydata) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  prep(data = Mydata)

#rec_obj

MybakedData <- bake(rec_obj, new_data = Mydata) %>% select(-target)

Train_1 <- MybakedData[1:250000,]
Train_2 <- MybakedData[250001:500000,]
Train_3 <- MybakedData[500001:750000,]
Train_4 <- MybakedData[750001:999999,]

y_1train <- Mydata[1:250000,1]
y_2train <- Mydata[250001:500000,1]
y_3train <- Mydata[500001:750000,1]
y_4train <- Mydata[750001:999999,1]

dim(Train_1)
dim(y_1train)
dim(Mydata[,1])

model1 <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(Train_1)) %>%
  #layer_dropout(rate=0.4) %>% 
  #layer_dense(units = 128, activation = "relu") %>%
  #layer_dropout(rate=0.4) %>% 
  #layer_dense(units = 128, activation = "relu") %>%
  #layer_dropout(rate=0.4) %>% 
  #layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model1 %>% compile(
  optimizer = "Adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history1 <- model1 %>% fit(
  as.matrix(Train_1), y_1train,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)

plot(history1)

model2 <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(Train_2)) %>%
  #layer_dropout(rate=0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  #layer_dropout(rate=0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  #layer_dropout(rate=0.4) %>% 
  #layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model2 %>% compile(
  optimizer = "Adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history2 <- model2 %>% fit(
  as.matrix(Train_2), y_2train,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)

plot(history2)

model3 <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(Train_3)) %>%
  layer_dropout(rate=0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate=0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate=0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model3 %>% compile(
  optimizer = "Adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history3 <- model3 %>% fit(
  as.matrix(Train_1), y_1train,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)

plot(history3)

model4 <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(Train_4)) %>%
  #layer_dropout(rate=0.4) %>% 
  #layer_dense(units = 128, activation = "relu") %>%
  #layer_dropout(rate=0.4) %>% 
  #layer_dense(units = 128, activation = "relu") %>%
  #layer_dropout(rate=0.4) %>% 
  #layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model4 %>% compile(
  optimizer = "Adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history4 <- model4 %>% fit(
  as.matrix(Train_4), y_4train,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)

plot(history4)
