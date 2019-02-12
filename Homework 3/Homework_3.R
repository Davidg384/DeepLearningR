library (keras)
library (dplyr)
library (rsample)
library(DataExplorer)
library(tidyverse)
library (recipes)

Mydata_raw <- read.csv(file="~/github/DeepLearningR/Homework 3/supersym.csv", header=TRUE, sep=",")

glimpse(Mydata[,1])

Mydata <- Mydata_raw %>%
  select(target, everything())

# Imports raw data
Mydata_raw <- read.csv(file="E:/DeepLearningDataSets/supersym.csv", header=TRUE, sep=",")
#                            ^ ^ ^ link to supersym.csv

# view raw data
glimpse(Mydata[,1])


# Organizes data 
Mydata <- Mydata_raw %>%
  select(target, everything())

# define recipe object to center, scale, and prep data
rec_obj <- recipe(target ~ ., data = Mydata) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  prep(data = Mydata)

#rec_obj

MybakedData <- bake(rec_obj, new_data = Mydata) %>% select(-target)

# Create new dataset from recipe
MybakedData <- bake(rec_obj, new_data = Mydata) %>% select(-target)

# split data into 4 equal size training sets from baked data
Train_1 <- MybakedData[1:250000,]
Train_2 <- MybakedData[250001:500000,]
Train_3 <- MybakedData[500001:750000,]
Train_4 <- MybakedData[750001:999999,]


# Response Variables for each training set 
y_1train <- Mydata[1:250000,1]
y_2train <- Mydata[250001:500000,1]
y_3train <- Mydata[500001:750000,1]
y_4train <- Mydata[750001:999999,1]

dim(Train_1)
dim(y_1train)
dim(Mydata[,1])


# Makes sequential Network for Dataset 1
model1 <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(Train_1)) %>%
  #layer_dropout(rate=0.4) %>% 
  #layer_dense(units = 128, activation = "relu") %>%
  #layer_dropout(rate=0.4) %>% 
  #layer_dense(units = 128, activation = "relu") %>%
  #layer_dropout(rate=0.4) %>% 
  #layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")


# Sets optimizer, loss, and metrics for model1
model1 %>% compile(
  optimizer = "Adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history1 <- model1 %>% fit(
  as.matrix(Train_1), y_1train,
  validation_split = .2,
  epochs = 10,
  batch_size = 256
)

# Runs (fit) the model and stores plottable info
history1 <- model1 %>% fit(
  as.matrix(Train_1), y_1train,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)

model1
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
  epochs = 10,
  batch_size = 256
)

plot(history4)
