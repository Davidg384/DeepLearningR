library (keras)
library (dplyr)
library (rsample)
library(DataExplorer)
library(tidyverse)
library (recipes)

Mydata_raw <- read.csv(file="~/github/DeepLearningR/Homework_4/Higgs.csv", header=TRUE, sep=",")

glimpse(Mydata_raw)

plot_missing(Mydata_raw)

set.seed(852)

Mydata <- Mydata_raw %>%
  select(Label, everything(), -EventId)

train_test_split <- initial_split(Mydata, prop=.8)

Train_tbl <- training(train_test_split)
Test_tbl <- testing(train_test_split)

rec_obj <- recipe(Label ~ ., data = Train_tbl) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  prep(data = Mydata)

#rec_obj

MybakedTrain <- bake(rec_obj, new_data = Train_tbl) %>% select(-Label)
MybakedTest <- bake(rec_obj, new_data = Test_tbl) %>% select(-Label)

y_train_vec <- ifelse(pull(Train_tbl, Label) == "s", 1, 0)
y_test_vec <- ifelse(pull(Test_tbl, Label) == "s", 1, 0)

model1 <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(MybakedTrain)) %>%
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
  as.matrix(MybakedTrain), y_train_vec,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)

plot(history1)

model2 <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(MybakedTrain)) %>%
  #layer_dropout(rate=0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  #layer_dropout(rate=0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  #layer_dropout(rate=0.4) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model2 %>% compile(
  optimizer = "Adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history2 <- model2 %>% fit(
  as.matrix(MybakedTrain), y_train_vec,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)

plot(history2)

model3 <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(MybakedTrain)) %>%
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
  as.matrix(MybakedTrain), y_train_vec,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)

plot(history3)

model4 <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(MybakedTrain)) %>%
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
  as.matrix(MybakedTrain), y_train_vec,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)

plot(history4)

history1 <- model1 %>% fit(
  as.matrix(MybakedTrain), y_train_vec,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)
history2 <- model2 %>% fit(
  as.matrix(MybakedTrain), y_train_vec,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)
history3 <- model4 %>% fit(
  as.matrix(MybakedTrain), y_train_vec,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)
history4 <- model4 %>% fit(
  as.matrix(MybakedTrain), y_train_vec,
  validation_split = .2,
  epochs = 5,
  batch_size = 256
)

result <- model3 %>% evaluate(as.matrix(MybakedTest), y_test_vec)

result

model1
model2
model3
model4