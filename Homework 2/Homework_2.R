library (keras)
library (dplyr)
library (rsample)
library (recipes)

Mydata_raw <- read.csv(file="~/github/DeepLearningR/Homework 2/housing.csv", header=TRUE, sep=",")

Mydata <- Mydata_raw %>%
  select(median_house_value, everything())

glimpse(Mydata)


set.seed(619)
train_test_split <- initial_split(Mydata, prop=.8)

Train_tbl <- training(train_test_split)
Test_tbl <- testing(train_test_split)

dim(Train_tbl)
dim(Test_tbl)

rec_obj <- recipe(median_house_value ~ ., data = Train_tbl) %>%
  step_bagimpute(all_predictors(), -all_outcomes()) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  prep(data = Train_tbl)

rec_obj

x_train_tbl <- bake(rec_obj, new_data = Train_tbl) %>% select(-median_house_value)
x_test_tbl <- bake(rec_obj, new_data = Test_tbl) %>% select(-median_house_value)


y_train <- Train_tbl[,1]
y_test <- Test_tbl[,1]

model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "selu", input_shape = ncol(x_train_tbl)) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = "Adam",
  loss = "mse",
  metrics = c("mae")
)

history <- model %>% fit(
  as.matrix(x_train_tbl), y_train,
  validation_split = .2,
  epochs = 18,
  batch_size = 32
)

plot(history)

result <- model %>% evaluate(as.matrix(x_test_tbl), y_test)

result