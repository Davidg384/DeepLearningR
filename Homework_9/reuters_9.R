library(keras)
library(DataExplorer)
library(tidyverse)

reuters = dataset_reuters()

max_features <- 100000

maxlen <- 3000


glimpse(reuters)

c(c(x_train, y_train), c(x_test, y_test)) %<-% reuters

x_train <- pad_sequences(x_train, maxlen = maxlen)

x_test <- pad_sequences(x_test, maxlen = maxlen)

y_train = to_categorical(y_train)

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, 
                  output_dim = 8,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  # layer_dense(units = 512, activation = "relu")%>%
  # layer_dropout(rate=0.05)%>%
  # layer_dense(units = 256, activation = "relu")%>%
  # layer_dropout(rate=0.02)%>%
  # layer_dense(units = 128, activation = "relu")%>%
  # layer_dropout(rate=0.01)%>%
  layer_dense(units = 64, activation = "relu")%>%
  layer_dense(units = 46, activation = "softmax")


model %>% compile(
  optimizer = "nadam",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

summary(model)

history <- model %>% fit(
  x_train, 
  y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)
