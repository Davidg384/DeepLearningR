library(keras)
max_features <- 1000
max_len <- 100
cat("Loading data...\n")
reuters <- dataset_reuters(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% reuters
cat(length(x_train), "train sequences\n")
cat(length(x_test), "test sequences")
cat("Pad sequences (samples x time)\n")
x_train <- pad_sequences(x_train, maxlen = max_len)
x_test <- pad_sequences(x_test, maxlen = max_len)
cat("x_train shape:", dim(x_train), "\n")
cat("x_test shape:", dim(x_test), "\n")
y_train <- to_categorical(y_train)


model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 128,
                  input_length = max_len) %>%
  layer_conv_1d(filters = 40, kernel_size = 7, activation = "selu") %>%
  layer_spatial_dropout_1d(rate = 0.25)%>%
  layer_average_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 40, kernel_size = 7, activation = "selu") %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 46, activation = "softmax")
summary(model)
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 1e-3),
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 30,
  batch_size = 16,
  validation_split = 0.2
)
