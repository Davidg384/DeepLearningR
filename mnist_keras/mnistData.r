# Loading mnist dataset in keras
library(keras)
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# Build the Neural Network
network <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", initializer_he_normal(), input_shape = c(28 * 28)) %>%
  layer_dense(units = 16, activation = "relu", initializer_he_normal())%>%
  layer_dense(units = 10, activation = "softmax")

# Compile the Neural Network
network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)