library(keras)
library(DataExplorer)
library(tidyverse)

# Load data
mnist_fashion <- dataset_fashion_mnist()
train_images <- mnist_fashion$train$x
train_labels <- mnist_fashion$train$y
test_images <- mnist_fashion$test$x
test_labels <- mnist_fashion$test$y

# View the data
glimpse(train_images)
glimpse(train_labels)

# View an image
pic <- train_images[1,,]
plot(as.raster(pic, max = 255))

# Show dimision
dim(train_images)
dim(test_images)

# Reshape the data
train_images <- array_reshape(train_images, c(60000, 28*28))
test_images <- array_reshape(test_images, c(10000, 28*28))

# Scale the data
train_images <- train_images / 255
test_images <- test_images / 255

# Categorize labels & add extra category
train_labels <- to_categorical(train_labels, num_classes = 11)#ONE-Hot encoding
test_labels <- to_categorical(test_labels, num_classes = 11)

# Build the model
network <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", initializer_he_normal(), input_shape = c(28 * 28)) %>%
  layer_dense(units = 16, activation = "relu", initializer_he_normal()) %>%
  layer_dense(units = 11, activation = "softmax")

network %>% compile(
  optimizer = "rmsprop", # optimizer_adam(lr=.0001, decay = 1E-6),
  loss = "categorical_crossentropy",
  metrics = c("accuracy"))

# View the network
network

# Train the model
network %>% fit(train_images, train_labels, epochs = 10, batch_size = 256, validation_split = 0.2)
 