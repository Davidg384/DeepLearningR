library(keras)

mnist <- dataset_fashion_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist

train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))

train_images <- train_images / 255
test_images <- test_images / 255

train_labels <- to_categorical(train_labels, num_classes = 11)
test_labels <- to_categorical(test_labels, num_classes = 11)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "selu",input_shape = c(28, 28, 1)) %>%
  #layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  #layer_dropout(rate = 0.4) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "selu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.4) %>% 
  layer_conv_2d(filters = 96, kernel_size = c(3, 3), activation = "selu")
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.4) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "selu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2))

#model

model <- model %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "selu") %>%
  layer_dropout(rate = 0.4) %>% 
  #layer_dense(units = 64, activation = "selu") %>%
  layer_dense(units = 11, activation = "softmax")

#model

model %>% compile(
  optimizer = "Adadelta",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(
  train_images, train_labels,
  epochs = 10, batch_size=64, validation_split = 0.2
)

model %>% save_model_hdf5("~/github/DeepLearningR/MNist_Fashion_91.h5")


results <- model %>% evaluate(test_images, test_labels)
results