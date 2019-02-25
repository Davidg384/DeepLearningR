library(keras)

base_dir <- "~/Desktop/2019/RScripts/Deep Learning/fruits-360"
train_dir <- file.path(base_dir, "Training")
#validation_dir <- file.path(base_dir, "Test")
test_dir <- file.path(base_dir, "Test")

train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(20, 20),
  batch_size = 20,
  class_mode = "categorical"
)

validation_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(20, 20),
  batch_size = 20,
  class_mode = "binary"
)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "selu",input_shape = c(20, 20, 3)) %>%
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
  layer_dense(units = 24, activation = "softmax")


model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),#"Adadelta",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

#model

callbacks_list <- list(
  callback_early_stopping(monitor = "acc", patience = 5),
  callback_model_checkpoint(filepath = "my_CP_model.h5", monitor ="val_loss", save_best_only = TRUE)
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  callbacks = callbacks_list,
  epochs = 60
  #validation_data = validation_generator,
  #validation_steps = 50
)

plot(history)

#results <- model %>% evaluate(test_images, test_labels)
#results