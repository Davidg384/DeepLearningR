library(keras)

conv_base <- application_vgg19(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(100, 100, 3)
)

freeze_weights(conv_base)
unfreeze_weights(conv_base, from = "block6_sepconv1_act")

model <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "selu") %>%
  layer_dense(units = 1, activation = "sigmoid")


base_dir <- "D:/DataSets/Malaria Detection"
train_dir <- file.path(base_dir, "Train")
validation_dir <- file.path(base_dir, "valid")
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
  target_size = c(100, 100),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(100, 100),
  batch_size = 20,
  class_mode = "binary"
)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy")
)

#model

callbacks_list <- list(
  callback_early_stopping(monitor = "acc", patience = 3),
  callback_model_checkpoint(filepath = "~/../scripts/R/DeepLearningR/my_MalRec_model1.h5", monitor ="val_loss", save_best_only = TRUE)
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 20,
  callbacks = callbacks_list,
  epochs = 10,
  validation_data = validation_generator,
  validation_steps = 20
)

#Tests accuracy of neural net
model %>% evaluate_generator(generator=validation_generator, steps = 20)
