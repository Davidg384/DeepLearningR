
train_dir <- "A:/CloudStore/Dropbox/Berea College/Courses/BUS 386 Deep Learning/Datasets Unzip/Dogs and Cats/cats_and_dogs_small/train"

validation_dir <- "A:/CloudStore/Dropbox/Berea College/Courses/BUS 386 Deep Learning/Datasets Unzip/Dogs and Cats/cats_and_dogs_small/validation"

test_dir <- "A:/CloudStore/Dropbox/Berea College/Courses/BUS 386 Deep Learning/Datasets Unzip/Dogs and Cats/cats_and_dogs_small/test"

train_cats_dir <- file.path(train_dir, "cats")

train_dogs_dir <- file.path(train_dir, "dogs")

validation_cats_dir <- file.path(validation_dir, "cats")

validation_dogs_dir <- file.path(validation_dir, "dogs")

test_cats_dir <- file.path(test_dir, "cats")

test_dogs_dir <- file.path(test_dir, "dogs")


cat("total training cat images:", length(list.files(train_cats_dir)), "\n")

cat("total training dog images:", length(list.files(train_dogs_dir)), "\n")

cat("total validation cat images:",
      length(list.files(validation_cats_dir)), "\n")

cat("total validation dog images:",
      length(list.files(validation_dogs_dir)), "\n")

cat("total test cat images:", length(list.files(test_cats_dir)), "\n")

cat("total test dog images:", length(list.files(test_dogs_dir)), "\n")


## Let's build a convnet model!

library(keras)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)


model %>% compile(
  optimizer = optimizer_rmsprop(lr = 1e-4),
  loss = "binary_crossentropy",
  metrics = c("acc")
)

train_datagen <- image_data_generator(rescale = 1/255)             # scale training set
validation_datagen <- image_data_generator(rescale = 1/255)        # scale validation set

train_generator <- flow_images_from_directory(
  train_dir,                                              # target directory         
  train_datagen,                                          # training data generator         
  target_size = c(150, 150),                              # size all pictures to 150 x 150 pixels        
  batch_size = 20,                                        # binary classification (cats OR dogs)         
  class_mode = "binary" 
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)


batch <- generator_next(train_generator) #generator yields these batches indefinitely; it loops endlessly over the images in the target folder
str(batch)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 50, # determines steps per epoch to set end-of-epoch event
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50 # how many batches to draw from the validation generator
)

model %>% save_model_hdf5("A:/CloudStore/Dropbox/Berea College/Courses/BUS 386 Deep Learning/Datasets/Dogs and Cats/cats_and_dogs_small/cats_and_dogs_small_1.h5")

plot(history)


## data augmentation configuration

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40, # randomly rotate pictures within a range
  width_shift_range = 0.2, # randomly translate picture as a % of width and height
  height_shift_range = 0.2,
  shear_range = 0.2, # randoml apply shearing transformations
  zoom_range = 0.2, # randomly apply zoom as a % of picture size
  horizontal_flip = TRUE, # randomly flipping half the images horizontally; when there is no assumption of horizontal asymmetry
  fill_mode = "nearest" # mode for filling in pixels as a result of previous transformations
)


## displaying randomly augmented training images

fnames <- list.files(train_cats_dir, full.names = TRUE)
img_path <- fnames[[3]]                                            # choose one image to augment   

img <- image_load(img_path, target_size = c(150, 150))            # read the image and resize 
img_array <- image_to_array(img)                                  # convert to array with shape 150 x 150 x 3 
img_array <- array_reshape(img_array, c(1, 150, 150, 3))          # reshape to include 1, 150, 150, 3


augmentation_generator <- flow_images_from_data(                  # generates batches of randomly transformed images; loops continuously ...
                           img_array,                             # ... so breakpoint must be inserted                          
                           generator = datagen,                                                
                           batch_size = 1
)

op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))        # plots the images 

for (i in 1:4) {                                                      
  batch <- generator_next(augmentation_generator)                     
  plot(as.raster(batch[1,,,]))                                        
}                                                                     
par(op)                                                               

## new convnet that includes dropouts

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 1e-4),
  loss = "binary_crossentropy",
  metrics = c("acc")
)

# random image augmentation

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(          # validation data is not augmented, only training data
                   train_dir,                           # target directory              
                   datagen,                             # data generator                
                   target_size = c(150, 150),           # resize images                 
                   batch_size = 32,
                   class_mode = "binary"                # binary classification (cats OR dogs)             
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "binary"
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

model %>% save_model_hdf5("cats_and_dogs_small_2.h5")

