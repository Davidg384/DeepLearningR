library(keras)
library(tidyverse)
library(stringr)
data_full <- read.csv("C:/Users/tahmi/Desktop/Insincere Questions ZIP/Insincere Questions/train.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
glimpse(data_full)
glimpse(data_full$question_text)
samp <- sample(row(data_full), nrow(data_full))
data_mixed <- data_full[samp,]
data_reduced <- data_mixed[1:100000,]
question <- c(data_reduced$question_text)
targets <- c(data_reduced$target)
maxlen <- 100
max_words <- 1000
tokenize <- text_tokenizer(num_words = max_words, 
                           filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
                           lower = TRUE, split = " ", char_level = FALSE) %>% 
  fit_text_tokenizer(question)
sequences <- texts_to_sequences(tokenize, question)   
word_index <- tokenize$word_index
cat("Found", length(word_index), "unique tokens. \n")
word_data <- pad_sequences(sequences, maxlen = maxlen)  
labels <- as.matrix(targets)
x_train <- word_data[1:50000,]
y_train <- labels[1:50000,]
x_val <- word_data[50001:75000,]
y_val <- labels[50001:75000,]
x_test <- word_data[75001:100000,]
y_test <- labels[75001:100000,]
cat("Shape of data tensor:", dim(word_data),"\n")
cat("Shape of label tensor:",dim(labels),"\n")
embedding_dim <- 100
embedding_matrix <- array(0,c(max_words, embedding_dim))
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>% 
  layer_lstm(units = embedding_dim, dropout = 0.1, recurrent_dropout = 0.2) %>% 
  layer_dense(units = 16, activation = "selu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
summary(model)
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
summary(model)
history <- model %>% fit(
  as.matrix(x_train), y_train,
  epochs = 3,
  batch_size = 512,
  validation_data = list(as.matrix(x_val),y_val)
)
