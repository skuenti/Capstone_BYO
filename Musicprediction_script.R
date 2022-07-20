knitr::opts_chunk$set(echo = FALSE)
options(digits = 5)

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(randomForest)
library(data.table)
library(dplyr)
library(knitr)
library(ggplot2)

Sys.time() # when processing started
# load data from the file in the repo (you should be able to do this after pulling)
music_raw_data <- read.csv("music_genre.csv", header=TRUE)

# find out what kind of data there are
kable(as.list(colnames(music_raw_data)), caption="Variables in the dataset")

# initial look at the data
kable(head(music_raw_data))
kable(str(music_raw_data))

# correcting data format in the raw data
music_raw_data$key <- as.factor(music_raw_data$key)
music_raw_data$mode <- as.factor(music_raw_data$mode)
music_raw_data$tempo <- as.numeric(music_raw_data$tempo)
music_raw_data$music_genre <- as.factor(music_raw_data$music_genre)
music_raw_data$artist_name <- as.factor(music_raw_data$artist_name)

# replace negative durations with NAs
music_raw_data$duration_ms <- na_if(music_raw_data$duration_ms, -1)
# head(music_raw_data)

kable(sapply(music_raw_data, {function(x) any(is.na(x))}), caption = "Columns with NAs")

# nrow(music_raw_data)
music_raw_data <- na.omit(music_raw_data) 
# nrow(music_raw_data)

kable(sapply(music_raw_data, {function(x) any(is.na(x))}), caption = "Columns with NAs after data cleaning")

str(music_raw_data)

n_fold <- 5 # setting a global n for cross validations
percent <- 50 # only using this many percent of the entire dataset

# create a subsample of the entire dataset to work with (to control processing speeds)
set.seed(1, sample.kind="Rounding")
light_index <- createDataPartition(music_raw_data$music_genre, times = 1, p = percent/100, list = FALSE)
music_raw_data <- music_raw_data[light_index,]

# create a training (80%) and a test (20%)
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(music_raw_data$music_genre, times = 1, p = 0.2, list=FALSE)
# 20% will go into the validation set
test_set <- music_raw_data[test_index,]
test_set <- droplevels(test_set)
# the rest into the train set
train_set <- music_raw_data[-test_index,]
train_set <- droplevels(train_set)

kable(unique(music_raw_data$music_genre), caption = "Musical Genres")
length(unique(music_raw_data$music_genre))

# train a knn-model with objective parameters
set.seed(1, sample.kind="Rounding")
train_control <- trainControl(method = "cv", number = n_fold, p = .8)
knn_objective <- train(music_genre ~ popularity + key + loudness + mode + tempo + duration_ms, # + artist_name, 
                       method = "knn", 
                       data = train_set, 
                       tuneGrid = data.frame(k = seq(2,502,50)),
                       trControl = train_control)
ggplot(knn_objective, highlight = TRUE)

# train a knn-model with objective parameters
set.seed(1, sample.kind="Rounding")
train_control <- trainControl(method = "cv", number = n_fold, p = .8)
knn_objective <- train(music_genre ~ popularity + key + loudness + mode + tempo + duration_ms, # + artist_name, 
                       method = "knn", 
                       data = train_set, 
                       tuneGrid = data.frame(k = seq(460,502,2)),
                       trControl = train_control)
ggplot(knn_objective, highlight = TRUE)
knn_objective$bestTune$k

knn_objective_accuracy <- confusionMatrix(predict(knn_objective, test_set, type = "raw"), test_set$music_genre)$overall["Accuracy"]
model_results <- tibble(Method = "knn (6 predictors)", Accuracy = knn_objective_accuracy) 
# kable(model_results, caption = "Model accuracies")

# setup of random forest parameters
nodesize_setting <- 5 # treesize, higher number = smaller trees (less branches)
ntree_setting <- 100 # size of the 'forest'
grid <- data.frame(mtry = seq(1,100,5)) # range of mtry = # of variables sampled at splits

set.seed(1, sample.kind="Rounding")
control <- trainControl(method = "cv", number = n_fold)
rf_objective <- train(music_genre ~ popularity + key + loudness + mode + tempo + duration_ms, # + artist_name, 
                      method = "rf",
                      ntree = ntree_setting,
                      trControl = control,
                      tuneGrid = grid,
                      nodesize = nodesize_setting,
                      data = train_set)
ggplot(rf_objective)
rf_objective$bestTune$mtry

test_set$music_genre)$overall["Accuracy"]
model_results <- bind_rows(model_results,
                           tibble(Method="Random forest (6 predictors)",
                                  Accuracy = rf_objective_accuracy))
# kable(model_results, caption = "Model accuracies")

# train a knn-model with objective and subjective parameters
set.seed(1, sample.kind="Rounding")
train_control <- trainControl(method = "cv", number = n_fold, p = .8)
knn_objective_subjective <- train(music_genre ~ popularity + key + loudness + mode + tempo + duration_ms +
                                    acousticness + danceability +
                                    energy + instrumentalness +
                                    liveness + speechiness + 
                                    valence, # + artist_name, 
                                  method = "knn", 
                                  data = train_set, 
                                  tuneGrid = data.frame(k = seq(2,502,50)),
                                  trControl = train_control)
ggplot(knn_objective_subjective, highlight = TRUE)
# knn_objective_subjective$bestTune$k

# train a knn-model with objective and subjective parameters
set.seed(1, sample.kind="Rounding")
train_control <- trainControl(method = "cv", number = n_fold, p = .8)
knn_objective_subjective <- train(music_genre ~ popularity + key + loudness + mode + tempo + duration_ms +
                                    acousticness + danceability +
                                    energy + instrumentalness +
                                    liveness + speechiness + 
                                    valence, # + artist_name, 
                                  method = "knn", 
                                  data = train_set, 
                                  tuneGrid = data.frame(k = seq(460,502,2)),
                                  trControl = train_control)
ggplot(knn_objective_subjective, highlight = TRUE)
knn_objective_subjective$bestTune$k

```{r}
knn_objective_subjective_accuracy <- confusionMatrix(predict(knn_objective_subjective, test_set, type = "raw"), test_set$music_genre)$overall["Accuracy"]
model_results <- bind_rows(model_results,
                           tibble(Method="knn (12 predictors)",
                                  Accuracy = knn_objective_subjective_accuracy))
# kable(model_results, caption = "Model accuracies")

set.seed(1, sample.kind="Rounding")
control <- trainControl(method = "cv", number = n_fold)
rf_objective_subjective <- train(music_genre ~ popularity + key + loudness + mode + tempo + duration_ms +
                                   acousticness + danceability +
                                   energy + instrumentalness +
                                   liveness + speechiness + 
                                   valence, # + artist_name, 
                                 method = "rf",
                                 ntree = ntree_setting,
                                 trControl = control,
                                 tuneGrid = grid,
                                 nodesize = nodesize_setting,
                                 data = train_set)
ggplot(rf_objective_subjective)
rf_objective_subjective$bestTune$mtry

rf_objective_accuracy <- confusionMatrix(predict(rf_objective, test_set, type = "raw"), test_set$music_genre)$overall["Accuracy"]
model_results <- bind_rows(model_results,
                           tibble(Method="Random forest (12 predictors)",
                                  Accuracy = rf_objective_accuracy))
# kable(model_results, caption = "Model accuracies")

kable(model_results, caption = "Model accuracies")

Sys.time()  # when processing ended (in order to find out how long the whole script took to run)