########### Random Forest Model ########### 
library(yardstick)
library(tidymodels)
library(tidyverse)
library(dplyr)
library(janitor)
library(kknn)

# Load in data
airline <- read.csv("data/airline.csv")
airline <- airline %>% clean_names()

# remove column x and id
airline <- airline %>% select(-x, -id)

# change datatype of variable
airline$arrival_delay_in_minutes = as.integer(airline$arrival_delay_in_minutes)

# change categorical variables to factors
airline$gender <- as.factor(airline$gender)
airline$customer_type <- as.factor(airline$customer_type)
airline$type_of_travel <- as.factor(airline$type_of_travel)
airline$class <- as.factor(airline$class)
airline$satisfaction <- as.factor(airline$satisfaction)

# replace NA with 0
airline$arrival_delay_in_minutes[is.na(airline$arrival_delay_in_minutes)] <-0

# set seed
set.seed(14789)

# split our data into training and testing
airline_split <- initial_split(airline, prop = 0.80, strata = satisfaction)
airline_train <- training(airline_split)
airline_test <- testing(airline_split)

# build recipe
airline_recipe <- recipe(satisfaction ~ customer_type + age + type_of_travel + class + flight_distance + inflight_wifi_service + departure_arrival_time_convenient + ease_of_online_booking +
                           gate_location + food_and_drink + online_boarding + seat_comfort + inflight_entertainment +
                           on_board_service + leg_room_service + baggage_handling + checkin_service + inflight_service +
                           cleanliness + departure_delay_in_minutes + arrival_delay_in_minutes, data = airline) %>%
  step_dummy(all_nominal_predictors()) %>%  # dummy-code all categorical predictors 
  step_center(all_predictors()) %>%  # standardizing predictors
  step_scale(all_predictors())

# k-fold
airline_fold <- vfold_cv(airline_train, v = 10, strata = satisfaction)

# set up Random Forest model
airline_rf_spec <- rand_forest(mtry = tune(), 
                             trees = tune(), 
                             min_n = tune()) %>%
  set_engine("ranger") %>% 
  set_mode("classification")

# create workflow, add model and appropriate recipe
airline_rf_wkflow <- workflow() %>% 
  add_model(airline_rf_spec) %>% 
  add_recipe(airline_recipe)

# create parameter grid
airline_rf_grid <- grid_regular(mtry(range = c(2, 10)), 
                        trees(range = c(1, 10)),
                        min_n(range = c(1,50)),
                        levels = 5)

# fit model to folded data
airline_rf_tune <- tune_grid(
  airline_rf_wkflow,
  resamples = airline_fold,
  grid = airline_rf_grid,
  metrics = metric_set(roc_auc)
)

# select the random forest with the best roc_auc
airline_best_rf <- select_best(airline_rf_tune, metric = "roc_auc")


# fit model to training set
airline_rf_model <- finalize_workflow(airline_rf_wkflow, airline_best_rf)
airline_rf_fit_model <- fit(airline_rf_model, airline_train)

# save
save(airline_rf_tune, airline_rf_model, airline_rf_fit_model, file = "airline_Random_Forest.rda")

