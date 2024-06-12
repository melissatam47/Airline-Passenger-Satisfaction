########### Quadratic Discriminant Model ########### 
library(discrim)
library(tidymodels)
library(janitor)

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

# set up quadratic discriminant model
airline_qda_mod <- discrim_quad() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

# create workflow, add model and appropriate recipe
airline_qda_wkflow <- workflow() %>% 
  add_model(airline_qda_mod) %>% 
  add_recipe(airline_recipe)

# fit model to training data
airline_qda_fit <- fit(airline_qda_wkflow, airline_train)

# fit model to folds
airline_qda_kfold_fit <- fit_resamples(airline_qda_wkflow, airline_fold, control = control_grid(save_pred = TRUE))

# save model
save(airline_qda_fit, airline_qda_kfold_fit, file = "airline_Quadratic_Discriminant.rda")
