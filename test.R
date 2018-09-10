library(rBayesianOptimization)
library(xgboost)

# ### Credit Card Fraud data 100% ###
# 
# CREDIT_CARD <- "dataset/creditcard.csv"
# cc <- read.csv(CREDIT_CARD, header = TRUE, sep = ",")
# cc$Class <- as.numeric(cc$Class)
# ccmat <- data.matrix(cc)
# dtrain <- xgb.DMatrix(ccmat[,c(1:30)],
#                       label = ccmat[,ncol(ccmat)])
# cv_folds <- KFold(ccmat[,ncol(ccmat)], nfolds = 5,
#                   stratified = TRUE, seed = 0)
# xgb_cv_bayes <- function(max.depth, min_child_weight, subsample) {
#   cv <- xgb.cv(params = list(booster = "gbtree", eta = 0.01,
#                              max_depth = max.depth,
#                              min_child_weight = min_child_weight,
#                              subsample = subsample, colsample_bytree = 0.3,
#                              lambda = 1, alpha = 0,
#                              objective = "binary:logistic",
#                              eval_metric = "auc"),
#                data = dtrain, nround = 100,
#                folds = cv_folds, prediction = TRUE, showsd = TRUE,
#                early_stopping_rounds = 5, maximize = TRUE, verbose = 0)
#   list(Score = cv$evaluation_log[, max(test_auc_mean)],
#        Pred = cv$pred)
# }
# OPT_Res <- BayesianOptimization(xgb_cv_bayes,
#                                 bounds = list(max.depth = c(2L, 6L),
#                                               min_child_weight = c(1L, 10L),
#                                               subsample = c(0.5, 0.8)),
#                                 init_grid_dt = NULL, init_points = 5, n_iter = 2,
#                                 acq = "ucb", kappa = 2.576, eps = 0.0,
#                                 verbose = TRUE)

### Credit Card Fraud data 25% ###

CREDIT_CARD <- "dataset/creditcardcsv.csv"
cc <- read.csv(CREDIT_CARD, header = TRUE, sep = ",")
cc$Class <- as.numeric(cc$Class)
ccmat <- data.matrix(cc)
dtrain <- xgb.DMatrix(ccmat[,c(1:30)],
                       label = ccmat[,ncol(ccmat)])
cv_folds <- KFold(ccmat[,ncol(ccmat)], nfolds = 5,
                   stratified = TRUE, seed = 0)
xgb_cv_bayes <- function(max.depth, min_child_weight, subsample) {
   cv <- xgb.cv(params = list(booster = "gbtree", eta = 0.01,
                              max_depth = max.depth,
                              min_child_weight = min_child_weight,
                              subsample = subsample, colsample_bytree = 0.3,
                              lambda = 1, alpha = 0,
                              objective = "binary:logistic",
                              eval_metric = "auc"),
                data = dtrain, nround = 100,
                folds = cv_folds, prediction = TRUE, showsd = TRUE,
                early_stopping_rounds = 5, maximize = TRUE, verbose = 0)
   list(Score = cv$evaluation_log[, max(test_auc_mean)],
        Pred = cv$pred)
}
OPT_Res <- BayesianOptimization(xgb_cv_bayes,
                                 bounds = list(max.depth = c(2L, 6L),
                                               min_child_weight = c(1L, 10L),
                                               subsample = c(0.5, 0.8)),
                                 init_grid_dt = NULL, init_points = 5, n_iter = 2,
                                 acq = "ucb", kappa = 2.576, eps = 0.0,
                                 verbose = TRUE)

### Agaricus Data ###
#data(agaricus.train, package = "xgboost")
# dtrain <- xgb.DMatrix(agaricus.train$data,
#                       label = agaricus.train$label)
# cv_folds <- KFold(agaricus.train$label, nfolds = 5,
#                   stratified = TRUE, seed = 0)
# xgb_cv_bayes <- function(max.depth, min_child_weight, subsample) {
#   cv <- xgb.cv(params = list(booster = "gbtree", eta = 0.01,
#                              max_depth = max.depth,
#                              min_child_weight = min_child_weight,
#                              subsample = subsample, colsample_bytree = 0.3,
#                              lambda = 1, alpha = 0,
#                              objective = "binary:logistic",
#                              eval_metric = "auc"),
#                data = dtrain, nround = 100,
#                folds = cv_folds, prediction = TRUE, showsd = TRUE,
#                early.stop.round = 5, maximize = TRUE, verbose = 0)
#   list(Score = cv$evaluation_log[, max(test_auc_mean)],
#        Pred = cv$pred)
# }
# OPT_Res <- BayesianOptimization(xgb_cv_bayes,
#                                 bounds = list(max.depth = c(2L, 6L),
#                                 min_child_weight = c(1L, 10L),
#                                 subsample = c(0.5, 0.8)),
#                                 init_grid_dt = NULL, init_points = 5, n_iter = 2,
#                                 acq = "ucb", kappa = 2.576, eps = 0.0,
#                                 verbose = TRUE)
