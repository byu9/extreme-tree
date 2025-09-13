#!/usr/bin/env Rscript
library(gamlss)
library(gamlssx)

train <- read.csv('datasets/pjm/peak_training.csv')
test <- read.csv('datasets/pjm/whole_testing.csv')

# Drop the first colummn (Timestamp)
train_no_timestamps <- subset(train, select = -1)
test_no_timestamps <- subset(test, select = -1)

model <- fitGEV(
    Load.MW ~ Day + DoW + Month + Hour,
    sigma.fo = ~ Day + DoW + Month + Hour,
    nu.fo = ~ Day + DoW + Month + Hour,
    data=train_no_timestamps
)

mu_hat    <- predict(model, what = "mu",  newdata = test_no_timestamps)
sigma_hat <- predict(model, what = "sigma", newdata = test_no_timestamps)
xi_hat    <- predict(model, what = "nu", newdata = test_no_timestamps)

parameter_estimates <- data.frame(index=test$Time,
    mu_hat=mu_hat,
    sigma_hat=sigma_hat,
    xi_hat=xi_hat)

write.csv(
    parameter_estimates,
    file='192-run_competitor2_on_pjm_testing.csv',
    quote=FALSE,
    row.names=FALSE
)