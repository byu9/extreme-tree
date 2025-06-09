#!/usr/bin/env Rscript
library(gamlss)
library(gamlssx)

train <- read.csv('datasets/pjm/peak_training.csv')
test <- read.csv('datasets/pjm/whole_testing.csv')

fit <- fitGEV(
    Load.MW ~ pb(Forecast.MW) + pb(Degrees.F) + Day + DoW + Month + Hour,
    sigma.fo = ~ pb(Forecast.MW) + pb(Degrees.F) + Day + DoW + Month + Hour,
    nu.fo = ~ pb(Forecast.MW) + pb(Degrees.F) + Day + DoW + Month + Hour,
    data=train
)

mu = fitted(fit, what='mu')
sigma = fitted(fit, what='sigma')
xi = fitted(fit, what='nu')

print(mu)
print(sigma)
print(xi)
