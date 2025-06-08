#!/usr/bin/env Rscript
library(gamlss)
library(gamlssx)
observations <- read.csv('datasets/synthetic/observations.csv')

fit <- fitGEV(y ~ pb(x), sigma.fo = ~ pb(x), nu.fo = ~ pb(x), data=observations)

parameter_estimates <- data.frame(
    index=observations$index,
    mu=fitted(fit, what='mu'),
    sigma=fitted(fit, what='sigma'),
    xi=fitted(fit, what='nu')
)

write.csv(
    parameter_estimates,
    file='022-run_competitor2_on_synthetic.csv',
    quote=FALSE,
    row.names=FALSE
)
