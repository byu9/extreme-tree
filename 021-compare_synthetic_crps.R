#!/usr/bin/env Rscript
library(scoringRules)

observations <- read.csv('datasets/synthetic/observations.csv')
our_estimates <- read.csv('090-run_ours_on_synthetic_parameters.csv')
competitor2_estimates <- read.csv('092-run_competitor2_on_synthetic_parameters.csv')

our_crps = crps_gev(
    y=observations$y,
    location=our_estimates$mu,
    scale=our_estimates$sigma,
    shape=our_estimates$xi)

competitor2_crps = crps_gev(
    y=observations$y,
    location=competitor2_estimates$mu,
    scale=competitor2_estimates$sigma,
    shape=competitor2_estimates$xi)

cat('our_crps=', mean(our_crps), '\n')
cat('competitor2_crps=', mean(competitor2_crps), '\n')
