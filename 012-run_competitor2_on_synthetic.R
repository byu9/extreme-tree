#!/usr/bin/env Rscript
library(gamlss)
library(gamlssx)
observations <- read.csv('datasets/synthetic/observations.csv')

fit <- fitGEV(y ~ cs(x), sigma.fo = ~ cs(x), nu.fo = ~ cs(x), data=observations)
mu = fitted(fit, what='mu')
sigma = fitted(fit, what='sigma')
xi = fitted(fit, what='nu')

parameter_estimates <- data.frame(index=observations$index, mu=mu, sigma=sigma, xi=xi)
write.csv(
    parameter_estimates,
    file='092-run_competitor2_on_synthetic_parameters.csv',
    quote=FALSE,
    row.names=FALSE
)

quantiles <- c(0.1, 0.5, 0.9, 0.99, 0.999, 0.999999)
quantile_values <- matrix(NA, nrow=nrow(observations), ncol=length(quantiles))

# Loop over time steps to calculate the quantile predictions
for (i in 1:nrow(observations)) {
    mu_i <- mu[[i]]
    sigma_i <- sigma[[i]]
    nu_i <- xi[[i]]

    # Compute quantiles for this time step
    quantile_values[i, ] <- qGEV(p=quantiles, mu=mu_i, sigma=sigma_i, nu=nu_i)
}

quantile_estimates <- data.frame(observations$index, quantile_values)
colnames(quantile_estimates) <- c('index', quantiles)
write.csv(
    quantile_estimates,
    file='092-run_competitor2_on_synthetic_quantiles.csv',
    quote=FALSE,
    row.names=FALSE
)
