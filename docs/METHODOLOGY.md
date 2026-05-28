# Methodology

## What is Causal Impact?

Causal impact analysis estimates the effect of an intervention on a time series by comparing what actually happened to what would have happened without the intervention (the "counterfactual").

## ARIMA Interrupted Time Series (ITS)

### How it works

1. **Pre-intervention period**: Fit an ARIMA model on the data before the intervention
2. **Post-intervention forecast**: Use the fitted model to predict what would have happened without the intervention
3. **Effect estimation**: Compare the actual post-intervention data to the counterfactual prediction
4. **Uncertainty**: Compute confidence intervals using the forecast variance

### When to use
- You have a single time series with a clearly defined intervention date
- You have at least 30 data points before the intervention
- The pre-intervention trend is relatively stable

### Assumptions
- The pre-intervention relationship between the metric and time remains stable
- No other major events occurred at the same time as the intervention
- The time series has sufficient data (30+ points)

## Bayesian Structural Time Series (BSTS)

### How it works
1. **Local linear trend**: Captures gradual changes in the level and trend
2. **Regression on covariates**: Controls for external factors
3. **Bayesian posterior**: Uses MCMC sampling to estimate uncertainty
4. **Counterfactual**: Projects the posterior predictive distribution forward

### When to use
- You want probabilistic uncertainty estimates
- You have control variables (other time series that weren't affected)
- You can tolerate slower computation

## Sensitivity Analysis (Placebo Tests)

### How it works
1. Run the same analysis at multiple fake intervention dates
2. Compare the real effect to the distribution of placebo effects
3. If the real effect is extreme (outside 95% of placebo effects), the result is robust

### Interpretation
- **p < 0.05**: Real effect is unlikely to be due to chance
- **Real effect is extreme**: Result is robust to model specification

## Limitations

- Cannot prove causation, only estimate causal effect under assumptions
- Results are sensitive to the choice of intervention date
- Confounding variables may bias the estimate
- Short time series (< 30 points) produce unreliable results

## References

1. Brodersen et al. "Inferring Causal Impact Using Bayesian Structural Time-Series Models." *Annals of Applied Statistics*, 2015.
2. Lopez Bernal et al. "Interrupted time series regression for the evaluation of public health interventions: a tutorial." *International Journal of Epidemiology*, 2017.
3. Runge et al. "Causal Inference for Time Series." *Nature Reviews Earth & Environment*, 2023.
