# 📊 Bayesian A/B Test Calculator

A production-ready Bayesian A/B testing calculator that provides actionable insights beyond traditional p-values.

**[Live Demo →](https://your-app-url.streamlit.app)**

![Bayesian A/B Test Calculator Screenshot](screenshot.png)

## Why Bayesian A/B Testing?

Traditional frequentist A/B tests give you a p-value and ask: *"Is the result statistically significant?"* 

But what you actually want to know is:
- **"What's the probability my treatment is better?"**
- **"How much better is it likely to be?"**
- **"What's the risk if I make the wrong decision?"**

Bayesian methods answer these questions directly.

| What You Want to Know | Frequentist Answer | Bayesian Answer |
|----------------------|-------------------|-----------------|
| Is treatment better? | "Reject null at α=0.05" | "92% probability treatment is better" |
| By how much? | Point estimate ± CI | Full posterior distribution |
| Should I ship it? | "Significant" ≠ "Ship it" | Expected loss quantifies decision risk |
| Can I peek at results? | No! Inflates false positives | Yes, valid at any sample size |

## Features

- **Posterior Distributions:** Visualize uncertainty in conversion rates for both variants
- **Probability of Winning:** Direct answer to "what's the chance treatment beats control?"
- **Credible Intervals:** 95% probability the true lift falls within this range
- **Expected Loss Analysis:** Quantify the risk of making the wrong decision
- **Interactive Priors:** Incorporate prior knowledge (or use uninformative priors)
- **Clear Recommendations:** Actionable guidance based on your results

## The Math

This calculator uses the **Beta-Binomial conjugate model**:

```
Prior:      p ~ Beta(α, β)
Likelihood: conversions ~ Binomial(visitors, p)  
Posterior:  p | data ~ Beta(α + conversions, β + visitors - conversions)
```

The conjugate relationship means we get exact posteriors without MCMC sampling, making the calculator fast and deterministic.

### Key Metrics

**P(Treatment > Control)**

Computed via Monte Carlo sampling from the posteriors:

```python
samples_control = control_posterior.rvs(100_000)
samples_treatment = treatment_posterior.rvs(100_000)
prob_treatment_better = np.mean(samples_treatment > samples_control)
```

**Expected Lift**

The mean of the relative lift distribution:

```python
lift_samples = (samples_treatment - samples_control) / samples_control
expected_lift = np.mean(lift_samples)
```

**Expected Loss (Risk)**

The average "regret" if you make the wrong decision:

```python
# Risk of choosing treatment when control is actually better
risk_treatment = np.mean(np.maximum(samples_control - samples_treatment, 0))

# Risk of choosing control when treatment is actually better  
risk_control = np.mean(np.maximum(samples_treatment - samples_control, 0))
```

## When to Ship

A common decision framework:

| P(Treatment > Control) | Recommendation |
|------------------------|----------------|
| > 95% | ✅ Ship with confidence |
| 90-95% | Consider business context |
| 50-90% | Gather more data |
| < 50% | Treatment likely worse |

But also consider **expected loss**—even with 85% confidence, if the expected loss is tiny (0.001 pp), shipping may be the right business decision.

## Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/bayesian-ab-calculator.git
cd bayesian-ab-calculator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deployment

This app is designed for [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push to GitHub
2. Connect your repo to Streamlit Cloud
3. Deploy

No additional configuration needed.

## Extending the Calculator

### Adding PyMC for Complex Models

For more complex experiments (e.g., continuous metrics, hierarchical models), you can extend with PyMC:

```python
import pymc as pm

with pm.Model() as model:
    # Priors
    p_control = pm.Beta('p_control', alpha=1, beta=1)
    p_treatment = pm.Beta('p_treatment', alpha=1, beta=1)
    
    # Likelihood
    obs_control = pm.Binomial('obs_control', n=n_control, p=p_control, observed=conv_control)
    obs_treatment = pm.Binomial('obs_treatment', n=n_treatment, p=p_treatment, observed=conv_treatment)
    
    # Derived quantities
    lift = pm.Deterministic('lift', (p_treatment - p_control) / p_control)
    
    # Sample
    trace = pm.sample(2000, return_inferencedata=True)
```

### Continuous Metrics

For revenue or time-on-site, use a Normal-Normal or Student-t model:

```python
# Student-t for robustness to outliers
with pm.Model() as model:
    mu_control = pm.Normal('mu_control', mu=prior_mean, sigma=prior_std)
    mu_treatment = pm.Normal('mu_treatment', mu=prior_mean, sigma=prior_std)
    
    sigma = pm.HalfNormal('sigma', sigma=1)
    nu = pm.Exponential('nu', 1/30) + 1  # Degrees of freedom
    
    obs_control = pm.StudentT('obs_control', nu=nu, mu=mu_control, sigma=sigma, observed=data_control)
    obs_treatment = pm.StudentT('obs_treatment', nu=nu, mu=mu_treatment, sigma=sigma, observed=data_treatment)
```

## References

- Kruschke, J. K. (2013). Bayesian estimation supersedes the t test. *Journal of Experimental Psychology*
- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)
- [VWO's Bayesian A/B Testing Whitepaper](https://vwo.com/downloads/VWO_SmartStats_technical_whitepaper.pdf)
- [Evan Miller's Bayesian A/B Testing Blog Post](https://www.evanmiller.org/bayesian-ab-testing.html)

## About

Built by **Michael Johnson** — a data scientist specializing in experimentation, causal inference, and marketing measurement.

- [LinkedIn](https://linkedin.com/in/data-arts-data-science)
- [Email](mailto:mchl.dvd.jhnsn@gmail.com)

This tool demonstrates the Bayesian experimentation methods I've implemented in production systems serving 100M+ users, including frameworks for A/B testing, incrementality measurement, and model validation.

## License

MIT License - feel free to use, modify, and distribute.
