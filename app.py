"""
Bayesian A/B Test Calculator
A portfolio demonstration of Bayesian experimentation methods.

Author: Michael Johnson
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Optional


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExperimentData:
    """Container for experiment data."""
    control_visitors: int
    control_conversions: int
    treatment_visitors: int
    treatment_conversions: int
    
    @property
    def control_rate(self) -> float:
        return self.control_conversions / self.control_visitors if self.control_visitors > 0 else 0
    
    @property
    def treatment_rate(self) -> float:
        return self.treatment_conversions / self.treatment_visitors if self.treatment_visitors > 0 else 0
    
    @property
    def observed_lift(self) -> float:
        if self.control_rate == 0:
            return 0
        return (self.treatment_rate - self.control_rate) / self.control_rate


@dataclass
class BayesianResults:
    """Container for Bayesian analysis results."""
    control_posterior: stats.rv_continuous
    treatment_posterior: stats.rv_continuous
    prob_treatment_better: float
    expected_lift: float
    lift_ci_lower: float
    lift_ci_upper: float
    absolute_diff_mean: float
    absolute_diff_ci: Tuple[float, float]
    risk_choosing_treatment: float
    risk_choosing_control: float
    samples_control: np.ndarray
    samples_treatment: np.ndarray


@dataclass
class SampleSizeResult:
    """Container for sample size calculation results."""
    sample_size_per_variant: int
    total_sample_size: int
    expected_runtime_days: Optional[float]
    power_at_size: float
    sample_sizes: np.ndarray
    powers: np.ndarray


@dataclass
class RevenueData:
    """Container for revenue experiment data (BTYD-style)."""
    control_visitors: int
    control_purchasers: int  # Number of unique customers who purchased
    control_transactions: int  # Total number of transactions (>= purchasers)
    control_total_revenue: float
    treatment_visitors: int
    treatment_purchasers: int
    treatment_transactions: int
    treatment_total_revenue: float
    
    @property
    def control_purchase_rate(self) -> float:
        """P(at least one purchase)"""
        return self.control_purchasers / self.control_visitors if self.control_visitors > 0 else 0
    
    @property
    def treatment_purchase_rate(self) -> float:
        return self.treatment_purchasers / self.treatment_visitors if self.treatment_visitors > 0 else 0
    
    @property
    def control_avg_transactions_per_buyer(self) -> float:
        """Average frequency among buyers"""
        return self.control_transactions / self.control_purchasers if self.control_purchasers > 0 else 0
    
    @property
    def treatment_avg_transactions_per_buyer(self) -> float:
        return self.treatment_transactions / self.treatment_purchasers if self.treatment_purchasers > 0 else 0
    
    @property
    def control_avg_order_value(self) -> float:
        """Average value per transaction"""
        return self.control_total_revenue / self.control_transactions if self.control_transactions > 0 else 0
    
    @property
    def treatment_avg_order_value(self) -> float:
        return self.treatment_total_revenue / self.treatment_transactions if self.treatment_transactions > 0 else 0
    
    @property
    def control_revenue_per_visitor(self) -> float:
        return self.control_total_revenue / self.control_visitors if self.control_visitors > 0 else 0
    
    @property
    def treatment_revenue_per_visitor(self) -> float:
        return self.treatment_total_revenue / self.treatment_visitors if self.treatment_visitors > 0 else 0
    
    @property
    def control_clv_per_buyer(self) -> float:
        """Total value per buyer = frequency × AOV"""
        return self.control_total_revenue / self.control_purchasers if self.control_purchasers > 0 else 0
    
    @property
    def treatment_clv_per_buyer(self) -> float:
        return self.treatment_total_revenue / self.treatment_purchasers if self.treatment_purchasers > 0 else 0


@dataclass
class RevenueResults:
    """Container for revenue analysis results."""
    # Purchase rate posteriors (Beta)
    control_purchase_rate_samples: np.ndarray
    treatment_purchase_rate_samples: np.ndarray
    prob_treatment_purchase_rate_better: float
    
    # Average order value posteriors (Gamma)
    control_aov_samples: np.ndarray
    treatment_aov_samples: np.ndarray
    prob_treatment_aov_better: float
    
    # Revenue per visitor (combined hurdle model)
    control_rpv_samples: np.ndarray
    treatment_rpv_samples: np.ndarray
    prob_treatment_rpv_better: float
    
    # Summary statistics
    expected_rpv_lift: float
    rpv_lift_ci: Tuple[float, float]
    expected_rpv_diff: float
    rpv_diff_ci: Tuple[float, float]
    
    # Risk analysis
    risk_choosing_treatment: float
    risk_choosing_control: float


# =============================================================================
# Bayesian Analysis Functions
# =============================================================================

def compute_bayesian_analysis(
    data: ExperimentData,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100_000,
    credible_level: float = 0.95
) -> BayesianResults:
    """
    Perform Bayesian analysis using Beta-Binomial conjugate model.
    
    The Beta-Binomial model is the gold standard for conversion rate experiments:
    - Prior: Beta(α, β) represents our prior belief about conversion rates
    - Likelihood: Binomial(n, p) for observed conversions
    - Posterior: Beta(α + conversions, β + non-conversions)
    
    Args:
        data: Experiment data
        prior_alpha: Beta prior α parameter (default 1 = uniform prior)
        prior_beta: Beta prior β parameter (default 1 = uniform prior)
        n_samples: Number of Monte Carlo samples for computing metrics
        credible_level: Credible interval level (default 95%)
    
    Returns:
        BayesianResults with posteriors and key metrics
    """
    # Compute posterior parameters (conjugate update)
    control_alpha = prior_alpha + data.control_conversions
    control_beta = prior_beta + (data.control_visitors - data.control_conversions)
    
    treatment_alpha = prior_alpha + data.treatment_conversions
    treatment_beta = prior_beta + (data.treatment_visitors - data.treatment_conversions)
    
    # Create posterior distributions
    control_posterior = stats.beta(control_alpha, control_beta)
    treatment_posterior = stats.beta(treatment_alpha, treatment_beta)
    
    # Monte Carlo sampling for derived metrics
    rng = np.random.default_rng(42)  # Reproducibility without polluting global state
    samples_control = control_posterior.rvs(n_samples, random_state=rng)
    samples_treatment = treatment_posterior.rvs(n_samples, random_state=rng)
    
    # Probability that treatment is better
    prob_treatment_better = np.mean(samples_treatment > samples_control)
    
    # Relative lift distribution: (treatment - control) / control
    # Handle edge case where control could be 0
    with np.errstate(divide='ignore', invalid='ignore'):
        lift_samples = (samples_treatment - samples_control) / samples_control
        lift_samples = lift_samples[np.isfinite(lift_samples)]
    
    if len(lift_samples) > 0:
        expected_lift = np.mean(lift_samples)
        alpha_tail = (1 - credible_level) / 2
        lift_ci_lower = np.percentile(lift_samples, alpha_tail * 100)
        lift_ci_upper = np.percentile(lift_samples, (1 - alpha_tail) * 100)
    else:
        expected_lift = 0
        lift_ci_lower = 0
        lift_ci_upper = 0
    
    # Absolute difference
    diff_samples = samples_treatment - samples_control
    absolute_diff_mean = np.mean(diff_samples)
    alpha_tail = (1 - credible_level) / 2
    absolute_diff_ci = (
        np.percentile(diff_samples, alpha_tail * 100),
        np.percentile(diff_samples, (1 - alpha_tail) * 100)
    )
    
    # Expected loss (risk) analysis
    # Risk of choosing treatment = E[max(control - treatment, 0)]
    # Risk of choosing control = E[max(treatment - control, 0)]
    risk_choosing_treatment = np.mean(np.maximum(samples_control - samples_treatment, 0))
    risk_choosing_control = np.mean(np.maximum(samples_treatment - samples_control, 0))
    
    return BayesianResults(
        control_posterior=control_posterior,
        treatment_posterior=treatment_posterior,
        prob_treatment_better=prob_treatment_better,
        expected_lift=expected_lift,
        lift_ci_lower=lift_ci_lower,
        lift_ci_upper=lift_ci_upper,
        absolute_diff_mean=absolute_diff_mean,
        absolute_diff_ci=absolute_diff_ci,
        risk_choosing_treatment=risk_choosing_treatment,
        risk_choosing_control=risk_choosing_control,
        samples_control=samples_control,
        samples_treatment=samples_treatment
    )


def compute_sample_size(
    baseline_rate: float,
    mde_relative: float,
    target_power: float = 0.80,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_simulations: int = 1000,
    daily_traffic: Optional[int] = None
) -> SampleSizeResult:
    """
    Compute required sample size using Bayesian simulation.
    
    This uses Monte Carlo simulation to determine the sample size needed
    to achieve a target probability of correctly detecting a true effect.
    
    Args:
        baseline_rate: Expected conversion rate for control (e.g., 0.05 for 5%)
        mde_relative: Minimum detectable effect as relative lift (e.g., 0.10 for 10% lift)
        target_power: Desired probability of detecting the effect (default 80%)
        prior_alpha: Beta prior α parameter
        prior_beta: Beta prior β parameter
        n_simulations: Number of Monte Carlo simulations per sample size
        daily_traffic: Optional daily visitors per variant for runtime estimate
    
    Returns:
        SampleSizeResult with recommended sample size and power curve
    """
    treatment_rate = baseline_rate * (1 + mde_relative)
    
    # Sample sizes to evaluate (per variant)
    sample_sizes = np.array([
        100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000,
        7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000
    ])
    
    rng = np.random.default_rng(42)
    powers = []
    
    for n in sample_sizes:
        # Simulate experiments
        detections = 0
        for _ in range(n_simulations):
            # Simulate observed data under the alternative hypothesis
            control_conv = rng.binomial(n, baseline_rate)
            treatment_conv = rng.binomial(n, treatment_rate)
            
            # Compute posteriors
            control_alpha_post = prior_alpha + control_conv
            control_beta_post = prior_beta + (n - control_conv)
            treatment_alpha_post = prior_alpha + treatment_conv
            treatment_beta_post = prior_beta + (n - treatment_conv)
            
            # Sample from posteriors to compute P(treatment > control)
            control_samples = stats.beta(control_alpha_post, control_beta_post).rvs(10000, random_state=rng)
            treatment_samples = stats.beta(treatment_alpha_post, treatment_beta_post).rvs(10000, random_state=rng)
            
            prob_better = np.mean(treatment_samples > control_samples)
            
            # Detection = P(treatment > control) >= 95%
            if prob_better >= 0.95:
                detections += 1
        
        power = detections / n_simulations
        powers.append(power)
    
    powers = np.array(powers)
    
    # Find minimum sample size achieving target power
    achieved_idx = np.where(powers >= target_power)[0]
    if len(achieved_idx) > 0:
        recommended_n = int(sample_sizes[achieved_idx[0]])
        power_at_size = powers[achieved_idx[0]]
    else:
        # If target not achieved, recommend largest tested
        recommended_n = int(sample_sizes[-1])
        power_at_size = powers[-1]
    
    # Calculate expected runtime
    runtime_days = None
    if daily_traffic is not None and daily_traffic > 0:
        runtime_days = (recommended_n * 2) / daily_traffic
    
    return SampleSizeResult(
        sample_size_per_variant=recommended_n,
        total_sample_size=recommended_n * 2,
        expected_runtime_days=runtime_days,
        power_at_size=power_at_size,
        sample_sizes=sample_sizes,
        powers=powers
    )


def compute_revenue_analysis(
    data: RevenueData,
    prior_purchase_alpha: float = 1.0,
    prior_purchase_beta: float = 1.0,
    prior_aov_shape: float = 1.0,
    prior_aov_rate: float = 0.01,
    n_samples: int = 100_000,
    credible_level: float = 0.95
) -> RevenueResults:
    """
    Perform Bayesian revenue analysis using a BTYD-style hurdle model.
    
    This combines two models following the "Buy Till You Die" framework:
    1. Beta-Binomial for purchase probability (hurdle component)
    2. Gamma-Gamma for average order value (Fader/Hardie 2005 specification)
    
    Revenue per visitor = P(purchase) × E[spend | purchase]
    
    The Gamma-Gamma model (Fader & Hardie, 2005) assumes:
    - Individual transaction values z ~ Gamma(p, ν) where ν varies per customer
    - ν ~ Gamma(q, γ) across the population (heterogeneity in spend)
    - This captures that some customers consistently spend more than others
    
    For aggregate data, we estimate the population-level expected spend using
    conjugate Bayesian updates on the Gamma-Gamma model parameters.
    
    Args:
        data: Revenue experiment data with visitors, purchasers, transactions, revenue
        prior_purchase_alpha: Beta prior α for purchase rate
        prior_purchase_beta: Beta prior β for purchase rate  
        prior_aov_shape: Gamma-Gamma prior shape (q) - controls heterogeneity
        prior_aov_rate: Gamma-Gamma prior rate (γ) - scales expected spend
        n_samples: Number of Monte Carlo samples
        credible_level: Credible interval level
    
    Returns:
        RevenueResults with posteriors and key metrics
    """
    rng = np.random.default_rng(42)
    
    # =========================================================================
    # Part 1: Beta-Binomial for Purchase Rate (Hurdle Component)
    # =========================================================================
    # Posterior: Beta(α + purchasers, β + non-purchasers)
    
    control_purchase_alpha = prior_purchase_alpha + data.control_purchasers
    control_purchase_beta = prior_purchase_beta + (data.control_visitors - data.control_purchasers)
    
    treatment_purchase_alpha = prior_purchase_alpha + data.treatment_purchasers
    treatment_purchase_beta = prior_purchase_beta + (data.treatment_visitors - data.treatment_purchasers)
    
    control_purchase_rate_samples = stats.beta(
        control_purchase_alpha, control_purchase_beta
    ).rvs(n_samples, random_state=rng)
    
    treatment_purchase_rate_samples = stats.beta(
        treatment_purchase_alpha, treatment_purchase_beta
    ).rvs(n_samples, random_state=rng)
    
    prob_treatment_purchase_rate_better = np.mean(
        treatment_purchase_rate_samples > control_purchase_rate_samples
    )
    
    # =========================================================================
    # Part 2: Gamma-Gamma Model for Average Order Value (Fader/Hardie 2005)
    # =========================================================================
    # The Gamma-Gamma model captures heterogeneity in customer spend:
    #
    # Individual transaction value: z_ij ~ Gamma(p, ν_i)  
    # Customer-level spend rate: ν_i ~ Gamma(q, γ)
    #
    # For a customer with x transactions and average spend m_x:
    # E[M | x, m_x, p, q, γ] = (q*γ + x*m_x) / (q + x - 1)  when p=1
    #
    # For aggregate data, we use conjugate Bayesian updates:
    # - n = number of transactions (information about mean)
    # - S = total spend
    # - m = S/n = observed average transaction value
    #
    # Posterior for population mean spend:
    # shape' = q + n (more transactions = more certainty)
    # rate' = γ + S (total spend updates the rate)
    # Expected value = shape' / rate' → shrinks toward prior with small n
    
    # Control AOV posterior (using transactions, not just purchasers)
    if data.control_transactions > 0:
        n_ctrl = data.control_transactions
        S_ctrl = data.control_total_revenue
        
        # Gamma-Gamma posterior parameters
        q_post_ctrl = prior_aov_shape + n_ctrl
        gamma_post_ctrl = prior_aov_rate + S_ctrl
        
        # The posterior for expected spend per transaction is Gamma
        # with shape = q_post and rate = gamma_post / n_ctrl
        # This gives E[μ] = (q + n) / (γ + S) * S/n ≈ observed mean with shrinkage
        control_aov_samples = stats.gamma(
            a=q_post_ctrl,
            scale=S_ctrl / (gamma_post_ctrl * (q_post_ctrl - 1)) if q_post_ctrl > 1 else S_ctrl / gamma_post_ctrl
        ).rvs(n_samples, random_state=rng)
        
        # Ensure positive samples and reasonable bounds
        control_aov_samples = np.maximum(control_aov_samples, 0.01)
    else:
        control_aov_samples = stats.gamma(
            a=prior_aov_shape,
            scale=1/prior_aov_rate
        ).rvs(n_samples, random_state=rng)
    
    # Treatment AOV posterior
    if data.treatment_transactions > 0:
        n_treat = data.treatment_transactions
        S_treat = data.treatment_total_revenue
        
        q_post_treat = prior_aov_shape + n_treat
        gamma_post_treat = prior_aov_rate + S_treat
        
        treatment_aov_samples = stats.gamma(
            a=q_post_treat,
            scale=S_treat / (gamma_post_treat * (q_post_treat - 1)) if q_post_treat > 1 else S_treat / gamma_post_treat
        ).rvs(n_samples, random_state=rng)
        
        treatment_aov_samples = np.maximum(treatment_aov_samples, 0.01)
    else:
        treatment_aov_samples = stats.gamma(
            a=prior_aov_shape,
            scale=1/prior_aov_rate
        ).rvs(n_samples, random_state=rng)
    
    prob_treatment_aov_better = np.mean(treatment_aov_samples > control_aov_samples)
    
    # =========================================================================
    # Part 3: Combined Revenue Per Visitor (Hurdle Model)
    # =========================================================================
    # RPV = P(purchase) × E[AOV | purchase]
    
    control_rpv_samples = control_purchase_rate_samples * control_aov_samples
    treatment_rpv_samples = treatment_purchase_rate_samples * treatment_aov_samples
    
    prob_treatment_rpv_better = np.mean(treatment_rpv_samples > control_rpv_samples)
    
    # =========================================================================
    # Part 4: Summary Statistics
    # =========================================================================
    
    # Relative lift in RPV
    with np.errstate(divide='ignore', invalid='ignore'):
        rpv_lift_samples = (treatment_rpv_samples - control_rpv_samples) / control_rpv_samples
        rpv_lift_samples_clean = rpv_lift_samples[np.isfinite(rpv_lift_samples)]
    
    if len(rpv_lift_samples_clean) > 0:
        expected_rpv_lift = np.mean(rpv_lift_samples_clean)
        alpha_tail = (1 - credible_level) / 2
        rpv_lift_ci = (
            np.percentile(rpv_lift_samples_clean, alpha_tail * 100),
            np.percentile(rpv_lift_samples_clean, (1 - alpha_tail) * 100)
        )
    else:
        expected_rpv_lift = 0
        rpv_lift_ci = (0, 0)
    
    # Absolute difference in RPV
    rpv_diff_samples = treatment_rpv_samples - control_rpv_samples
    expected_rpv_diff = np.mean(rpv_diff_samples)
    alpha_tail = (1 - credible_level) / 2
    rpv_diff_ci = (
        np.percentile(rpv_diff_samples, alpha_tail * 100),
        np.percentile(rpv_diff_samples, (1 - alpha_tail) * 100)
    )
    
    # Risk analysis (in dollar terms)
    risk_choosing_treatment = np.mean(np.maximum(control_rpv_samples - treatment_rpv_samples, 0))
    risk_choosing_control = np.mean(np.maximum(treatment_rpv_samples - control_rpv_samples, 0))
    
    return RevenueResults(
        control_purchase_rate_samples=control_purchase_rate_samples,
        treatment_purchase_rate_samples=treatment_purchase_rate_samples,
        prob_treatment_purchase_rate_better=prob_treatment_purchase_rate_better,
        control_aov_samples=control_aov_samples,
        treatment_aov_samples=treatment_aov_samples,
        prob_treatment_aov_better=prob_treatment_aov_better,
        control_rpv_samples=control_rpv_samples,
        treatment_rpv_samples=treatment_rpv_samples,
        prob_treatment_rpv_better=prob_treatment_rpv_better,
        expected_rpv_lift=expected_rpv_lift,
        rpv_lift_ci=rpv_lift_ci,
        expected_rpv_diff=expected_rpv_diff,
        rpv_diff_ci=rpv_diff_ci,
        risk_choosing_treatment=risk_choosing_treatment,
        risk_choosing_control=risk_choosing_control
    )


# =============================================================================
# Visualization Functions
# =============================================================================

def create_posterior_plot(results: BayesianResults, data: ExperimentData) -> go.Figure:
    """Create interactive posterior distribution plot."""
    
    # Create x range for plotting PDFs
    all_samples = np.concatenate([results.samples_control, results.samples_treatment])
    x_min = max(0, np.percentile(all_samples, 0.1) - 0.01)
    x_max = min(1, np.percentile(all_samples, 99.9) + 0.01)
    x = np.linspace(x_min, x_max, 500)
    
    fig = go.Figure()
    
    # Control posterior
    y_control = results.control_posterior.pdf(x)
    fig.add_trace(go.Scatter(
        x=x, y=y_control,
        mode='lines',
        name='Control',
        line=dict(color='#636EFA', width=3),
        fill='tozeroy',
        fillcolor='rgba(99, 110, 250, 0.2)'
    ))
    
    # Treatment posterior
    y_treatment = results.treatment_posterior.pdf(x)
    fig.add_trace(go.Scatter(
        x=x, y=y_treatment,
        mode='lines',
        name='Treatment',
        line=dict(color='#00CC96', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.2)'
    ))
    
    # Add vertical lines for observed rates
    fig.add_vline(
        x=data.control_rate, 
        line_dash="dash", 
        line_color="#636EFA",
        annotation_text=f"Control: {data.control_rate:.2%}",
        annotation_position="top left"
    )
    fig.add_vline(
        x=data.treatment_rate, 
        line_dash="dash", 
        line_color="#00CC96",
        annotation_text=f"Treatment: {data.treatment_rate:.2%}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="Posterior Distributions of Conversion Rates",
        xaxis_title="Conversion Rate",
        yaxis_title="Probability Density",
        xaxis_tickformat='.1%',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=400
    )
    
    return fig


def create_lift_distribution_plot(results: BayesianResults) -> go.Figure:
    """Create lift distribution plot with credible interval."""
    
    # Compute lift samples
    with np.errstate(divide='ignore', invalid='ignore'):
        lift_samples = (results.samples_treatment - results.samples_control) / results.samples_control
        lift_samples = lift_samples[np.isfinite(lift_samples)]
    
    fig = go.Figure()
    
    # Histogram of lift
    fig.add_trace(go.Histogram(
        x=lift_samples,
        nbinsx=100,
        name='Relative Lift Distribution',
        marker_color='#AB63FA',
        opacity=0.7,
        histnorm='probability density'
    ))
    
    # Add vertical line at 0 (no effect)
    fig.add_vline(
        x=0, 
        line_dash="solid", 
        line_color="red",
        line_width=2,
        annotation_text="No Effect",
        annotation_position="top"
    )
    
    # Add credible interval
    fig.add_vrect(
        x0=results.lift_ci_lower,
        x1=results.lift_ci_upper,
        fillcolor="rgba(171, 99, 250, 0.2)",
        line_width=0,
        annotation_text="95% Credible Interval",
        annotation_position="top left"
    )
    
    # Add expected lift line
    fig.add_vline(
        x=results.expected_lift,
        line_dash="dash",
        line_color="#AB63FA",
        line_width=2,
        annotation_text=f"Expected: {results.expected_lift:+.1%}",
        annotation_position="bottom"
    )
    
    fig.update_layout(
        title="Distribution of Relative Lift (Treatment vs Control)",
        xaxis_title="Relative Lift",
        yaxis_title="Probability Density",
        xaxis_tickformat='+.0%',
        showlegend=False,
        height=400
    )
    
    return fig


def create_risk_plot(results: BayesianResults) -> go.Figure:
    """Create risk analysis visualization."""
    
    fig = go.Figure()
    
    categories = ['Choose Treatment', 'Choose Control']
    risks = [
        results.risk_choosing_treatment * 100,  # Convert to percentage points
        results.risk_choosing_control * 100
    ]
    colors = ['#00CC96', '#636EFA']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=risks,
        marker_color=colors,
        text=[f'{r:.3f} pp' for r in risks],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Expected Loss (Risk) Analysis",
        yaxis_title="Expected Loss (percentage points)",
        showlegend=False,
        height=300
    )
    
    return fig


def create_power_curve_plot(result: SampleSizeResult, target_power: float) -> go.Figure:
    """Create power curve visualization."""
    
    fig = go.Figure()
    
    # Power curve
    fig.add_trace(go.Scatter(
        x=result.sample_sizes,
        y=result.powers * 100,
        mode='lines+markers',
        name='Statistical Power',
        line=dict(color='#636EFA', width=3),
        marker=dict(size=8)
    ))
    
    # Target power line
    fig.add_hline(
        y=target_power * 100,
        line_dash="dash",
        line_color="#00CC96",
        annotation_text=f"Target: {target_power:.0%}",
        annotation_position="right"
    )
    
    # Recommended sample size marker
    fig.add_vline(
        x=result.sample_size_per_variant,
        line_dash="dot",
        line_color="#EF553B",
        annotation_text=f"Recommended: {result.sample_size_per_variant:,}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Power Curve: Sample Size vs Detection Probability",
        xaxis_title="Sample Size (per variant)",
        yaxis_title="Power (%)",
        xaxis_type="log",
        yaxis_range=[0, 105],
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_revenue_decomposition_plot(results: RevenueResults, data: RevenueData) -> go.Figure:
    """Create a 3-panel plot showing purchase rate, AOV, and RPV posteriors."""
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Purchase Rate",
            "Avg Order Value (AOV)",
            "Revenue Per Visitor (RPV)"
        ),
        horizontal_spacing=0.08
    )
    
    # Panel 1: Purchase Rate (Beta posteriors)
    fig.add_trace(go.Histogram(
        x=results.control_purchase_rate_samples,
        name='Control',
        marker_color='#636EFA',
        opacity=0.6,
        histnorm='probability density',
        nbinsx=50
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=results.treatment_purchase_rate_samples,
        name='Treatment',
        marker_color='#00CC96',
        opacity=0.6,
        histnorm='probability density',
        nbinsx=50
    ), row=1, col=1)
    
    # Panel 2: AOV (Gamma posteriors)
    fig.add_trace(go.Histogram(
        x=results.control_aov_samples,
        name='Control',
        marker_color='#636EFA',
        opacity=0.6,
        histnorm='probability density',
        nbinsx=50,
        showlegend=False
    ), row=1, col=2)
    
    fig.add_trace(go.Histogram(
        x=results.treatment_aov_samples,
        name='Treatment',
        marker_color='#00CC96',
        opacity=0.6,
        histnorm='probability density',
        nbinsx=50,
        showlegend=False
    ), row=1, col=2)
    
    # Panel 3: RPV (Combined hurdle model)
    fig.add_trace(go.Histogram(
        x=results.control_rpv_samples,
        name='Control',
        marker_color='#636EFA',
        opacity=0.6,
        histnorm='probability density',
        nbinsx=50,
        showlegend=False
    ), row=1, col=3)
    
    fig.add_trace(go.Histogram(
        x=results.treatment_rpv_samples,
        name='Treatment',
        marker_color='#00CC96',
        opacity=0.6,
        histnorm='probability density',
        nbinsx=50,
        showlegend=False
    ), row=1, col=3)
    
    # Update axes
    fig.update_xaxes(title_text="Rate", tickformat='.1%', row=1, col=1)
    fig.update_xaxes(title_text="$ per order", tickprefix="$", row=1, col=2)
    fig.update_xaxes(title_text="$ per visitor", tickprefix="$", row=1, col=3)
    
    fig.update_layout(
        height=350,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        barmode='overlay'
    )
    
    return fig


def create_rpv_lift_plot(results: RevenueResults) -> go.Figure:
    """Create RPV lift distribution plot."""
    
    # Compute lift samples
    with np.errstate(divide='ignore', invalid='ignore'):
        lift_samples = (results.treatment_rpv_samples - results.control_rpv_samples) / results.control_rpv_samples
        lift_samples = lift_samples[np.isfinite(lift_samples)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=lift_samples,
        nbinsx=100,
        name='RPV Lift Distribution',
        marker_color='#AB63FA',
        opacity=0.7,
        histnorm='probability density'
    ))
    
    # No effect line
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color="red",
        line_width=2,
        annotation_text="No Effect",
        annotation_position="top"
    )
    
    # Credible interval
    fig.add_vrect(
        x0=results.rpv_lift_ci[0],
        x1=results.rpv_lift_ci[1],
        fillcolor="rgba(171, 99, 250, 0.2)",
        line_width=0,
        annotation_text="95% CI",
        annotation_position="top left"
    )
    
    # Expected lift
    fig.add_vline(
        x=results.expected_rpv_lift,
        line_dash="dash",
        line_color="#AB63FA",
        line_width=2,
        annotation_text=f"Expected: {results.expected_rpv_lift:+.1%}",
        annotation_position="bottom"
    )
    
    fig.update_layout(
        title="Distribution of Revenue Per Visitor Lift",
        xaxis_title="Relative Lift in RPV",
        yaxis_title="Probability Density",
        xaxis_tickformat='+.0%',
        showlegend=False,
        height=400
    )
    
    return fig


def create_revenue_risk_plot(results: RevenueResults) -> go.Figure:
    """Create revenue risk analysis visualization."""
    
    fig = go.Figure()
    
    categories = ['Choose Treatment', 'Choose Control']
    risks = [results.risk_choosing_treatment, results.risk_choosing_control]
    colors = ['#00CC96', '#636EFA']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=risks,
        marker_color=colors,
        text=[f'${r:.4f}' for r in risks],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Expected Loss (Risk) in Revenue",
        yaxis_title="Expected Loss ($ per visitor)",
        yaxis_tickprefix="$",
        showlegend=False,
        height=300
    )
    
    return fig


# =============================================================================
# Streamlit App
# =============================================================================

def main():
    st.set_page_config(
        page_title="Bayesian A/B Test Calculator",
        page_icon="📊",
        layout="wide"
    )
    
    # Header
    st.title("📊 Bayesian A/B Test Calculator")
    st.markdown("""
    **Make better decisions with uncertainty quantification.**
    
    Unlike traditional frequentist tests that only give you p-values, Bayesian analysis tells you:
    - The *probability* that your treatment is better
    - The *expected* lift with credible intervals  
    - The *risk* of making the wrong decision
    """)
    
    # Main tabs
    tab_analyze, tab_revenue, tab_sample_size = st.tabs([
        "📈 Conversion Analysis", 
        "💰 Revenue Analysis",
        "🎯 Sample Size Calculator"
    ])
    
    # ==========================================================================
    # Tab 1: Analyze Results (Conversion Rate)
    # ==========================================================================
    with tab_analyze:
        analyze_results_tab()
    
    # ==========================================================================
    # Tab 2: Revenue Analysis (Hurdle Model)
    # ==========================================================================
    with tab_revenue:
        revenue_analysis_tab()
    
    # ==========================================================================
    # Tab 3: Sample Size Calculator
    # ==========================================================================
    with tab_sample_size:
        sample_size_calculator_tab()
    
    # Footer
    st.divider()
    st.caption("""
    **About this tool:** Built to demonstrate Bayesian experimentation methods used in 
    production A/B testing systems. The same statistical framework powers experimentation 
    platforms at companies like Netflix, Spotify, and Booking.com.
    
    [GitHub](https://github.com/yourusername/bayesian-ab-calculator) | 
    [LinkedIn](https://linkedin.com/in/data-arts-data-science) |
    Built by Michael Johnson
    """)


def analyze_results_tab():
    """A/B test results analyzer tab."""
    
    st.markdown("""
    **Analyze binary conversion experiments.**
    
    Enter your visitor and conversion counts for each variant.
    """)
    
    st.divider()
    
    # Input section - inline columns like Revenue Analysis
    col_ctrl, col_treat = st.columns(2)
    
    with col_ctrl:
        st.subheader("🔵 Control Group")
        
        control_visitors = st.number_input(
            "Visitors", 
            min_value=1, 
            value=10000,
            step=100,
            key="conv_control_visitors",
            help="Number of users in the control group"
        )
        control_conversions = st.number_input(
            "Conversions", 
            min_value=0, 
            max_value=control_visitors,
            value=500,
            step=10,
            key="conv_control_conversions",
            help="Number of conversions in the control group"
        )
    
    with col_treat:
        st.subheader("🟢 Treatment Group")
        
        treatment_visitors = st.number_input(
            "Visitors", 
            min_value=1, 
            value=10000,
            step=100,
            key="conv_treatment_visitors",
            help="Number of users in the treatment group"
        )
        treatment_conversions = st.number_input(
            "Conversions", 
            min_value=0, 
            max_value=treatment_visitors,
            value=550,
            step=10,
            key="conv_treatment_conversions",
            help="Number of conversions in the treatment group"
        )
    
    # Validate inputs
    valid_inputs = True
    if control_visitors < 100:
        st.error(f"❌ Control visitors ({control_visitors}) must be at least 100")
        valid_inputs = False
    if treatment_visitors < 100:
        st.error(f"❌ Treatment visitors ({treatment_visitors}) must be at least 100")
        valid_inputs = False
    if control_conversions > control_visitors:
        st.error(f"❌ Control conversions ({control_conversions}) cannot exceed visitors ({control_visitors})")
        control_conversions = control_visitors
        valid_inputs = False
    if treatment_conversions > treatment_visitors:
        st.error(f"❌ Treatment conversions ({treatment_conversions}) cannot exceed visitors ({treatment_visitors})")
        treatment_conversions = treatment_visitors
        valid_inputs = False
    
    # Advanced settings with session state
    if "conv_prior_alpha" not in st.session_state:
        st.session_state.conv_prior_alpha = 1.0
    if "conv_prior_beta" not in st.session_state:
        st.session_state.conv_prior_beta = 1.0
    if "conv_credible_pct" not in st.session_state:
        st.session_state.conv_credible_pct = 95
    
    with st.expander("🔧 Advanced Settings"):
        st.markdown("**Prior Parameters (Beta)**")
        st.caption("The Beta prior represents your belief about conversion rates *before* seeing data. α=1, β=1 is uniform.")
        
        col_prior1, col_prior2 = st.columns(2)
        with col_prior1:
            st.slider(
                "Prior α", 
                min_value=0.1, 
                max_value=10.0, 
                value=st.session_state.conv_prior_alpha,
                step=0.1,
                key="conv_prior_alpha",
                help="Beta prior alpha parameter"
            )
        with col_prior2:
            st.slider(
                "Prior β", 
                min_value=0.1, 
                max_value=10.0, 
                value=st.session_state.conv_prior_beta,
                step=0.1,
                key="conv_prior_beta",
                help="Beta prior beta parameter"
            )
        
        st.slider(
            "Credible Interval %",
            min_value=80,
            max_value=99,
            value=st.session_state.conv_credible_pct,
            step=1,
            format="%d%%",
            key="conv_credible_pct",
            help="Width of the credible interval"
        )
        
        # Show prior visualization
        st.markdown("**Your Prior Distribution:**")
        prior = stats.beta(st.session_state.conv_prior_alpha, st.session_state.conv_prior_beta)
        x_prior = np.linspace(0, 1, 100)
        y_prior = prior.pdf(x_prior)
        
        fig_prior = go.Figure()
        fig_prior.add_trace(go.Scatter(
            x=x_prior, y=y_prior,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#FFA15A')
        ))
        fig_prior.update_layout(
            height=150,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_tickformat='.0%',
            showlegend=False
        )
        st.plotly_chart(fig_prior, use_container_width=True)
    
    # Read from session state
    prior_alpha = st.session_state.conv_prior_alpha
    prior_beta = st.session_state.conv_prior_beta
    credible_level = st.session_state.conv_credible_pct / 100
    
    # Create experiment data object
    data = ExperimentData(
        control_visitors=control_visitors,
        control_conversions=control_conversions,
        treatment_visitors=treatment_visitors,
        treatment_conversions=treatment_conversions
    )
    
    # Run analysis
    results = compute_bayesian_analysis(
        data=data,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        credible_level=credible_level
    )
    
    # ==========================================================================
    # Results Display
    # ==========================================================================
    
    # Key Metrics Row
    st.header("📈 Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="P(Treatment > Control)",
            value=f"{results.prob_treatment_better:.1%}",
            help="Probability that the treatment conversion rate is higher than control"
        )
    
    with col2:
        st.metric(
            label="Expected Lift",
            value=f"{results.expected_lift:+.2%}",
            delta=f"Observed: {data.observed_lift:+.2%}",
            help="Expected relative improvement of treatment over control"
        )
    
    with col3:
        st.metric(
            label=f"{int(credible_level*100)}% Credible Interval",
            value=f"[{results.lift_ci_lower:+.1%}, {results.lift_ci_upper:+.1%}]",
            help="95% probability the true lift falls within this range"
        )
    
    with col4:
        # Recommendation based on results
        if results.prob_treatment_better > 0.95:
            recommendation = "✅ Ship It"
            rec_color = "green"
        elif results.prob_treatment_better < 0.05:
            recommendation = "❌ Don't Ship"
            rec_color = "red"
        else:
            recommendation = "⏳ Gather More Data"
            rec_color = "orange"
        
        st.metric(
            label="Recommendation",
            value=recommendation,
            help="Based on 95% confidence threshold"
        )
    
    st.divider()
    
    # Visualizations
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.plotly_chart(
            create_posterior_plot(results, data),
            use_container_width=True
        )
    
    with col_right:
        st.plotly_chart(
            create_lift_distribution_plot(results),
            use_container_width=True
        )
    
    # Risk Analysis
    st.subheader("⚖️ Decision Risk Analysis")
    
    col_risk1, col_risk2 = st.columns([1, 2])
    
    with col_risk1:
        st.plotly_chart(
            create_risk_plot(results),
            use_container_width=True
        )
    
    with col_risk2:
        st.markdown(f"""
        **Understanding Expected Loss:**
        
        Expected loss quantifies the "cost" of making a wrong decision:
        
        - **If you choose Treatment:** You risk losing **{results.risk_choosing_treatment * 100:.3f}** 
          percentage points on average if Control was actually better.
        
        - **If you choose Control:** You risk losing **{results.risk_choosing_control * 100:.3f}** 
          percentage points on average if Treatment was actually better.
        
        **Interpretation:** Choose the option with the *lower* expected loss. Currently, 
        {'**Treatment**' if results.risk_choosing_treatment < results.risk_choosing_control else '**Control**'} 
        has lower risk.
        """)
    
    # Detailed Statistics
    with st.expander("📋 Detailed Statistics"):
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            st.markdown("**Observed Data:**")
            st.markdown(f"""
            | Metric | Control | Treatment |
            |--------|---------|-----------|
            | Visitors | {data.control_visitors:,} | {data.treatment_visitors:,} |
            | Conversions | {data.control_conversions:,} | {data.treatment_conversions:,} |
            | Conversion Rate | {data.control_rate:.2%} | {data.treatment_rate:.2%} |
            """)
        
        with col_stats2:
            st.markdown("**Posterior Statistics:**")
            st.markdown(f"""
            | Metric | Control | Treatment |
            |--------|---------|-----------|
            | Posterior Mean | {results.control_posterior.mean():.4f} | {results.treatment_posterior.mean():.4f} |
            | Posterior Std | {results.control_posterior.std():.4f} | {results.treatment_posterior.std():.4f} |
            | 95% CI Lower | {results.control_posterior.ppf(0.025):.4f} | {results.treatment_posterior.ppf(0.025):.4f} |
            | 95% CI Upper | {results.control_posterior.ppf(0.975):.4f} | {results.treatment_posterior.ppf(0.975):.4f} |
            """)
        
        st.markdown(f"""
        **Lift Statistics:**
        - Absolute Difference: {results.absolute_diff_mean:.4f} ({results.absolute_diff_ci[0]:.4f}, {results.absolute_diff_ci[1]:.4f})
        - Relative Lift: {results.expected_lift:.2%} ({results.lift_ci_lower:.2%}, {results.lift_ci_upper:.2%})
        """)
    
    # Methodology explanation
    with st.expander("📚 Methodology"):
        st.markdown("""
        ### Beta-Binomial Bayesian Model
        
        This calculator uses a **Beta-Binomial conjugate model**, the gold standard for 
        analyzing conversion rate experiments:
        
        **The Model:**
        ```
        Prior:      p ~ Beta(α, β)
        Likelihood: conversions ~ Binomial(visitors, p)
        Posterior:  p | data ~ Beta(α + conversions, β + non-conversions)
        ```
        
        **Why Bayesian?**
        
        | Frequentist (p-values) | Bayesian |
        |------------------------|----------|
        | "Reject/fail to reject null" | "X% probability treatment is better" |
        | Binary decision | Full uncertainty quantification |
        | Can't say "95% sure treatment is better" | Directly answers business questions |
        | Sensitive to stopping rules | Valid at any sample size |
        
        **Key Metrics Explained:**
        
        - **P(Treatment > Control):** Direct probability that treatment has higher conversion rate
        - **Credible Interval:** 95% probability the true parameter falls in this range 
          (unlike frequentist CIs which have a more convoluted interpretation)
        - **Expected Loss:** Average "regret" if you make the wrong decision—useful for 
          risk-averse decision making
        
        **Prior Selection:**
        
        The default uniform prior (α=1, β=1) is uninformative—it assumes all conversion 
        rates from 0% to 100% are equally likely before seeing data. For most A/B tests 
        with reasonable sample sizes, the prior has minimal impact on results.
        
        ---
        *Built with Python, Streamlit, SciPy, and Plotly*
        """)


def sample_size_calculator_tab():
    """Sample size calculator tab for experiment planning."""
    
    st.markdown("""
    **Plan your experiment before you start.**
    
    Determine how many users you need to detect a meaningful effect with high confidence.
    """)
    
    st.divider()
    
    # Input columns
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.subheader("📊 Expected Metrics")
        
        baseline_rate_pct = st.number_input(
            "Baseline Conversion Rate (%)",
            min_value=0.1,
            max_value=99.0,
            value=5.0,
            step=0.5,
            help="Your current/expected conversion rate for the control group"
        )
        baseline_rate = baseline_rate_pct / 100
        
        mde_pct = st.number_input(
            "Minimum Detectable Effect (%)",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="Smallest relative lift you want to detect (e.g., 10% = detect if treatment is 10% better)"
        )
        mde_relative = mde_pct / 100
    
    with col_input2:
        st.subheader("🎯 Experiment Parameters")
        
        target_power_pct = st.slider(
            "Target Power (%)",
            min_value=50,
            max_value=99,
            value=80,
            step=5,
            format="%d%%",
            help="Probability of detecting the effect if it truly exists"
        )
        target_power = target_power_pct / 100
        
        daily_traffic = st.number_input(
            "Daily Traffic (total, optional)",
            min_value=0,
            value=0,
            step=100,
            help="Total daily visitors to estimate experiment runtime (leave 0 to skip)"
        )
        if daily_traffic == 0:
            daily_traffic = None
    
    # Calculate button
    if st.button("🔬 Calculate Sample Size", type="primary", use_container_width=True):
        with st.spinner("Running simulations..."):
            result = compute_sample_size(
                baseline_rate=baseline_rate,
                mde_relative=mde_relative,
                target_power=target_power,
                daily_traffic=daily_traffic
            )
        
        # Results
        st.divider()
        st.header("📋 Recommendations")
        
        # Key metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric(
                label="Sample Size (per variant)",
                value=f"{result.sample_size_per_variant:,}",
                help="Number of users needed in each group"
            )
        
        with col_m2:
            st.metric(
                label="Total Sample Size",
                value=f"{result.total_sample_size:,}",
                help="Total users across both variants"
            )
        
        with col_m3:
            if result.expected_runtime_days is not None:
                if result.expected_runtime_days < 1:
                    runtime_str = f"{result.expected_runtime_days * 24:.1f} hours"
                elif result.expected_runtime_days < 7:
                    runtime_str = f"{result.expected_runtime_days:.1f} days"
                else:
                    runtime_str = f"{result.expected_runtime_days / 7:.1f} weeks"
                st.metric(
                    label="Estimated Runtime",
                    value=runtime_str,
                    help="Based on your daily traffic"
                )
            else:
                st.metric(
                    label="Estimated Runtime",
                    value="—",
                    help="Enter daily traffic to estimate"
                )
        
        # Power curve plot
        st.plotly_chart(
            create_power_curve_plot(result, target_power),
            use_container_width=True
        )
        
        # Interpretation
        treatment_rate = baseline_rate * (1 + mde_relative)
        st.info(f"""
        **Interpretation:**
        
        With **{result.sample_size_per_variant:,} users per variant**, you have a **{result.power_at_size:.0%}** 
        probability of detecting a **{mde_pct:.0f}%** relative lift (from {baseline_rate_pct:.1f}% to {treatment_rate*100:.2f}% conversion rate) 
        with 95% confidence.
        
        This means if the treatment truly improves conversion by {mde_pct:.0f}% or more, 
        you'll correctly identify it as a winner {result.power_at_size:.0%} of the time.
        """)
        
        # Assumptions
        with st.expander("📚 Methodology & Assumptions"):
            st.markdown(f"""
            **How this works:**
            
            This calculator uses **Monte Carlo simulation** to determine sample size:
            
            1. For each candidate sample size, we simulate {1000:,} experiments
            2. Each simulation draws data assuming the treatment has the specified lift
            3. We compute the Bayesian posterior P(treatment > control)
            4. An experiment "detects" the effect if P(treatment > control) ≥ 95%
            5. Power = proportion of simulations that detect the effect
            
            **Key Assumptions:**
            - Equal sample sizes in control and treatment
            - 50/50 traffic split
            - Beta(1,1) uniform prior (can be adjusted)
            - Decision threshold: 95% posterior probability
            
            **Compared to Frequentist Power Analysis:**
            
            | Aspect | Frequentist | Bayesian (this tool) |
            |--------|-------------|---------------------|
            | Criterion | Reject H₀ at α=0.05 | P(B>A) ≥ 95% |
            | Interpretation | Type I/II error rates | Direct probability statements |
            | Stopping | Fixed sample size | Can peek anytime |
            """)


def revenue_analysis_tab():
    """Revenue analysis tab using Beta-Binomial hurdle + Gamma-Gamma model."""
    
    st.markdown("""
    **Analyze revenue experiments using the BTYD (Buy Till You Die) framework.**
    
    This uses a **hurdle model** with Fader/Hardie (2005) Gamma-Gamma specification:
    1. **Purchase probability** (Beta-Binomial) — who converts
    2. **Average order value** (Gamma-Gamma with heterogeneity) — spend per transaction
    
    Combined: **Revenue Per Visitor = P(purchase) × E[AOV | purchase]**
    """)
    
    st.divider()
    
    # Input section
    col_ctrl, col_treat = st.columns(2)
    
    with col_ctrl:
        st.subheader("🔵 Control Group")
        
        ctrl_visitors = st.number_input(
            "Visitors",
            min_value=1,
            value=10000,
            step=100,
            key="rev_ctrl_visitors",
            help="Total visitors in control"
        )
        
        ctrl_purchasers = st.number_input(
            "Unique Purchasers",
            min_value=0,
            max_value=ctrl_visitors,
            value=500,
            step=10,
            key="rev_ctrl_purchasers",
            help="Number of unique visitors who made at least one purchase"
        )
        
        ctrl_transactions = st.number_input(
            "Total Transactions",
            min_value=0,
            value=600,
            step=10,
            key="rev_ctrl_transactions",
            help="Total number of transactions (can be > purchasers if repeat purchases)"
        )
        
        ctrl_revenue = st.number_input(
            "Total Revenue ($)",
            min_value=0.0,
            value=30000.0,
            step=100.0,
            key="rev_ctrl_revenue",
            help="Total revenue from control group"
        )
    
    with col_treat:
        st.subheader("🟢 Treatment Group")
        
        treat_visitors = st.number_input(
            "Visitors",
            min_value=1,
            value=10000,
            step=100,
            key="rev_treat_visitors",
            help="Total visitors in treatment"
        )
        
        treat_purchasers = st.number_input(
            "Unique Purchasers",
            min_value=0,
            max_value=treat_visitors,
            value=550,
            step=10,
            key="rev_treat_purchasers",
            help="Number of unique visitors who made at least one purchase"
        )
        
        treat_transactions = st.number_input(
            "Total Transactions",
            min_value=0,
            value=700,
            step=10,
            key="rev_treat_transactions",
            help="Total number of transactions (can be > purchasers if repeat purchases)"
        )
        
        treat_revenue = st.number_input(
            "Total Revenue ($)",
            min_value=0.0,
            value=36400.0,
            step=100.0,
            key="rev_treat_revenue",
            help="Total revenue from treatment group"
        )
    
    # Validate inputs
    if ctrl_visitors < 100:
        st.error(f"❌ Control visitors ({ctrl_visitors}) must be at least 100")
    if treat_visitors < 100:
        st.error(f"❌ Treatment visitors ({treat_visitors}) must be at least 100")
    if ctrl_purchasers > ctrl_visitors:
        st.error(f"❌ Control purchasers ({ctrl_purchasers}) cannot exceed visitors ({ctrl_visitors})")
        ctrl_purchasers = ctrl_visitors
    if treat_purchasers > treat_visitors:
        st.error(f"❌ Treatment purchasers ({treat_purchasers}) cannot exceed visitors ({treat_visitors})")
        treat_purchasers = treat_visitors
    if ctrl_transactions < ctrl_purchasers:
        st.error(f"❌ Control transactions ({ctrl_transactions}) must be ≥ purchasers ({ctrl_purchasers})")
        ctrl_transactions = ctrl_purchasers
    if treat_transactions < treat_purchasers:
        st.error(f"❌ Treatment transactions ({treat_transactions}) must be ≥ purchasers ({treat_purchasers})")
        treat_transactions = treat_purchasers
    
    # Advanced settings - define sliders outside expander visually but track via session state
    # Initialize session state defaults
    if "rev_prior_purchase_alpha" not in st.session_state:
        st.session_state.rev_prior_purchase_alpha = 1.0
    if "rev_prior_purchase_beta" not in st.session_state:
        st.session_state.rev_prior_purchase_beta = 1.0
    if "rev_prior_aov_shape" not in st.session_state:
        st.session_state.rev_prior_aov_shape = 1.0
    if "rev_prior_aov_rate" not in st.session_state:
        st.session_state.rev_prior_aov_rate = 0.01
    
    with st.expander("🔧 Advanced Settings"):
        st.markdown("**Purchase Rate Prior (Beta)**")
        st.caption("Controls the prior belief about purchase probability. α=1, β=1 is uniform.")
        
        col_prior1, col_prior2 = st.columns(2)
        with col_prior1:
            st.slider(
                "Prior α (purchases)",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.rev_prior_purchase_alpha,
                step=0.1,
                key="rev_prior_purchase_alpha",
                help="Beta prior alpha for purchase rate"
            )
        with col_prior2:
            st.slider(
                "Prior β (purchases)",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.rev_prior_purchase_beta,
                step=0.1,
                key="rev_prior_purchase_beta",
                help="Beta prior beta for purchase rate"
            )
        
        st.markdown("**AOV Prior (Gamma)**")
        st.caption("Controls the prior belief about average order value. Higher shape = more concentrated.")
        
        col_prior3, col_prior4 = st.columns(2)
        with col_prior3:
            st.slider(
                "Prior Shape (AOV)",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.rev_prior_aov_shape,
                step=0.1,
                key="rev_prior_aov_shape",
                help="Gamma prior shape parameter for AOV"
            )
        with col_prior4:
            st.slider(
                "Prior Rate (AOV)",
                min_value=0.001,
                max_value=1.0,
                value=st.session_state.rev_prior_aov_rate,
                step=0.001,
                format="%.3f",
                key="rev_prior_aov_rate",
                help="Gamma prior rate parameter for AOV"
            )
        
        # Show prior visualizations
        st.markdown("**Your Prior Distributions:**")
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Purchase rate prior
            prior_purchase = stats.beta(
                st.session_state.rev_prior_purchase_alpha, 
                st.session_state.rev_prior_purchase_beta
            )
            x_purchase = np.linspace(0, 1, 100)
            y_purchase = prior_purchase.pdf(x_purchase)
            
            fig_purchase_prior = go.Figure()
            fig_purchase_prior.add_trace(go.Scatter(
                x=x_purchase, y=y_purchase,
                mode='lines', fill='tozeroy',
                line=dict(color='#636EFA')
            ))
            fig_purchase_prior.update_layout(
                title="Purchase Rate Prior",
                height=150,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_tickformat='.0%',
                showlegend=False
            )
            st.plotly_chart(fig_purchase_prior, use_container_width=True)
        
        with col_viz2:
            # AOV prior
            prior_aov = stats.gamma(
                a=st.session_state.rev_prior_aov_shape, 
                scale=1/st.session_state.rev_prior_aov_rate
            )
            x_aov = np.linspace(0, min(500, prior_aov.ppf(0.99)), 100)
            y_aov = prior_aov.pdf(x_aov)
            
            fig_aov_prior = go.Figure()
            fig_aov_prior.add_trace(go.Scatter(
                x=x_aov, y=y_aov,
                mode='lines', fill='tozeroy',
                line=dict(color='#00CC96')
            ))
            fig_aov_prior.update_layout(
                title="AOV Prior",
                height=150,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_tickprefix="$",
                showlegend=False
            )
            st.plotly_chart(fig_aov_prior, use_container_width=True)
    
    # Read prior values from session state (persists even when expander is closed)
    prior_purchase_alpha = st.session_state.rev_prior_purchase_alpha
    prior_purchase_beta = st.session_state.rev_prior_purchase_beta
    prior_aov_shape = st.session_state.rev_prior_aov_shape
    prior_aov_rate = st.session_state.rev_prior_aov_rate
    
    # Create data object
    data = RevenueData(
        control_visitors=ctrl_visitors,
        control_purchasers=ctrl_purchasers,
        control_transactions=ctrl_transactions,
        control_total_revenue=ctrl_revenue,
        treatment_visitors=treat_visitors,
        treatment_purchasers=treat_purchasers,
        treatment_transactions=treat_transactions,
        treatment_total_revenue=treat_revenue
    )
    
    # Validate data
    if ctrl_purchasers == 0 or treat_purchasers == 0:
        st.warning("⚠️ At least one group has zero purchasers. Results may be unreliable.")
    
    # Run analysis with configured priors
    results = compute_revenue_analysis(
        data=data,
        prior_purchase_alpha=prior_purchase_alpha,
        prior_purchase_beta=prior_purchase_beta,
        prior_aov_shape=prior_aov_shape,
        prior_aov_rate=prior_aov_rate
    )
    
    # ==========================================================================
    # Results Display
    # ==========================================================================
    
    st.header("📊 Results")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="P(Treatment RPV > Control)",
            value=f"{results.prob_treatment_rpv_better:.1%}",
            help="Probability that treatment generates more revenue per visitor"
        )
    
    with col2:
        st.metric(
            label="Expected RPV Lift",
            value=f"{results.expected_rpv_lift:+.2%}",
            delta=f"${results.expected_rpv_diff:+.4f}/visitor",
            help="Expected relative improvement in revenue per visitor"
        )
    
    with col3:
        st.metric(
            label="95% Credible Interval",
            value=f"[{results.rpv_lift_ci[0]:+.1%}, {results.rpv_lift_ci[1]:+.1%}]",
            help="95% probability the true RPV lift falls within this range"
        )
    
    with col4:
        if results.prob_treatment_rpv_better > 0.95:
            recommendation = "✅ Ship It"
        elif results.prob_treatment_rpv_better < 0.05:
            recommendation = "❌ Don't Ship"
        else:
            recommendation = "⏳ Gather More Data"
        
        st.metric(
            label="Recommendation",
            value=recommendation,
            help="Based on 95% confidence threshold"
        )
    
    st.divider()
    
    # Component breakdown
    st.subheader("📈 Revenue Decomposition")
    
    col_pr, col_aov = st.columns(2)
    
    with col_pr:
        st.markdown("**Purchase Rate**")
        st.markdown(f"""
        | Metric | Control | Treatment |
        |--------|---------|-----------|
        | Rate | {data.control_purchase_rate:.2%} | {data.treatment_purchase_rate:.2%} |
        | P(T > C) | {results.prob_treatment_purchase_rate_better:.1%} | |
        """)
    
    with col_aov:
        st.markdown("**Average Order Value**")
        st.markdown(f"""
        | Metric | Control | Treatment |
        |--------|---------|-----------|
        | AOV | ${data.control_avg_order_value:.2f} | ${data.treatment_avg_order_value:.2f} |
        | P(T > C) | {results.prob_treatment_aov_better:.1%} | |
        """)
    
    # Posterior distributions
    st.plotly_chart(
        create_revenue_decomposition_plot(results, data),
        use_container_width=True
    )
    
    # RPV Lift distribution
    col_lift, col_risk = st.columns([2, 1])
    
    with col_lift:
        st.plotly_chart(
            create_rpv_lift_plot(results),
            use_container_width=True
        )
    
    with col_risk:
        st.plotly_chart(
            create_revenue_risk_plot(results),
            use_container_width=True
        )
        
        st.markdown(f"""
        **Risk Analysis:**
        
        - **Choose Treatment:** Risk losing **${results.risk_choosing_treatment:.4f}**/visitor
        - **Choose Control:** Risk losing **${results.risk_choosing_control:.4f}**/visitor
        
        Lower risk: {'**Treatment**' if results.risk_choosing_treatment < results.risk_choosing_control else '**Control**'}
        """)
    
    # Detailed statistics
    with st.expander("📋 Detailed Statistics"):
        st.markdown(f"""
        **Observed Data:**
        
        | Metric | Control | Treatment | Lift |
        |--------|---------|-----------|------|
        | Visitors | {data.control_visitors:,} | {data.treatment_visitors:,} | — |
        | Unique Purchasers | {data.control_purchasers:,} | {data.treatment_purchasers:,} | {(data.treatment_purchasers/data.control_purchasers - 1)*100:+.1f}% |
        | Purchase Rate | {data.control_purchase_rate:.2%} | {data.treatment_purchase_rate:.2%} | {(data.treatment_purchase_rate/data.control_purchase_rate - 1)*100:+.1f}% |
        | Total Transactions | {data.control_transactions:,} | {data.treatment_transactions:,} | {(data.treatment_transactions/data.control_transactions - 1)*100:+.1f}% |
        | Txns/Buyer | {data.control_avg_transactions_per_buyer:.2f} | {data.treatment_avg_transactions_per_buyer:.2f} | {(data.treatment_avg_transactions_per_buyer/data.control_avg_transactions_per_buyer - 1)*100:+.1f}% |
        | Total Revenue | ${data.control_total_revenue:,.2f} | ${data.treatment_total_revenue:,.2f} | {(data.treatment_total_revenue/data.control_total_revenue - 1)*100:+.1f}% |
        | AOV (per txn) | ${data.control_avg_order_value:.2f} | ${data.treatment_avg_order_value:.2f} | {(data.treatment_avg_order_value/data.control_avg_order_value - 1)*100:+.1f}% |
        | CLV/Buyer | ${data.control_clv_per_buyer:.2f} | ${data.treatment_clv_per_buyer:.2f} | {(data.treatment_clv_per_buyer/data.control_clv_per_buyer - 1)*100:+.1f}% |
        | Revenue/Visitor | ${data.control_revenue_per_visitor:.4f} | ${data.treatment_revenue_per_visitor:.4f} | {(data.treatment_revenue_per_visitor/data.control_revenue_per_visitor - 1)*100:+.1f}% |
        """)
        
        st.markdown(f"""
        **Posterior Summary (RPV):**
        
        | Statistic | Control | Treatment |
        |-----------|---------|-----------|
        | Mean | ${np.mean(results.control_rpv_samples):.4f} | ${np.mean(results.treatment_rpv_samples):.4f} |
        | Std Dev | ${np.std(results.control_rpv_samples):.4f} | ${np.std(results.treatment_rpv_samples):.4f} |
        | 2.5% | ${np.percentile(results.control_rpv_samples, 2.5):.4f} | ${np.percentile(results.treatment_rpv_samples, 2.5):.4f} |
        | 97.5% | ${np.percentile(results.control_rpv_samples, 97.5):.4f} | ${np.percentile(results.treatment_rpv_samples, 97.5):.4f} |
        """)
    
    # Methodology
    with st.expander("📚 Methodology: BTYD Framework"):
        st.markdown("""
        ### Buy Till You Die (BTYD) Framework
        
        This implements a **simplified BTYD model** for A/B testing, following Fader & Hardie's work.
        
        Revenue per user is challenging to model because:
        - **Most users spend $0** (zero-inflated)
        - **Positive spends are right-skewed** (some users spend a lot)
        - **Customers are heterogeneous** (some consistently spend more)
        
        The **BTYD hurdle model** decomposes revenue:
        
        ```
        Revenue Per Visitor = P(purchase) × E[spend | purchase]
        ```
        
        **Component 1: Purchase Probability (Beta-Binomial)**
        ```
        Prior:      p ~ Beta(α, β)
        Likelihood: purchasers ~ Binomial(visitors, p)
        Posterior:  p | data ~ Beta(α + purchasers, β + non-purchasers)
        ```
        
        **Component 2: Average Order Value (Gamma-Gamma, Fader/Hardie 2005)**
        
        The Gamma-Gamma model captures **heterogeneity** in customer spend:
        
        ```
        Individual transaction:  z_ij ~ Gamma(p, ν_i)    [varies per transaction]
        Customer spend rate:     ν_i ~ Gamma(q, γ)       [varies per customer]
        ```
        
        This hierarchical structure means:
        - Some customers consistently spend more than others (heterogeneity)
        - Transaction amounts vary around each customer's personal mean
        - Posterior estimates shrink toward the population mean with limited data
        
        **Key Insight:** Using total *transactions* (not just purchasers) provides more 
        information about spending patterns when customers make repeat purchases.
        
        **Why This Approach?**
        
        | Alternative | Problem |
        |-------------|---------|
        | Simple t-test on revenue | Assumes normality; violated by zeros and skew |
        | Log-transform | Can't handle zeros; back-transform bias |
        | Mann-Whitney | No uncertainty quantification |
        | Simple Gamma | Ignores customer heterogeneity |
        | **BTYD Gamma-Gamma** | ✅ Handles zeros, skew, heterogeneity, full posteriors |
        
        **References:**
        - Fader, Hardie, Lee (2005) - "Counting Your Customers" (BG/NBD model)
        - Fader & Hardie (2005) - "The Gamma-Gamma Model of Monetary Value"
        - Fader & Hardie (2013) - "Probability Models for Customer-Base Analysis"
        """)


if __name__ == "__main__":
    main()
