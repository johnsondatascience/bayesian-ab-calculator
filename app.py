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
    np.random.seed(42)  # Reproducibility
    samples_control = control_posterior.rvs(n_samples)
    samples_treatment = treatment_posterior.rvs(n_samples)
    
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
    
    ---
    """)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("⚙️ Experiment Data")
        
        st.subheader("Control Group")
        control_visitors = st.number_input(
            "Visitors", 
            min_value=1, 
            value=10000,
            step=100,
            key="control_visitors",
            help="Number of users in the control group"
        )
        control_conversions = st.number_input(
            "Conversions", 
            min_value=0, 
            max_value=control_visitors,
            value=500,
            step=10,
            key="control_conversions",
            help="Number of conversions in the control group"
        )
        
        st.subheader("Treatment Group")
        treatment_visitors = st.number_input(
            "Visitors", 
            min_value=1, 
            value=10000,
            step=100,
            key="treatment_visitors",
            help="Number of users in the treatment group"
        )
        treatment_conversions = st.number_input(
            "Conversions", 
            min_value=0, 
            max_value=treatment_visitors,
            value=550,
            step=10,
            key="treatment_conversions",
            help="Number of conversions in the treatment group"
        )
        
        st.divider()
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            st.markdown("**Prior Parameters**")
            st.caption("""
            The Beta prior represents your belief about conversion rates *before* 
            seeing data. α=1, β=1 is a uniform (uninformative) prior.
            """)
            
            prior_alpha = st.slider(
                "Prior α", 
                min_value=0.1, 
                max_value=10.0, 
                value=1.0,
                step=0.1,
                help="Beta prior alpha parameter"
            )
            prior_beta = st.slider(
                "Prior β", 
                min_value=0.1, 
                max_value=10.0, 
                value=1.0,
                step=0.1,
                help="Beta prior beta parameter"
            )
            
            credible_level = st.slider(
                "Credible Interval",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                format="%.0f%%",
                help="Width of the credible interval"
            )
            
            # Show prior visualization
            st.markdown("**Your Prior Distribution:**")
            prior = stats.beta(prior_alpha, prior_beta)
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
        lift_delta = results.expected_lift - data.observed_lift if data.observed_lift != 0 else 0
        st.metric(
            label="Expected Lift",
            value=f"{results.expected_lift:+.2%}",
            delta=f"Observed: {data.observed_lift:+.2%}",
            help="Expected relative improvement of treatment over control"
        )
    
    with col3:
        st.metric(
            label="95% Credible Interval",
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
        
        - **If you choose Treatment:** You risk losing **{results.risk_choosing_treatment:.4f}** 
          percentage points on average if Control was actually better.
        
        - **If you choose Control:** You risk losing **{results.risk_choosing_control:.4f}** 
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


if __name__ == "__main__":
    main()
