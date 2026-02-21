## 1. Modeling Objective and Theoretical Background

This study models the evolution of market-implied volatility using the CBOE Volatility Index (VIX) as a proxy for aggregate market fear and uncertainty. The primary objective is to capture realistic volatility dynamics and their interaction with asset returns in a manner that supports portfolio optimization, dynamic programming (DP), and reinforcement learning (RL)–based decision-making frameworks.

The modeling approach is grounded in the Heston stochastic volatility framework, which establishes that asset returns and volatility innovations are contemporaneously correlated. Empirical evidence consistently documents a strong negative correlation between stock returns and changes in VIX, typically ranging from $-60\%$ to $-80\%$. This relationship implies that volatility reacts immediately to market information rather than with a lag.

In the Heston framework, asset prices and variance evolve according to the following stochastic differential equations:

$$
dS_t = r S_t \ dt + \sqrt{V_t}\ S_t \ dW_1(t)
$$

$$
dV_t = \kappa (\theta - V_t)\ dt + \eta \sqrt{V_t}\ dW_2(t)
$$

where the Brownian motions driving returns and volatility are correlated as:

$$
\mathbb{E}[dW_1 dW_2] = \rho \ dt, \quad \rho < 0
$$

This correlation structure motivates the central design choice of this work: VIX and asset returns are driven by shared contemporaneous shocks, ensuring that VIX acts as a leading indicator of market stress within the same decision period.

## 2. Mean-Reverting Jump-Diffusion (MRJD) VIX Model

### 2.1 Model Specification

The Mean-Reverting Jump-Diffusion (MRJD) model is designed to capture both the smooth evolution of implied volatility and the sudden spikes observed during periods of market stress. The dynamics of the VIX process are specified by the following stochastic differential equation:

$$
dV_t = \kappa (\theta - V_t)\ dt + \sigma_v V_t^{\beta}\ dW_t + J\ dN_t
$$

where $\kappa$ denotes the mean-reversion speed, $\theta$ represents the long-run average level of VIX, $\sigma_v$ controls the magnitude of continuous volatility fluctuations, and $\beta$ determines the level dependence of the diffusion term. The jump component is governed by a Poisson process $N_t$ with intensity $\lambda$, and $J$ denotes the jump magnitude.

---

### 2.2 Model Interpretation

The MRJD specification incorporates three essential features that characterize empirical VIX behavior.

**Mean Reversion.**  
The drift term $\kappa (\theta - V_t)$ ensures that VIX reverts toward its historical long-run mean $\theta$, which is calibrated to 20\%. The parameter $\kappa$ controls the speed at which volatility normalizes following periods of elevated market stress.

**Level-Dependent Diffusion.**  
The diffusion component $\sigma_v V_t^{\beta} dW_t$ introduces continuous stochastic variation in volatility. Setting $\beta = 0.5$ yields a square-root diffusion structure analogous to the Cox–Ingersoll–Ross (CIR) process, ensuring non-negativity of VIX while allowing volatility to increase with higher VIX levels.

**Jump Component.**  
The jump term $J\, dN_t$ captures abrupt increases in implied volatility associated with market dislocations and crisis events. Jump arrivals follow a Poisson process with baseline intensity $\lambda$, while jump magnitudes are calibrated to replicate empirically observed VIX surges. Jump probabilities are dynamically increased following large negative market shocks to reflect heightened systemic risk.

---

### 2.3 Numerical and Stability Considerations

To preserve numerical stability and empirical realism, simulated VIX levels are constrained to lie within the interval [9\%, 85\%], consistent with historical extremes observed in the data. Jump magnitudes are restricted to be non-negative, reflecting the asymmetric nature of volatility shocks, which predominantly manifest as upward movements in uncertainty.

## 3. Markov Regime-Switching (MRS) VIX Model

### 3.1 Model Motivation

While the Mean-Reverting Jump-Diffusion (MRJD) model captures continuous volatility fluctuations and abrupt crisis-driven spikes within a single stochastic process, empirical evidence indicates that volatility dynamics differ structurally across market environments. Periods of tranquility, elevated uncertainty, and systemic crisis exhibit distinct persistence, variance, and mean-reversion characteristics that cannot be fully represented by a single regime.

To address this limitation, we employ a Markov Regime-Switching (MRS) framework that allows the VIX process to evolve under multiple latent market regimes. This approach enables explicit modeling of persistent changes in volatility dynamics and provides a more realistic representation of market stress transitions.

---

### 3.2 Regime Structure

The MRS model assumes that VIX dynamics are governed by a discrete, unobserved regime variable $r_t \in \{1,2,3\}$, corresponding to three market states:

- **Tranquil regime:** low volatility and stable market conditions  
- **Turmoil regime:** elevated uncertainty and transitional stress  
- **Crisis regime:** extreme and persistent volatility  

Each regime $r$ is characterized by regime-specific parameters, including the long-run mean $\theta_r$, volatility $\sigma_r$, and mean-reversion speed $\kappa_r$.

Conditional on the current regime, VIX evolves according to the stochastic differential equation:

$$
dV_t = \kappa_r (\theta_r - V_t)\ dt + \sigma_r\ dW_t
$$

This formulation allows both the level and persistence of volatility to vary across market environments while preserving tractability.

---

### 3.3 Regime Transitions

Regime transitions are governed by a first-order Markov process with transition probabilities:

$$
\mathbb{P}(r_{t+1} = j \mid r_t = i) = p_{ij}
$$

where $P = [p_{ij}]$ denotes the regime transition matrix calibrated from historical VIX observations. Transition probabilities are allowed to depend on contemporaneous market conditions, with large negative market shocks increasing the likelihood of transitioning into higher-stress regimes.

The crisis regime is characterized by high self-transition probabilities, reflecting the empirically observed persistence of systemic stress periods. This structure enables the model to capture both rapid escalation into crisis states and slow recoveries back to tranquil conditions.

## 4. Correlation Structure and Shared Market Shocks

Empirical evidence consistently documents a strong negative contemporaneous relationship between equity market returns and changes in implied volatility. To preserve this fundamental stylized fact, both VIX models incorporate a shared shock structure that explicitly links market returns and volatility innovations.

Let $z_t$ denote the standardized shock driving asset returns. The Brownian motion governing VIX innovations is constructed as:

$$
dW_t = -\rho \ z_t \sqrt{dt} + \sqrt{1 - \rho^2}\ dW_t^{\perp}
$$

where $dW_t^{\perp}$ is an independent Brownian motion and $\rho < 0$ represents the correlation between returns and volatility changes.

This formulation implies the instantaneous correlation condition:

$$
\mathbb{E}[dW_1\, dW_2] = \rho\, dt, \quad \rho < 0
$$

ensuring that negative return shocks generate positive innovations in VIX.

This shared-shock construction enforces three key properties:

- **Simultaneous response:** asset returns and VIX react contemporaneously to market information  
- **Asymmetric reaction:** negative market shocks lead to volatility spikes  
- **Consistency across algorithms:** identical shock realizations are used across Dynamic Programming (DP), pure Reinforcement Learning (RL), and Sentiment-Aware RL simulations  

By aligning the stochastic drivers across asset returns and volatility, the framework ensures theoretical consistency and enables fair comparison of algorithmic decision-making approaches under identical market conditions.

## 5. VIX-Based Return Adjustments

To translate information from implied volatility into portfolio dynamics, VIX is incorporated directly into the expected return and risk structure of asset returns. This allows market sentiment, as reflected by volatility levels, to influence portfolio decisions in a systematic and interpretable manner.

Expected returns and return volatility are adjusted using linear sensitivity functions:

$$
\mu_{\text{adj}} = \mu_{\text{base}} + \beta(VIX), \qquad
\sigma_{\text{adj}} = \sigma_{\text{base}} - \delta(VIX)
$$

where $\mu_{\text{base}}$ and $\sigma_{\text{base}}$ denote baseline estimates obtained from historical data.

The adjustment terms are defined as:

$$
\beta(VIX) = \beta_{\text{sensitivity}} \frac{\theta - VIX_{\text{avg}}}{\theta}, \qquad
\delta(VIX) = \delta_{\text{sensitivity}} \frac{\theta - VIX_{\text{avg}}}{\theta}
$$

where $VIX_{\text{avg}}$ represents a short-term moving average of VIX used to smooth transitory noise while preserving sentiment information, and $\theta$ denotes the long-run mean level of implied volatility.

This specification implies that elevated VIX levels reduce expected returns and increase return volatility, reflecting heightened risk aversion and market uncertainty. Conversely, periods of low implied volatility correspond to favorable risk–return conditions.

The linear structure preserves interpretability, maintains numerical stability, and enables seamless integration into Dynamic Programming (DP) and Reinforcement Learning (RL)–based portfolio optimization frameworks.

## 6. Sentiment Feature Representation for Reinforcement Learning

For reinforcement learning (RL) applications, the continuous-time dynamics of VIX are mapped into a compact and stationary sentiment state that can be efficiently consumed by learning algorithms. This representation balances informational richness with numerical stability and sample efficiency.

The sentiment state at time $t$ consists of the following features:

- **Normalized current VIX level**
- **Normalized short-term VIX average**
- **VIX momentum**, defined as the rate of change in implied volatility

Each feature is normalized relative to the long-run mean level $\theta$:

$$
\tilde{V}_t = \frac{V_t - \theta}{\theta}, \qquad
\tilde{V}_{t}^{\text{avg}} = \frac{VIX_{\text{avg},t} - \theta}{\theta}
$$

VIX momentum is computed as:

$$
M_t = \frac{V_t - V_{t-1}}{\theta}
$$

All features are clipped to predefined bounds to prevent extreme values from destabilizing learning and to ensure numerical robustness across market regimes.

This normalized sentiment representation ensures stationarity, facilitates generalization across tranquil and crisis environments, and enables consistent comparison between pure RL and sentiment-aware RL agents within the same market simulation framework.

## 7. Summary

This methodology integrates continuous-time stochastic volatility modeling, crisis dynamics, and regime-dependent behavior into a unified framework for volatility-aware portfolio optimization and learning-based decision making. By explicitly modeling the evolution of market-implied volatility through VIX, the framework preserves the empirically observed contemporaneous relationship between asset returns and volatility innovations.

The Mean-Reverting Jump-Diffusion (MRJD) model provides a structurally grounded representation of volatility dynamics, capturing mean reversion, level-dependent uncertainty, and abrupt volatility spikes associated with systemic events. Complementarily, the Markov Regime-Switching (MRS) model captures persistent shifts across tranquil, turmoil, and crisis market environments, allowing volatility behavior to differ structurally across regimes.

A shared-shock correlation design ensures that asset returns and VIX respond simultaneously to common market innovations, maintaining theoretical consistency and enabling fair comparison across dynamic programming, pure reinforcement learning, and sentiment-aware reinforcement learning approaches. The proposed VIX-based return adjustments and sentiment feature construction translate volatility information into actionable signals while preserving stationarity and numerical stability.

Together, these components form a coherent and extensible foundation for goals-based wealth management under uncertainty, enabling robust evaluation of volatility-sensitive strategies across both traditional optimization and modern learning-based frameworks.