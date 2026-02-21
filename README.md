## 1. Modeling Objective and Theoretical Background

This study models the evolution of market-implied volatility using the CBOE Volatility Index (VIX) as a proxy for aggregate market fear and uncertainty. The primary objective is to capture realistic volatility dynamics and their interaction with asset returns in a manner that supports portfolio optimization, dynamic programming (DP), and reinforcement learning (RL)–based decision-making frameworks.

The modeling approach is grounded in the Heston stochastic volatility framework, which establishes that asset returns and volatility innovations are contemporaneously correlated. Empirical evidence consistently documents a strong negative correlation between stock returns and changes in VIX, typically ranging from $-60\%$ to $-80\%$. This relationship implies that volatility reacts immediately to market information rather than with a lag.

In the Heston framework, asset prices and variance evolve according to the following stochastic differential equations:

$$
dS_t = r S_t \, dt + \sqrt{V_t}\, S_t \, dW_1(t)
$$

$$
dV_t = \kappa (\theta - V_t)\, dt + \eta \sqrt{V_t}\, dW_2(t)
$$

where the Brownian motions driving returns and volatility are correlated as:

$$
\mathbb{E}[dW_1 dW_2] = \rho \, dt, \quad \rho < 0
$$

This correlation structure motivates the central design choice of this work: VIX and asset returns are driven by shared contemporaneous shocks, ensuring that VIX acts as a leading indicator of market stress within the same decision period.