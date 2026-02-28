# Monthly VIX Baseline Mode: Code Execution Flow

This document traces the **theoretical steps** that occur when executing the `monthly_vix` baseline mode, following the actual code execution sequence from start to finish.

## Command Execution
```bash
python experiments/evaluate_sentiment_rl.py --baseline_mode monthly_vix --vix_model_type mrjd --num_simulations 100000 --num_iterations 20 --seed 42 --goal_counts 1 2 4 8 16 --policy_type hierarchical --value_type dual_head --encoder_type attention --use_real_ef --force_recompute --use_delta_adjustment --volatility_method return_squared --output_dir "data/results/monthly_vix_eval_100000"
```

## What Happens: Five-Method Comparison
The system trains and evaluates **5 distinct algorithms** using identical market conditions (stable market, no VIX adjustments):

1. **DP (Annual)** → 16 annual time steps, optimal policy
2. **Pure RL (Annual)** → 16 annual time steps, learned policy  
3. **DP (Monthly)** → 192 monthly time steps, optimal policy
4. **Pure RL (Monthly)** → 192 monthly time steps, learned policy (2D state)
5. **Sentiment RL (Monthly)** → 192 monthly time steps, learned policy (5D state with VIX)

## Code Execution Flow

### Step 1: Initialize System (Logs: "SentimentRLEvaluator initialized")
- **Time Horizon**: 192 monthly steps (16 years × 12 months)
- **Goal Counts**: [1, 2, 4, 8, 16] 
- **Monte Carlo**: 100,000 simulations
- **Seed**: 42 (reproducibility)
- **Market Type**: Stable (no VIX adjustments to returns)

### Step 2: Load Pre-Trained Parameters (Logs: "Loading β/δ parameters")
**Efficient Frontier**: 15 portfolios from real data (AGG, SPY, EFA)
$$\min_{\mathbf{w}} \frac{1}{2} \mathbf{w}^T \Sigma \mathbf{w} \text{ s.t. } \mathbf{w}^T \mathbf{1} = 1, \mathbf{w}^T \boldsymbol{\mu} = \mu_{\text{target}}$$

**Results**: Returns 5.26%-8.86%, Volatilities 3.71%-19.56%

**Beta-Delta Parameters** (loaded but unused in stable market):
- $\beta \in [-0.0577, -0.0517]$, $\delta \in [0.0163, 0.0901]$

### Step 3: Generate Shared Random Shocks (Logs: "Generating shared random state")
$$Z_{s,t} \sim \mathcal{N}(0,1) \text{ for } s = 1,\ldots,100000 \text{ and } t = 1,\ldots,192$$
- **Fair Comparison**: All 5 methods use identical market paths
- **VIX Correlation**: $\varepsilon_t = -0.7 Z_t + \sqrt{0.51} \xi_t$ (VIX shock correlated with market)

### Step 4: Solve Multi-Goal Dynamic Programming (Logs: "Solving DP for X goals")

**For each goal count** (1, 2, 4, 8, 16), solve optimal policy via backward induction:

**State**: $(w, t, \mathbf{g})$ where $w$ = wealth, $t$ = time, $\mathbf{g}$ = achieved goals

**Bellman Equation**:
$$V(w, t, \mathbf{g}) = \max_{(\text{goal}, \text{portfolio})} \mathbb{E}[U(\text{goal}) + V(w', t+1, \mathbf{g}')]$$

**Wealth Evolution** (Geometric Brownian Motion):
$$W_{t+1} = W_t \exp\left( \left(\mu_p - \frac{1}{2}\sigma_p^2\right)\Delta t + \sigma_p \sqrt{\Delta t} Z_{t+1} \right)$$

**Key Components**:
- $\mu_p \Delta t$: Expected return drift
- $-\frac{1}{2}\sigma_p^2 \Delta t$: Itô volatility correction  
- $\sigma_p \sqrt{\Delta t} Z_{t+1}$: **Brownian motion** random component

**Goal Utilities**: Years 4,8,12,16 → Utilities 14,18,22,26

### Step 5: Train Pure RL Monthly (Logs: "Training Pure RL for X goals")

**State Space**: 2D only (no VIX features)
$$\mathbf{s}_t = \begin{bmatrix} \frac{t}{192} \\ \min\left(\frac{W_t}{10W_0}, 1\right) \end{bmatrix}$$

**PPO Training**: 20 iterations, 4800 batch size, 0.01 learning rate

**Policy Architecture**: Hierarchical (separate goal/portfolio heads)
$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

**Market**: Same GBM wealth evolution, 192 monthly steps

### Step 6: Train Pure RL Annual (Logs: "Training Pure RL for X goals")

**Time Horizon**: 16 annual steps (faster training)
**Shock Aggregation**: $Z_{\text{annual},t} = \frac{1}{\sqrt{12}} \sum_{m=1}^{12} Z_{\text{monthly},12(t-1)+m}$
**Annual GBM**: $W_{t+1} = W_t \exp\left( (\mu_p - \frac{1}{2}\sigma_p^2) + \sigma_p Z_{\text{annual},t} \right)$

### Step 7: Train Sentiment RL with VIX Features (Logs: "Training Sentiment RL for X goals")

**Enhanced State Space**: 5D with VIX sentiment features
$$\mathbf{s}_t = \begin{bmatrix}
\frac{t}{192} \\
\min\left(\frac{W_t}{10W_0}, 1\right) \\
\frac{VIX_t - 20}{20} \\
\frac{\overline{VIX}_t - 20}{20} \\
\frac{VIX_t - VIX_{t-1}}{20}
\end{bmatrix}$$

All VIX features clipped to $[-2, 2]$ for stability.

**VIX Model**: Mean-Reverting Jump-Diffusion (MRJD) provides sentiment information
$$dVIX_t = \kappa(\theta - VIX_t)dt + \sigma_v VIX_t^{0.5} dW_t + J dN_t$$

**Parameters**: $\kappa=3.0$, $\theta=20.0$, $\sigma_v=0.8$, $\lambda=1.5$, $\mu_{\text{jump}}=20.0$, $\sigma_{\text{jump}}=15.0$

**Monthly VIX Evolution**:
1. **Mean Reversion**: $0.25 \times (20.0 - VIX_t)$ (pulls toward long-term mean)
2. **Diffusion**: $0.231 \times \sqrt{VIX_t} \times \varepsilon_t$ (continuous randomness)
3. **Jumps**: 12.5% monthly probability, $\mathcal{N}(20, 15^2)$ size (crisis spikes)
4. **Correlation**: $\varepsilon_t = -0.7 Z_t + \sqrt{0.51} \xi_t$ (negative correlation with market)

**Complete Update**:
$$VIX_{t+1} = \max(9, \min(85, VIX_t + 0.25(20-VIX_t) + 0.231\sqrt{VIX_t}\varepsilon_t + J_t))$$

**Key Insight**: VIX **information only** - does not affect returns in stable market mode

**Attention Architecture**: Multi-head attention processes VIX features
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Dual-Head Value**: $V(\mathbf{s}) = V_{\text{goal}}(\mathbf{s}) + V_{\text{portfolio}}(\mathbf{s})$

### Step 8: Monte Carlo Simulation (Logs: "100,000 simulations")

**All 5 Methods** execute on **identical** market paths using shared random seeds:

**Core Wealth Evolution** (same for all methods):
$$W_{t+1} = W_t \exp\left( \left(\mu_p - \frac{1}{2}\sigma_p^2\right)\Delta t + \sigma_p\sqrt{\Delta t} Z_{t+1} \right)$$

**Key Points**:
- **Pure GBM**: No VIX adjustments ($\mu_{\text{adj}} = \mu_p$, $\sigma_{\text{adj}} = \sigma_p$)
- **Shared Shocks**: Same $Z_t$ sequence for all methods
- **VIX Role**: Information only (for Sentiment RL state features)
- **Goal Rewards**: Utilities 14, 18, 22, 26 for years 4, 8, 12, 16

### Step 9: Performance Evaluation

**Metrics Computed**:
- Mean reward: $\bar{U} = \frac{1}{100000} \sum_{s=1}^{100000} U_s$
- Efficiency: $\frac{\bar{U}_{\text{method}}}{\bar{U}_{\text{DP}}}$
- Goal success rate, final wealth statistics

**Expected Hierarchy**:
$$\text{DP (Monthly)} \geq \text{DP (Annual)} \geq \text{Sentiment RL} \geq \text{Pure RL (Monthly)} \geq \text{Pure RL (Annual)}$$

**Analysis Dimensions**:
1. **Time Granularity**: Monthly vs Annual decisions
2. **Learning vs Optimization**: DP optimal vs RL approximation
3. **Information Value**: VIX features vs pure wealth/time

## Summary: What the Code Does

The `monthly_vix` baseline mode trains and evaluates **5 different goal-based wealth management algorithms**:

1. **DP (Annual)** - Backward induction optimal policy, 16 annual steps
2. **Pure RL (Annual)** - PPO-learned policy, 16 annual steps, 2D state
3. **DP (Monthly)** - Backward induction optimal policy, 192 monthly steps  
4. **Pure RL (Monthly)** - PPO-learned policy, 192 monthly steps, 2D state
5. **Sentiment RL (Monthly)** - PPO-learned policy, 192 monthly steps, 5D state with VIX

**Key Theoretical Elements**:
- **Wealth Evolution**: Pure Geometric Brownian Motion (no VIX adjustments)
- **VIX Model**: MRJD evolution provides sentiment features (information only)
- **Shared Randomness**: All methods use identical market shock sequences  
- **Goal Structure**: Multi-period utilities (14,18,22,26) at years 4,8,12,16

**Final Output**: Performance comparison isolating the value of time granularity, learning vs optimization, and sentiment information in goal-based investment timing decisions under stable market conditions.