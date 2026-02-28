# Goals-Based Wealth Management with Sentiment-Aware Reinforcement Learning

A comprehensive framework integrating market sentiment modeling, dynamic programming, and reinforcement learning for multi-goal investment optimization under uncertainty.

## Table of Contents

1. [Overview and Objectives](#1-overview-and-objectives)
2. [Theoretical Framework](#2-theoretical-framework)
3. [VIX Modeling and Market Dynamics](#3-vix-modeling-and-market-dynamics)
4. [Sentiment-Aware Reinforcement Learning](#4-sentiment-aware-reinforcement-learning)
5. [Implementation Architecture](#5-implementation-architecture)
6. [Experimental Framework](#6-experimental-framework)
7. [Usage Guide](#7-usage-guide)
8. [Reproducibility and Configuration](#8-reproducibility-and-configuration)

## 1. Overview and Objectives

### 1.1 Core Problem

This system addresses the challenge of optimal investment decisions for multiple financial goals across time horizons, incorporating market sentiment through volatility indicators. The framework compares three algorithmic approaches:

- **Dynamic Programming (DP)**: Optimal baseline using backward induction
- **Pure Reinforcement Learning**: Learned policies with basic state features
- **Sentiment-Aware RL**: Enhanced RL incorporating VIX-based market sentiment

### 1.2 Key Innovation

The primary innovation is the integration of market sentiment through VIX (CBOE Volatility Index) features in a reinforcement learning framework, enabling agents to make investment decisions that account for market fear and uncertainty dynamics.

### 1.3 System Architecture

```
Market Data → Efficient Frontier → β/δ Learning → VIX Modeling → RL Training → Monte Carlo Evaluation
     ↓              ↓                 ↓             ↓             ↓              ↓
Real Returns   15 Portfolios    Sensitivity     MRJD/MRS      5D State      Performance
 (AGG/SPY/     (3.7%-19.6%     Parameters      Evolution     Features       Comparison
  EFA)          volatility)     (β,δ)          (Jumps)       (VIX Info)     (Efficiency)
```

## 2. Theoretical Framework

### 2.1 Wealth Evolution Models

#### Standard Heston Framework (Theoretical)
The mathematical foundation assumes stochastic volatility:

$$dS_t = r S_t dt + \sqrt{V_t} S_t dW_1(t)$$

$$dV_t = \kappa (\theta - V_t) dt + \eta \sqrt{V_t} dW_2(t)$$

where $\mathbb{E}[dW_1 dW_2] = \rho dt$ with $\rho < 0$.

#### Actual Implementation: Geometric Brownian Motion
In practice, the system uses **fixed volatility** from efficient frontier portfolios:

$$W_{t+1} = W_t \exp\left(\left(\mu_p - \frac{1}{2}\sigma_p^2\right)\Delta t + \sigma_p\sqrt{\Delta t} z_{t+1}\right)$$

**Key Design Choice**: VIX provides **information only** without directly modifying return dynamics, enabling clean evaluation of sentiment signals.

### 2.2 Goal-Based Utility Structure

#### Multi-Goal Framework
Goals are distributed across time horizons with increasing utility:

- **Goal Years**: 4, 8, 12, 16
- **Goal Utilities**: 14, 18, 22, 26
- **Goal Costs**: $C(t) = 100,000 × 1.08^t$

#### Total Utility
$$U_{\text{total}} = \sum_{i \in \text{achieved goals}} U(g_i) + \alpha × W_{\text{final}}$$

### 2.3 Shared Shock Structure

Critical for fair comparison across methods:

$$z_t \sim \mathcal{N}(0,1) \text{ (drives wealth evolution)}$$
$$\varepsilon_t = -\rho z_t \sqrt{dt} + \sqrt{1-\rho^2} dW_t^{\perp} \text{ (drives VIX)}$$

This ensures contemporaneous negative correlation between returns and VIX changes ($\rho \approx -0.7$).

## 3. VIX Modeling and Market Dynamics

### 3.1 Mean-Reverting Jump-Diffusion (MRJD) Model

#### Mathematical Specification
$$dV_t = \kappa (\theta - V_t) dt + \sigma_v V_t^{\beta} dW_t + J dN_t$$

#### Parameter Configuration
- **$\kappa = 3.0$**: Mean reversion speed (aggressive reversion)
- **$\theta = 20.0$**: Long-run VIX mean (historical average)
- **$\sigma_v = 0.8$**: Volatility of volatility
- **$\beta = 0.5$**: Square-root diffusion (Heston-type)
- **$\lambda = 1.5$**: Jump intensity (annual rate)
- **Jump size**: $\mathcal{N}(20, 15^2)$

#### Monthly Evolution Process
1. **Mean Reversion**: $0.25 × (20.0 - VIX_t)$
2. **Diffusion**: $0.231 × \sqrt{VIX_t} × \varepsilon_t$  
3. **Jumps**: 12.5% monthly probability
4. **Bounds**: $VIX_t \in [9, 85]$

### 3.2 Alternative: Markov Regime-Switching (MRS) Model

#### Three-Regime Structure
- **Tranquil**: Low volatility, stable conditions
- **Turmoil**: Elevated uncertainty, transitional stress  
- **Crisis**: Extreme, persistent volatility

$$dV_t = \kappa_r (\theta_r - V_t) dt + \sigma_r dW_t$$

#### Regime Transitions
$$\mathbb{P}(r_{t+1} = j | r_t = i) = p_{ij}$$

Crisis regimes feature high self-transition probabilities, modeling persistent stress periods.

### 3.3 VIX-Return Correlation Implementation

The correlation structure preserves empirical stylized facts while maintaining model tractability:

- **Negative Correlation**: Market crashes coincide with VIX spikes
- **Contemporaneous**: No lag between return and volatility shocks
- **Consistent**: Identical shock sequences across all algorithms

## 4. Sentiment-Aware Reinforcement Learning

### 4.1 Enhanced State Space

#### 5-Dimensional State Vector
$$\mathbf{s}_t = \begin{bmatrix}
\frac{t}{T} \\
\min\left(\frac{W_t}{10W_0}, 1\right) \\
\frac{VIX_t - \theta}{\theta} \\
\frac{\overline{VIX}_t - \theta}{\theta} \\
\frac{VIX_t - VIX_{t-1}}{20}
\end{bmatrix}$$

#### VIX Feature Engineering

**VIX Moving Average** (2-month window):
```python
def get_vix_average(lookback=2):
    return np.mean(vix_history[-lookback:]) if len(vix_history) >= lookback else 20.0
```

**VIX Momentum** (percentage change):
```python
vix_momentum = (VIX_t - VIX_{t-1}) / VIX_{t-1} if VIX_{t-1} > 0 else 0.0
```

**Feature Normalization**:
- VIX features normalized around $\theta = 20.0$
- Clipped bounds: VIX level/avg $\in [-0.5, 3.0]$, momentum $\in [-1.0, 1.0]$

### 4.2 Multi-Head Attention Architecture

#### Attention Mechanism
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### Implementation Details
- **Number of heads**: 4 parallel attention mechanisms
- **Head dimension**: $d_k = 32$ per head  
- **Input embedding**: 5D → 128D
- **Residual connections** and **layer normalization**

#### Implicit Head Specialization
- **Head 1**: Time-wealth correlation patterns
- **Head 2**: VIX regime detection (low/medium/high)
- **Head 3**: VIX momentum and trend analysis  
- **Head 4**: Cross-feature interactions

### 4.3 Hierarchical Policy Structure

#### Two-Level Decision Decomposition
```
φ_attention(s_t) → [Goal Head, Portfolio Head] → Joint Action
```

#### Goal Selection Head
$$\pi_{\text{goal}}(s_t) = \text{softmax}(W_{\text{goal}} \cdot \phi_{\text{attention}}(s_t) + b_{\text{goal}})$$

- **Architecture**: 128 → 64 → num_available_goals
- **Constraints**: Only unachieved, time-feasible goals
- **Features**: Time urgency, wealth adequacy, market conditions

#### Portfolio Selection Head  
$$\pi_{\text{portfolio}}(s_t, \text{goal}) = \text{softmax}(W_{\text{portfolio}} \cdot [\phi_{\text{attention}}(s_t), e_{\text{goal}}] + b_{\text{portfolio}})$$

- **Architecture**: 144 → 96 → 64 → 15
- **Goal embedding**: 16D learned representation
- **Output**: Distribution over efficient frontier portfolios

### 4.4 Dual-Head Value Function

#### Value Decomposition
$$V(s_t) = V_{\text{goal}}(s_t) + V_{\text{portfolio}}(s_t)$$

#### Separate Learning Dynamics
- **Goal value learning rate**: 0.005 (strategic decisions)
- **Portfolio value learning rate**: 0.01 (tactical decisions)  
- **Regularization**: L2 penalty $\lambda = 10^{-4}$

### 4.5 PPO Training Configuration

#### Clipped Objective
$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t)\right]$$

#### Hyperparameters
- **Learning rate**: 0.01 with Adam optimizer
- **Batch size**: 4,800 trajectories per iteration
- **Training iterations**: 20 per goal configuration
- **Clipping parameter**: $\varepsilon = 0.2$
- **GAE lambda**: 0.95
- **Entropy coefficient**: 0.01

## 5. Implementation Architecture

### 5.1 Efficient Frontier Construction

#### Portfolio Optimization
$$\min_{\mathbf{w}} \frac{1}{2} \mathbf{w}^T \Sigma \mathbf{w}$$
subject to:
- $\mathbf{w}^T \mathbf{1} = 1$ (weights sum to 1)
- $\mathbf{w}^T \boldsymbol{\mu} = \mu_{\text{target}}$ (target return)
- $w_i \geq 0$ (no short selling)

#### Asset Universe
- **AGG**: Bond ETF (aggregate bonds)
- **SPY**: US equity ETF (S&P 500)
- **EFA**: International equity ETF (EAFE)
- **Data period**: 2003-present (limited by AGG)

#### Results
- **15 portfolios** spanning risk-return spectrum
- **Returns**: 5.26% to 8.86% annually
- **Volatilities**: 3.71% to 19.56% annually

### 5.2 Beta-Delta Parameter Learning

#### VIX Sensitivity Regression

**Return sensitivity (β)**:
$$R_t = \alpha + \beta × \frac{\theta - VIX_{t-1}}{\theta} + \varepsilon_t$$

**Volatility sensitivity (δ)**:
$$\sigma_{\text{forward},t} = \gamma + \delta × \frac{\theta - VIX_{t-1}}{\theta} + \eta_t$$

#### Empirical Results
- **β range**: [-0.0577, -0.0517] (negative indicates mean reversion)
- **δ range**: [0.0163, 0.0901] (positive indicates volatility amplification)

#### Adjustment Formulas
$$\mu_{\text{adj}} = \mu_{\text{base}} + \beta × \frac{\theta - VIX_{\text{avg}}}{\theta}$$
$$\sigma_{\text{adj}} = \sigma_{\text{base}} - \delta × \frac{\theta - VIX_{\text{avg}}}{\theta}$$

### 5.3 Multi-Goal Dynamic Programming

#### State Space
$(w, t, \mathbf{g})$ where:
- $w$: Current wealth level
- $t$: Time step 
- $\mathbf{g}$: Binary vector of achieved goals

#### Bellman Equation
$$V(w, t, \mathbf{g}) = \max_{(\text{goal}, \text{portfolio})} \mathbb{E}[U(\text{goal}) + V(w', t+1, \mathbf{g}')]$$

#### Implementation Details
- **Wealth discretization**: 200 grid points
- **Time horizon**: 16 years (annual) or 192 months
- **Goal states**: $2^{\text{num_goals}}$ combinations
- **Backward induction**: Bellman iteration from terminal time

## 6. Experimental Framework

### 6.1 Baseline Mode Comparison

#### Five Algorithm Evaluation

1. **DP (Annual)**: Optimal policy, 16 annual time steps
2. **Pure RL (Annual)**: Learned policy, 16 annual time steps, 2D state
3. **DP (Monthly)**: Optimal policy, 192 monthly time steps
4. **Pure RL (Monthly)**: Learned policy, 192 monthly time steps, 2D state  
5. **Sentiment RL (Monthly)**: Learned policy, 192 monthly time steps, 5D state

#### Key Experimental Controls
- **Identical market paths**: Shared random shock sequences
- **Fixed efficient frontier**: Same portfolio universe
- **Goal structure**: Consistent utilities and timing
- **Stable market**: Pure GBM without VIX adjustments (information mode)

### 6.2 Monte Carlo Simulation

#### Simulation Scale
- **Default**: 100,000 independent trajectories
- **Shared seeding**: Identical market conditions across methods
- **Time horizon**: 16 years (192 monthly decisions)

#### Performance Metrics
- **Mean utility**: $\bar{U} = \frac{1}{N} \sum_{i=1}^N U_i$
- **Efficiency ratio**: $\frac{\bar{U}_{\text{method}}}{\bar{U}_{\text{DP}}}$
- **Goal achievement rates**: Success probability by goal type
- **Wealth statistics**: Mean, median, percentiles

### 6.3 Expected Performance Hierarchy
$$\text{DP (Monthly)} \geq \text{DP (Annual)} \geq \text{Sentiment RL} \geq \text{Pure RL (Monthly)} \geq \text{Pure RL (Annual)}$$

#### Analysis Dimensions
1. **Time granularity**: Monthly vs annual decision frequency
2. **Optimization method**: DP optimal vs RL learned approximation  
3. **Information content**: VIX sentiment features vs basic state

## 7. Usage Guide

### 7.1 Environment Setup

#### Conda Environment
```bash
conda create -n gbwm_rl python=3.10
conda activate gbwm_rl
pip install -r requirements.txt
```

#### Required Dependencies
All dependencies specified in `requirements.txt` include:
- **Deep Learning**: torch>=2.0.0, stable-baselines3>=2.0.0
- **Scientific Computing**: numpy>=1.21.0, scipy>=1.9.0, pandas>=1.5.0
- **Visualization**: matplotlib>=3.6.0, seaborn>=0.12.0, plotly>=5.15.0
- **RL Environment**: gymnasium>=0.29.0
- **Optimization**: pymoo>=0.6.0, optuna>=3.0.0

### 7.2 Running Experiments

#### Monthly VIX Evaluation (Full Configuration)
```bash
python experiments/evaluate_sentiment_rl.py \
  --baseline_mode monthly_vix \
  --vix_model_type mrjd \
  --num_simulations 100000 \
  --num_iterations 20 \
  --seed 42 \
  --goal_counts 1 2 4 8 16 \
  --batch_size 4800 \
  --learning_rate 0.01 \
  --hidden_dim 128 \
  --policy_type hierarchical \
  --value_type dual_head \
  --encoder_type attention \
  --use_real_ef \
  --force_recompute \
  --use_delta_adjustment \
  --volatility_method return_squared \
  --output_dir "data/results/monthly_vix_eval_100000"
```

#### Annual Stable Market Evaluation
```bash
python experiments/evaluate_sentiment_rl.py \
  --baseline_mode annual_stable \
  --use_real_ef \
  --force_recompute \
  --num_simulations 100000 \
  --goal_counts 1 2 4 8 16 \
  --output_dir "data/results/annual_stable_real_ef_vix_dp"
```

#### Monthly VIX (Information Only, No Adjustments)
```bash
python experiments/evaluate_sentiment_rl.py \
  --baseline_mode monthly_vix \
  --vix_model_type mrjd \
  --num_simulations 100000 \
  --num_iterations 20 \
  --seed 42 \
  --goal_counts 1 2 4 8 16 \
  --batch_size 4800 \
  --learning_rate 0.01 \
  --hidden_dim 128 \
  --policy_type hierarchical \
  --value_type dual_head \
  --encoder_type attention \
  --use_real_ef \
  --force_recompute \
  --volatility_method return_squared \
  --output_dir "data/results/monthly_vix_no_adjustments"
```

### 7.3 Command-Line Arguments

#### Essential Parameters
- `--baseline_mode`: Choose from `annual_stable`, `monthly_vix`
- `--vix_model_type`: VIX evolution model (`mrjd`, `mrs`)  
- `--num_simulations`: Monte Carlo sample size (default 100,000)
- `--goal_counts`: Goal configurations to evaluate
- `--seed`: Random seed for reproducibility

#### RL Architecture Options
- `--policy_type`: Policy architecture (`hierarchical`, `flat`)
- `--value_type`: Value function (`dual_head`, `single`)
- `--encoder_type`: State encoder (`attention`, `mlp`)
- `--hidden_dim`: Network hidden dimension
- `--learning_rate`: PPO learning rate

#### Market Configuration
- `--use_real_ef`: Use real market data for efficient frontier
- `--use_delta_adjustment`: Enable VIX-based return adjustments
- `--volatility_method`: Method for volatility estimation
- `--force_recompute`: Ignore cached results

### 7.4 Output Structure

#### Results Directory Layout
```
data/results/[experiment_name]/
├── evaluation_results.json          # Summary statistics
├── simulation_logs/                 # Detailed trajectory data  
├── training_logs/                   # RL training progress
├── figures/                         # Performance plots
└── config.json                      # Experiment configuration
```

#### Key Metrics
- **Efficiency scores**: Performance relative to optimal DP
- **Goal achievement rates**: Success probability by time horizon
- **Wealth distribution**: Final wealth statistics across simulations
- **Strategy analysis**: Portfolio allocation patterns

## 8. Reproducibility and Configuration

### 8.1 Deterministic Execution

#### Random Seed Control
- **Fixed seeds**: All stochastic components use controlled randomization
- **Shared sequences**: Identical market paths across algorithm comparison
- **Reproducible results**: Same configuration yields identical outcomes

#### Cache Management  
- `--force_recompute`: Ignore cached efficient frontier and β/δ parameters
- **Result preservation**: Previous results retained unless explicitly overwritten
- **Configuration tracking**: Full parameter sets saved with results

### 8.2 Computational Considerations

#### Performance Scaling
- **Large-scale simulations**: 100,000+ trajectories computationally intensive
- **Development workflow**: Start with 1,000-5,000 simulations for debugging
- **Parallel execution**: Automatic utilization of available cores

#### Memory Management
- **State discretization**: DP memory scales with wealth grid resolution
- **Trajectory storage**: Optional detailed logging for post-analysis
- **Batch processing**: RL training uses configurable batch sizes

### 8.3 Extensibility Framework

#### Modular Design
The system supports straightforward extensions:

- **Alternative VIX models**: New volatility processes in `src/models/`
- **Different market regimes**: Extended regime-switching frameworks
- **Goal structures**: Alternative utility functions and time horizons  
- **RL architectures**: New policy and value function designs

#### Configuration Management
- **Experiment configs**: Centralized parameter management
- **Command-line interface**: Full parameter control via CLI
- **Result organization**: Systematic output directory structure

---

## Summary

This comprehensive framework integrates modern reinforcement learning with traditional financial optimization, enabling systematic evaluation of sentiment-aware investment strategies. The modular design supports extensive experimentation while maintaining rigorous scientific standards through controlled randomization and reproducible configurations.

The key innovation—incorporating VIX-based market sentiment into RL state representations—demonstrates how behavioral finance insights can enhance algorithmic trading systems. The framework provides a robust foundation for goals-based wealth management research under uncertainty.