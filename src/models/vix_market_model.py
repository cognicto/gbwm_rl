"""
VIX Market Models with Shared Random Shocks

This module implements two VIX modeling approaches:
1. Mean-Reverting Jump-Diffusion (MRJD) - Based on academic literature
2. Markov Regime-Switching (MRS) - Best for crisis patterns

CRITICAL: Both models use the SAME random shocks as market simulations
to ensure consistent comparison between Pure RL and Sentiment RL.

=== VIX-Return Correlation (Based on Heston Model Research) ===

From academic research (Heston model, CBOE studies):
- VIX and stock returns are CONTEMPORANEOUSLY correlated (same time)
- The correlation is negative: when stocks fall, VIX rises
- Typical correlation: -60% to -80%
- Stock returns are the fundamental driver of VIX changes

Mathematical Foundation (Heston Model):
    dS_t = r S_t dt + √V_t S_t dW₁(t)     # Stock price
    dV_t = κ(θ - V_t) dt + η √V_t dW₂(t)  # Variance/VIX
    E[dW₁ dW₂] = ρ dt  where ρ ≈ -0.7 to -0.8

Key Insight: dW₁ and dW₂ are correlated at the SAME time t!
- VIX_t is correlated with z_t (same period shock)
- This is NOT lagged - they move together (inversely) at the same moment

Implementation:
- VIX model uses market_shocks[t] to compute VIX_t
- Wealth evolution uses the SAME market_shocks[t]
- Agent observes VIX_t BEFORE making decision for period t
- High VIX_t indicates negative z_t → poor returns in period t

Key Design:
- VIX models receive external `market_shocks` array during initialization
- The same shocks are used by DP, Pure RL, and Sentiment RL for fair comparison
- VIX is correlated with market shocks (high VIX ↔ negative market)

References:
- Heston Model: https://en.wikipedia.org/wiki/Heston_model
- CBOE VIX-S&P Correlation: https://www.cboe.com/insights/posts/inside-volatility-trading-breaking-down-the-vix-index-and-its-correlation-to-the-s-p-500-index/
- MRJD: https://link.springer.com/article/10.1007/s11156-009-0153-8
- MRS: https://onlinelibrary.wiley.com/doi/10.1002/fut.70041
- Regime-Switching Heston: https://economics.princeton.edu/published-papers/a-regime-switching-heston-model-for-vix-and-sp-500-implied-volatilities/
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
from enum import Enum


class VIXModelType(Enum):
    """Available VIX model types"""
    MRJD = "mrjd"  # Mean-Reverting Jump-Diffusion
    REGIME_SWITCHING = "regime_switching"  # Markov Regime-Switching


@dataclass
class VIXModelParams:
    """
    Parameters for VIX dynamics and VIX → Returns relationship

    Common to both MRJD and Regime-Switching models.
    """
    # Long-term VIX mean
    theta: float = 20.0

    # VIX → Returns parameters (calibrated to paper's regime effects)
    beta_sensitivity: float = 0.04    # How much VIX affects μ (stronger impact)
    delta_sensitivity: float = 0.06   # How much VIX affects σ (higher volatility impact)

    # VIX-Market correlation (how much market shocks affect VIX)
    # Higher = stronger relationship: negative market shock → VIX spike
    # Paper shows -60% to -80% correlation between VIX and stock returns
    vix_market_correlation: float = 0.7

    # Bounds (based on paper's empirical observations)
    vix_min: float = 9.0    # Paper shows VIX can go below 10% in calm periods
    vix_max: float = 85.0   # Paper documents VIX reaching 80%+ in 2008 crisis


@dataclass
class MRJDParams(VIXModelParams):
    """
    Mean-Reverting Jump-Diffusion specific parameters

    dV = κ(θ - V)dt + σ_v × V^β × dW + J × dN
    """
    # Mean reversion
    kappa: float = 3.0              # Mean reversion speed (fast reversion)
    sigma_v: float = 0.8            # Vol-of-vol
    beta_power: float = 0.5         # Level effect (√V like CIR)

    # Jump parameters (calibrated to match Princeton paper empirical findings)
    lambda_jump: float = 1.5        # Jump intensity (jumps/year) - increased for crisis modeling
    mu_jump: float = 20.0           # Mean jump size - larger for realistic VIX spikes
    sigma_jump: float = 15.0        # Jump size volatility - higher for crisis periods


@dataclass
class RegimeSwitchingParams(VIXModelParams):
    """
    Markov Regime-Switching specific parameters

    Three regimes: Tranquil, Turmoil, Crisis
    """
    # Regime parameters: {regime_name: {mean, std, mean_reversion, jump_prob, jump_size}}
    regimes: Dict = field(default_factory=lambda: {
        'tranquil': {
            'mean': 14.0,           # Low VIX environment
            'std': 3.0,
            'mean_reversion': 5.0,  # Fast reversion
            'jump_prob': 0.02,
            'jump_size': 5.0
        },
        'turmoil': {
            'mean': 25.0,           # Elevated VIX
            'std': 6.0,
            'mean_reversion': 2.5,  # Moderate reversion (paper: κ ~3-5)
            'jump_prob': 0.10,      # Higher jump probability
            'jump_size': 12.0       # Larger jumps in turmoil
        },
        'crisis': {
            'mean': 50.0,           # Crisis VIX (paper shows 50-80% levels)
            'std': 15.0,            # Higher volatility in crisis
            'mean_reversion': 1.2,  # Very slow reversion (paper: low exit probability)
            'jump_prob': 0.20,      # High jump probability in crisis
            'jump_size': 25.0       # Large crisis jumps
        }
    })

    # Transition matrix (annual probabilities)
    # Calibrated based on Princeton paper's crisis vs non-crisis findings
    # Rows: from state, Columns: to state [Tranquil, Turmoil, Crisis]
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([
        #  Tranquil  Turmoil  Crisis
        [    0.88,    0.10,    0.02],  # From Tranquil (stable normal periods)
        [    0.25,    0.65,    0.10],  # From Turmoil (elevated uncertainty)
        [    0.03,    0.22,    0.75],  # From Crisis (high persistence, paper Table 5)
    ]))


class BaseVIXModel(ABC):
    """
    Abstract base class for VIX models

    All VIX models must:
    1. Accept external market_shocks for consistency
    2. Implement step_vix() to evolve VIX
    3. Provide get_sentiment_features() for Sentiment RL state
    """

    def __init__(
        self,
        params: VIXModelParams,
        market_shocks: Optional[np.ndarray] = None,
        seed: int = 42
    ):
        """
        Initialize VIX model

        Args:
            params: Model-specific parameters
            market_shocks: Pre-generated market shocks for consistency.
                          Shape: (num_simulations, time_horizon) or (time_horizon,)
                          If provided, VIX will be correlated with these shocks.
            seed: Random seed for reproducibility
        """
        self.params = params
        self.market_shocks = market_shocks
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # State
        self.current_vix = params.theta
        self.vix_history: List[float] = []
        self.current_step = 0
        self.current_sim_idx = 0

    def reset(self, episode_idx: int = 0, sim_idx: int = 0) -> float:
        """
        Reset VIX for new episode

        Args:
            episode_idx: Episode index for seed variation
            sim_idx: Simulation index (for accessing correct market_shocks row)

        Returns:
            Initial VIX value
        """
        self.current_sim_idx = sim_idx
        self.current_step = 0
        self.rng = np.random.RandomState(self.seed + episode_idx)

        # Initialize VIX with some randomness around mean
        initial_vix = self.params.theta + self.rng.normal(0, 3)
        self.current_vix = np.clip(initial_vix, self.params.vix_min, self.params.vix_max)
        self.vix_history = [self.current_vix]

        return self.current_vix

    def get_market_shock(self) -> float:
        """
        Get the market shock for LEADING indicator behavior.

        VIX as LEADING indicator means:
        - VIX observed at time t predicts returns in period t
        - High VIX at t → expect negative z_t → poor wealth evolution in period t

        Implementation:
        - current_step tracks which shock we're using
        - Returns market_shocks[current_step] which will affect wealth in the same period
        - This ensures VIX uses the SAME random shock as DP/RL simulations
        """
        if self.market_shocks is None:
            return self.rng.normal(0, 1)

        if self.market_shocks.ndim == 1:
            # Shape: (time_horizon,) - single simulation
            if self.current_step < len(self.market_shocks):
                return self.market_shocks[self.current_step]
        else:
            # Shape: (num_simulations, time_horizon)
            if (self.current_sim_idx < self.market_shocks.shape[0] and
                self.current_step < self.market_shocks.shape[1]):
                return self.market_shocks[self.current_sim_idx, self.current_step]

        return self.rng.normal(0, 1)

    @abstractmethod
    def step_vix(self, dt: float = 1.0, market_shock: Optional[float] = None) -> float:
        """
        Evolve VIX by one time step

        Args:
            dt: Time step (1.0 = 1 year)
            market_shock: Optional market shock for correlation (during training).
                         If provided, this shock will be used instead of generating a new one.
                         Critical for ensuring VIX-wealth correlation during training.

        Returns:
            New VIX value
        """
        pass

    def get_vix_average(self, lookback: int = 2) -> float:
        """
        Get average VIX over last `lookback` periods
        """
        if len(self.vix_history) >= lookback:
            return np.mean(self.vix_history[-lookback:])
        elif len(self.vix_history) > 0:
            return np.mean(self.vix_history)
        else:
            return self.params.theta

    def get_adjustments(self, vix_avg: Optional[float] = None) -> Tuple[float, float]:
        """
        Compute β(VIX) and δ(VIX) adjustments

        VIX → Returns relationship:
            β = β_sensitivity × (θ - VIX_avg) / θ
            δ = δ_sensitivity × (θ - VIX_avg) / θ

        When VIX_avg > θ (fear):  β < 0, δ < 0  → Lower μ, Higher σ
        When VIX_avg < θ (greed): β > 0, δ > 0  → Higher μ, Lower σ
        """
        if vix_avg is None:
            vix_avg = self.get_vix_average()

        p = self.params
        vix_normalized = (p.theta - vix_avg) / p.theta

        beta = p.beta_sensitivity * vix_normalized
        delta = p.delta_sensitivity * vix_normalized

        return beta, delta

    def get_adjusted_params(
        self,
        base_mu: float,
        base_sigma: float,
        vix_avg: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Get VIX-adjusted μ and σ
        """
        beta, delta = self.get_adjustments(vix_avg)

        mu_adj = base_mu + beta
        sigma_adj = base_sigma - delta  # Higher VIX → lower delta → higher sigma

        # Bounds
        mu_adj = np.clip(mu_adj, -0.15, 0.30)
        sigma_adj = np.clip(sigma_adj, 0.02, 0.60)

        return mu_adj, sigma_adj

    def get_sentiment_features(self) -> np.ndarray:
        """
        Get VIX-based features for Sentiment RL agent's state

        Returns:
            Array of [vix_level_norm, vix_avg_norm, vix_momentum]
            - vix_level_norm: Current VIX normalized around θ
            - vix_avg_norm: Average VIX normalized around θ
            - vix_momentum: Rate of change (current vs previous), in [-1, 1]
        """
        vix_avg = self.get_vix_average()

        # Normalized VIX level (centered at θ)
        vix_level_norm = (self.current_vix - self.params.theta) / self.params.theta
        vix_avg_norm = (vix_avg - self.params.theta) / self.params.theta

        # VIX momentum: rate of change from previous value
        if len(self.vix_history) >= 2:
            prev_vix = self.vix_history[-2]
            if prev_vix > 0:
                vix_momentum = (self.current_vix - prev_vix) / prev_vix
            else:
                vix_momentum = 0.0
        else:
            vix_momentum = 0.0

        # Clip to reasonable bounds
        vix_level_norm = np.clip(vix_level_norm, -0.5, 3.0)
        vix_avg_norm = np.clip(vix_avg_norm, -0.5, 3.0)
        vix_momentum = np.clip(vix_momentum, -1.0, 1.0)

        return np.array([vix_level_norm, vix_avg_norm, vix_momentum], dtype=np.float32)


class MRJDVIXModel(BaseVIXModel):
    """
    Mean-Reverting Jump-Diffusion VIX Model

    dV = κ(θ - V)dt + σ_v × V^β × dW + J × dN

    Based on: https://link.springer.com/article/10.1007/s11156-009-0153-8

    Key features:
    - Mean reversion to long-term level θ
    - Level-dependent volatility (√V like CIR)
    - Jump component for crisis events
    - Correlated with market shocks
    """

    def __init__(
        self,
        params: Optional[MRJDParams] = None,
        market_shocks: Optional[np.ndarray] = None,
        seed: int = 42
    ):
        params = params or MRJDParams()
        super().__init__(params, market_shocks, seed)
        self.mrjd_params = params

    def step_vix(self, dt: float = 1.0, market_shock: Optional[float] = None) -> float:
        """
        Evolve VIX using Mean-Reverting Jump-Diffusion

        dV = κ(θ - V)dt + σ_v × V^β × dW + J × dN

        The diffusion component dW is correlated with market shocks:
        - Negative market shock → positive VIX diffusion
        
        Args:
            dt: Time step (1.0 = 1 year)
            market_shock: Optional external market shock for correlation.
                         If provided, this shock will be used for VIX-wealth correlation.
                         Critical for ensuring proper correlation during training.
        """
        V = self.current_vix
        p = self.mrjd_params

        # Use external shock if provided (during training), otherwise get from array
        if market_shock is not None:
            # CRITICAL FIX: Use the same shock as wealth evolution (during training)
            shock_for_correlation = market_shock
        else:
            # Use pre-generated shocks (during simulation)
            shock_for_correlation = self.get_market_shock()

        # ═══════════════════════════════════════════════════════
        # MEAN REVERSION: κ(θ - V)dt
        # ═══════════════════════════════════════════════════════
        drift = p.kappa * (p.theta - V) * dt

        # ═══════════════════════════════════════════════════════
        # DIFFUSION: σ_v × V^β × dW (correlated with market)
        # Negative market shock → positive VIX change
        # ═══════════════════════════════════════════════════════
        # Generate VIX-specific randomness
        dW_independent = self.rng.normal(0, np.sqrt(dt))

        # Correlated shock: combine market shock with independent component
        # Negative market shock → positive contribution to VIX
        dW = (-p.vix_market_correlation * shock_for_correlation * np.sqrt(dt) +
              np.sqrt(1 - p.vix_market_correlation**2) * dW_independent)

        diffusion = p.sigma_v * (max(V, 1) ** p.beta_power) * dW

        # ═══════════════════════════════════════════════════════
        # JUMPS: J × dN (more likely after negative market shock)
        # ═══════════════════════════════════════════════════════
        # Increase jump probability after negative market shocks
        jump_prob_adj = p.lambda_jump * dt
        if shock_for_correlation < -1.5:  # Severe negative shock
            jump_prob_adj *= 3.0
        elif shock_for_correlation < -1.0:
            jump_prob_adj *= 2.0
        elif shock_for_correlation < -0.5:
            jump_prob_adj *= 1.5

        n_jumps = self.rng.poisson(jump_prob_adj)
        jump = 0.0
        if n_jumps > 0:
            jump = self.rng.normal(p.mu_jump, p.sigma_jump)
            jump = max(0, jump)  # VIX jumps are typically upward

        # Update VIX
        V_new = V + drift + diffusion + jump
        V_new = np.clip(V_new, p.vix_min, p.vix_max)

        self.current_vix = V_new
        self.vix_history.append(V_new)
        self.current_step += 1

        return V_new


class RegimeSwitchingVIXModel(BaseVIXModel):
    """
    Markov Regime-Switching VIX Model

    Three regimes: Tranquil, Turmoil, Crisis
    Each regime has different VIX dynamics.
    Transitions are influenced by market shocks.

    Based on:
    - https://onlinelibrary.wiley.com/doi/10.1002/fut.70041
    - https://economics.princeton.edu/published-papers/a-regime-switching-heston-model-for-vix-and-sp-500-implied-volatilities/
    """

    REGIME_NAMES = ['tranquil', 'turmoil', 'crisis']

    def __init__(
        self,
        params: Optional[RegimeSwitchingParams] = None,
        market_shocks: Optional[np.ndarray] = None,
        seed: int = 42
    ):
        params = params or RegimeSwitchingParams()
        super().__init__(params, market_shocks, seed)
        self.rs_params = params
        self.current_regime = 'tranquil'
        self.regime_history: List[str] = []

    def reset(self, episode_idx: int = 0, sim_idx: int = 0) -> float:
        """Reset with initial regime determination"""
        initial_vix = super().reset(episode_idx, sim_idx)

        # Determine initial regime based on VIX level
        if initial_vix < 20:
            self.current_regime = 'tranquil'
        elif initial_vix < 35:
            self.current_regime = 'turmoil'
        else:
            self.current_regime = 'crisis'

        self.regime_history = [self.current_regime]
        return initial_vix

    def _transition_regime(self, market_shock: float) -> str:
        """
        Determine regime transition based on market shock

        Negative market shocks increase probability of transitioning
        to higher-volatility regimes.
        """
        regime_idx = {'tranquil': 0, 'turmoil': 1, 'crisis': 2}

        # Get base transition probabilities
        probs = self.rs_params.transition_matrix[regime_idx[self.current_regime]].copy()

        # Adjust probabilities based on market shock
        if market_shock < -2:  # Severe negative shock
            probs[2] += 0.30  # Much more likely to enter crisis
            probs[0] -= 0.20
        elif market_shock < -1:  # Moderate negative shock
            probs[2] += 0.15
            probs[1] += 0.10
            probs[0] -= 0.15
        elif market_shock < -0.5:  # Mild negative shock
            probs[1] += 0.10
            probs[0] -= 0.05
        elif market_shock > 1:  # Strong positive market
            probs[0] += 0.15  # More likely to return to tranquil
            probs[2] -= 0.10
        elif market_shock > 0.5:  # Moderate positive market
            probs[0] += 0.08
            probs[2] -= 0.05

        # Ensure valid probabilities
        probs = np.clip(probs, 0.01, 0.99)
        probs /= probs.sum()

        # Sample new regime
        new_regime = self.rng.choice(self.REGIME_NAMES, p=probs)
        return new_regime

    def step_vix(self, dt: float = 1.0, market_shock: Optional[float] = None) -> float:
        """
        Evolve VIX using Regime-Switching dynamics

        1. Get market shock (same as other methods)
        2. Transition regime (influenced by market shock)
        3. Apply regime-specific mean-reversion
        4. Add regime-specific jumps
        
        Args:
            dt: Time step (1.0 = 1 year)
            market_shock: Optional external market shock for correlation.
                         If provided, this shock will be used for VIX-wealth correlation.
                         Critical for ensuring proper correlation during training.
        """
        V = self.current_vix

        # Use external shock if provided (during training), otherwise get from array
        if market_shock is not None:
            # CRITICAL FIX: Use the same shock as wealth evolution (during training)
            shock_for_correlation = market_shock
        else:
            # Use pre-generated shocks (during simulation)
            shock_for_correlation = self.get_market_shock()

        # ═══════════════════════════════════════════════════════
        # REGIME TRANSITION (influenced by market shock)
        # ═══════════════════════════════════════════════════════
        self.current_regime = self._transition_regime(shock_for_correlation)
        self.regime_history.append(self.current_regime)

        regime_params = self.rs_params.regimes[self.current_regime]

        # ═══════════════════════════════════════════════════════
        # MEAN REVERSION within regime
        # ═══════════════════════════════════════════════════════
        kappa = regime_params['mean_reversion']
        theta = regime_params['mean']
        sigma = regime_params['std']

        drift = kappa * (theta - V) * dt

        # ═══════════════════════════════════════════════════════
        # DIFFUSION (correlated with market shock)
        # ═══════════════════════════════════════════════════════
        dW_independent = self.rng.normal(0, np.sqrt(dt))

        # Negative market shock → positive VIX diffusion
        corr = self.rs_params.vix_market_correlation
        dW = (-corr * shock_for_correlation * np.sqrt(dt) +
              np.sqrt(1 - corr**2) * dW_independent)

        diffusion = sigma * dW

        # ═══════════════════════════════════════════════════════
        # REGIME-SPECIFIC JUMPS
        # More likely in turmoil/crisis and after negative shocks
        # ═══════════════════════════════════════════════════════
        jump = 0.0
        jump_prob = regime_params['jump_prob']

        # Increase jump probability after negative market shocks
        if shock_for_correlation < -1.5:
            jump_prob *= 2.5
        elif shock_for_correlation < -1.0:
            jump_prob *= 1.8
        elif shock_for_correlation < -0.5:
            jump_prob *= 1.3

        if self.rng.random() < jump_prob:
            jump = self.rng.exponential(regime_params['jump_size'])

        # Update VIX
        V_new = V + drift + diffusion + jump
        V_new = np.clip(V_new, self.params.vix_min, self.params.vix_max)

        self.current_vix = V_new
        self.vix_history.append(V_new)
        self.current_step += 1

        return V_new

    def get_regime_history(self) -> List[str]:
        """Get the history of regime states"""
        return self.regime_history


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_vix_model(
    model_type: str = "mrjd",
    market_shocks: Optional[np.ndarray] = None,
    seed: int = 42,
    **kwargs
) -> BaseVIXModel:
    """
    Factory function to create VIX model

    Args:
        model_type: "mrjd" or "regime_switching"
        market_shocks: Pre-generated market shocks for consistency
                      Shape: (num_simulations, time_horizon) or (time_horizon,)
        seed: Random seed
        **kwargs: Additional parameters passed to model

    Returns:
        VIX model instance

    Example:
        # Create MRJD model with market correlation
        vix_model = create_vix_model(
            model_type="mrjd",
            market_shocks=market_shocks,
            seed=42
        )

        # Create Regime-Switching model
        vix_model = create_vix_model(
            model_type="regime_switching",
            market_shocks=market_shocks,
            seed=42
        )
    """
    model_type = model_type.lower()

    if model_type in ["mrjd", "mean_reverting", "jump_diffusion"]:
        params = MRJDParams(**{k: v for k, v in kwargs.items()
                               if hasattr(MRJDParams, k)})
        return MRJDVIXModel(params=params, market_shocks=market_shocks, seed=seed)

    elif model_type in ["regime_switching", "regime", "mrs", "markov"]:
        params = RegimeSwitchingParams(**{k: v for k, v in kwargs.items()
                                          if hasattr(RegimeSwitchingParams, k)})
        return RegimeSwitchingVIXModel(params=params, market_shocks=market_shocks, seed=seed)

    else:
        raise ValueError(f"Unknown VIX model type: {model_type}. "
                        f"Use 'mrjd' or 'regime_switching'.")


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

# Alias for backward compatibility
VIXMarketModel = MRJDVIXModel
VIXModelParams = MRJDParams


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("VIX Market Model Test - Shared Random Shocks")
    print("=" * 70)

    # Generate market shocks (same as would be used by DP/RL)
    np.random.seed(42)
    time_horizon = 16
    num_sims = 3
    market_shocks = np.random.normal(0, 1, size=(num_sims, time_horizon))

    print("\nMarket shocks (first simulation):")
    print([f"{s:+.2f}" for s in market_shocks[0]])

    # Test MRJD Model
    print("\n" + "=" * 70)
    print("Test 1: Mean-Reverting Jump-Diffusion (MRJD)")
    print("=" * 70)

    mrjd_model = create_vix_model(
        model_type="mrjd",
        market_shocks=market_shocks,
        seed=42
    )
    mrjd_model.reset(episode_idx=0, sim_idx=0)

    print("\nYear | Market | VIX     | VIX Avg | Features")
    print("-" * 60)

    for t in range(time_horizon):
        shock = market_shocks[0, t]
        vix_before = mrjd_model.current_vix
        vix_new = mrjd_model.step_vix(dt=1.0)
        vix_avg = mrjd_model.get_vix_average()
        features = mrjd_model.get_sentiment_features()

        print(f"{t+1:4d} | {shock:+.2f}  | {vix_before:5.1f}→{vix_new:5.1f} | "
              f"{vix_avg:5.1f}   | [{features[0]:+.2f}, {features[1]:+.2f}]")

    # Test Regime-Switching Model
    print("\n" + "=" * 70)
    print("Test 2: Markov Regime-Switching")
    print("=" * 70)

    rs_model = create_vix_model(
        model_type="regime_switching",
        market_shocks=market_shocks,
        seed=42
    )
    rs_model.reset(episode_idx=0, sim_idx=0)

    print("\nYear | Market | VIX     | Regime    | Features")
    print("-" * 65)

    for t in range(time_horizon):
        shock = market_shocks[0, t]
        vix_before = rs_model.current_vix
        vix_new = rs_model.step_vix(dt=1.0)
        regime = rs_model.current_regime
        features = rs_model.get_sentiment_features()

        print(f"{t+1:4d} | {shock:+.2f}  | {vix_before:5.1f}→{vix_new:5.1f} | "
              f"{regime:9s} | [{features[0]:+.2f}, {features[1]:+.2f}]")

    # Test consistency: same shocks → correlated VIX
    print("\n" + "=" * 70)
    print("Test 3: Verify VIX-Market Correlation")
    print("=" * 70)

    # Run multiple simulations
    vix_changes = []
    market_returns = []

    for sim_idx in range(num_sims):
        mrjd_model.reset(episode_idx=sim_idx, sim_idx=sim_idx)

        for t in range(time_horizon):
            vix_before = mrjd_model.current_vix
            vix_new = mrjd_model.step_vix(dt=1.0)

            vix_changes.append(vix_new - vix_before)
            market_returns.append(market_shocks[sim_idx, t])

    correlation = np.corrcoef(market_returns, vix_changes)[0, 1]
    print(f"\nCorrelation between market shocks and VIX changes: {correlation:.3f}")
    print("(Negative correlation expected: negative market → VIX increase)")

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)