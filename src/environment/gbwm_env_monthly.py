"""
GBWM Environment with Monthly Time Steps and Correct VIX Causality

Key improvements over yearly version:
1. Monthly time steps (192 steps = 16 years × 12 months)
2. More dynamic VIX learning opportunities
3. Explicit β and δ tracking in info dict
4. Same time resolution for both Pure RL and Sentiment RL

VIX → Returns Causality:
- VIX follows mean-reverting jump-diffusion (independent)
- VIX history PREDICTS returns via β(VIX) and δ(VIX)
- μ_adj = μ + β, σ_adj = σ - δ
- Return: R = (μ_adj - ½σ_adj²)Δt + σ_adj√Δt × Z
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List
import logging

from src.models.vix_market_model import VIXMarketModel, VIXModelParams, create_vix_model
from config.environment_config import DEFAULT_ENV_CONFIG

logger = logging.getLogger(__name__)


class GBWMEnvMonthly(gym.Env):
    """
    Goals-Based Wealth Management Environment with Monthly Time Steps

    Time Resolution:
        - 192 monthly steps (16 years × 12 months)
        - dt = 1/12 (one month)
        - Goals available at months 48, 96, 144, 192 (years 4, 8, 12, 16)

    State space:
        Pure RL:      [normalized_time, normalized_wealth]
        Sentiment RL: [normalized_time, normalized_wealth, vix_level, vix_avg, vix_momentum]

    Action space:
        [goal_decision (0/1), portfolio_choice (0-14)]

    VIX → Returns (Correct Causality):
        - Agent sees VIX at BEGINNING of each month
        - VIX determines β and δ adjustments
        - μ_adj = μ + β(VIX), σ_adj = σ - δ(VIX)
        - Return: R = (μ_adj - ½σ_adj²)Δt + σ_adj√Δt × Z
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_goals: int = 4,
        years_horizon: int = 16,
        use_sentiment: bool = True,
        use_vix_market_adjustments: bool = False,
        vix_params: Optional[VIXModelParams] = None,
        portfolio_betas: Optional[np.ndarray] = None,
        portfolio_deltas: Optional[np.ndarray] = None,
        volatility_method: str = 'rolling_vol',
        vix_model_type: str = 'mrjd',
        config: Any = None
    ):
        super().__init__()

        self.num_goals = num_goals
        self.years_horizon = years_horizon
        self.months_per_year = 12
        self.total_months = years_horizon * self.months_per_year  # 192 months
        self.dt = 1.0 / self.months_per_year  # Monthly time step

        self.use_sentiment = use_sentiment
        self.use_vix_market_adjustments = use_vix_market_adjustments
        self.config = config or DEFAULT_ENV_CONFIG

        # Volatility method determines how δ is applied:
        # - 'rolling_vol': δ adjusts σ directly → σ_adj = σ - δ
        # - 'return_squared': δ adjusts variance → σ_adj = √(σ² - δ)
        self.volatility_method = volatility_method
        if volatility_method not in ['rolling_vol', 'return_squared']:
            raise ValueError(f"volatility_method must be 'rolling_vol' or 'return_squared', got '{volatility_method}'")

        # VIX market model - use factory function for configurability
        self.vix_params = vix_params or VIXModelParams()
        self.vix_model_type = vix_model_type
        
        # Create VIX model using factory function to support both MRJD and regime-switching
        # Note: During training, we don't have market_shocks pre-generated, so VIX model
        # will generate its own shocks. This is consistent with training behavior.
        self.vix_model = create_vix_model(
            model_type=vix_model_type,
            market_shocks=None,  # Will be set later during simulation if needed
            seed=42,  # Will be overridden during reset
            kappa=self.vix_params.kappa,
            theta=self.vix_params.theta,
            sigma_v=self.vix_params.sigma_v,
            lambda_jump=self.vix_params.lambda_jump,
            mu_jump=self.vix_params.mu_jump,
            sigma_jump=self.vix_params.sigma_jump,
            beta_sensitivity=self.vix_params.beta_sensitivity,
            delta_sensitivity=self.vix_params.delta_sensitivity
        )

        # Portfolio-specific β and δ sensitivity arrays
        # These are the learned coefficients from beta_delta_learner
        # β_adj = β_sensitivity[portfolio] × (θ - VIX_avg) / θ
        # δ_adj = δ_sensitivity[portfolio] × (θ - VIX_avg) / θ
        self.portfolio_betas = portfolio_betas  # (num_portfolios,) array or None
        self.portfolio_deltas = portfolio_deltas  # (num_portfolios,) array or None
        self.use_portfolio_specific_beta_delta = (
            portfolio_betas is not None and portfolio_deltas is not None
        )

        # Portfolio parameters - convert from annual to monthly
        annual_means = np.array(self.config.portfolio_config.mean_returns)
        annual_stds = np.array(self.config.portfolio_config.return_stds)

        # Monthly parameters: μ_m = μ_a/12, σ_m = σ_a/√12
        self.portfolio_means_monthly = annual_means / 12
        self.portfolio_stds_monthly = annual_stds / np.sqrt(12)

        # Keep annual for reference
        self.portfolio_means_annual = annual_means
        self.portfolio_stds_annual = annual_stds
        self.num_portfolios = len(annual_means)

        # Goal configuration (in months)
        self._setup_goals()

        # Initial wealth (scales with number of goals)
        self.initial_wealth = 12 * (num_goals ** 0.85) * 10000

        # State and action spaces
        self._setup_spaces()

        # Episode state
        self.current_month = 0
        self.current_wealth = self.initial_wealth
        self.goals_taken = []
        self.total_utility = 0.0
        self.episode_idx = 0

        # Episode-scoped random generator for deterministic training
        # This ensures each episode gets the same shock sequence across training iterations
        self.episode_rng: Optional[np.random.Generator] = None
        self.base_seed = 42  # Default base seed, can be overridden in reset()

        # Track β and δ for analysis
        self.beta_history = []
        self.delta_history = []

        logger.info(f"GBWMEnvMonthly initialized: "
                    f"use_sentiment={use_sentiment}, num_goals={num_goals}, "
                    f"total_months={self.total_months}")

    def _setup_goals(self):
        """Setup goal months based on num_goals"""
        if self.num_goals == 1:
            self.goal_years = [16]
        elif self.num_goals == 2:
            self.goal_years = [8, 16]
        elif self.num_goals == 4:
            self.goal_years = [4, 8, 12, 16]
        elif self.num_goals == 8:
            self.goal_years = [2, 4, 6, 8, 10, 12, 14, 16]
        else:
            step = self.years_horizon // self.num_goals
            self.goal_years = [(i + 1) * step for i in range(self.num_goals)]

        # Convert to months (goals available at end of goal year)
        self.goal_months = [y * self.months_per_year for y in self.goal_years]

    def _setup_spaces(self):
        """Setup observation and action spaces"""
        if self.use_sentiment:
            # Sentiment RL: [time, wealth, vix_level, vix_avg, vix_momentum]
            self.observation_space = spaces.Box(
                low=np.array([0, 0, -0.5, -0.5, -1.0], dtype=np.float32),
                high=np.array([1, 1, 3.0, 3.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Pure RL: [time, wealth]
            self.observation_space = spaces.Box(
                low=np.array([0, 0], dtype=np.float32),
                high=np.array([1, 1], dtype=np.float32),
                dtype=np.float32
            )

        # Action: [goal_decision, portfolio_choice]
        self.action_space = spaces.MultiDiscrete([2, self.num_portfolios])

    def _get_goal_cost(self, year: int) -> float:
        """Goal cost grows at 8% per year"""
        return 100000 * (1.08 ** year)

    def _get_goal_utility(self, year: int) -> float:
        """Goal utility: 10 + year"""
        return 10 + year

    def _is_goal_month(self, month: int) -> bool:
        """Check if this month has a goal available"""
        return month in self.goal_months and month not in [
            m for m in self.goal_months if self._month_to_year(m) in self.goals_taken
        ]

    def _month_to_year(self, month: int) -> int:
        """Convert month to year (1-indexed)"""
        return month // self.months_per_year

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Normalized time [0, 1] over total months
        norm_time = self.current_month / self.total_months

        # Normalized wealth [0, 1] (capped at 10x initial)
        norm_wealth = min(self.current_wealth / (self.initial_wealth * 10), 1.0)

        if self.use_sentiment:
            # Get VIX features
            vix_features = self.vix_model.get_sentiment_features()
            return np.array([norm_time, norm_wealth] + list(vix_features), dtype=np.float32)
        else:
            return np.array([norm_time, norm_wealth], dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode with episode-scoped random generator"""
        super().reset(seed=seed)

        self.episode_idx += 1

        # Create episode-specific random generator for deterministic training
        # This ensures each episode gets the same shock sequence across training iterations
        if seed is not None:
            self.base_seed = seed
        
        episode_seed = self.base_seed + self.episode_idx
        self.episode_rng = np.random.default_rng(episode_seed)
        
        # Reset VIX model with episode-specific seed for consistency
        # The VIX model uses its base seed + episode_idx for deterministic reset
        # This ensures VIX evolution is also deterministic per episode
        self.vix_model.reset(episode_idx=self.episode_idx)

        # Reset state
        self.current_month = 0
        self.current_wealth = self.initial_wealth
        self.goals_taken = []
        self.total_utility = 0.0

        # Reset history
        self.beta_history = []
        self.delta_history = []

        observation = self._get_observation()

        # Get initial β and δ
        beta, delta = self.vix_model.get_adjustments()

        info = {
            'month': self.current_month,
            'year': self.current_month / self.months_per_year,
            'wealth': self.current_wealth,
            'vix': self.vix_model.current_vix,
            'vix_avg': self.vix_model.get_vix_average(),
            'beta': beta,
            'delta': delta,
            'goals_available': [m // self.months_per_year for m in self.goal_months
                               if self._month_to_year(m) not in self.goals_taken]
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one monthly time step

        Timeline for each month:
        1. Agent sees VIX at BEGINNING of month
        2. Agent chooses action based on observation
        3. Execute goal decision (if goal month)
        4. Generate return with VIX-adjusted μ, σ (if use_sentiment)
        5. Update wealth
        6. Advance VIX for next month

        Args:
            action: [goal_decision, portfolio_choice]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        goal_action, portfolio_action = int(action[0]), int(action[1])

        reward = 0.0
        wealth_after_goal = self.current_wealth

        # ═══════════════════════════════════════════════════════
        # STEP 1: Execute goal decision (only on goal months)
        # ═══════════════════════════════════════════════════════
        current_year = self._month_to_year(self.current_month + 1)

        if self._is_goal_month(self.current_month + 1):  # Check next month (1-indexed)
            goal_cost = self._get_goal_cost(current_year)
            goal_utility = self._get_goal_utility(current_year)

            if goal_action == 1 and self.current_wealth >= goal_cost:
                wealth_after_goal = self.current_wealth - goal_cost
                reward = goal_utility
                self.goals_taken.append(current_year)
                self.total_utility += goal_utility

        # ═══════════════════════════════════════════════════════
        # STEP 2: Get β and δ from VIX (BEFORE return generation)
        # ═══════════════════════════════════════════════════════
        vix_avg = self.vix_model.get_vix_average()

        # Compute VIX normalization: (θ - VIX_avg) / θ
        # Positive when VIX < θ (greed), negative when VIX > θ (fear)
        vix_normalized = (self.vix_params.theta - vix_avg) / self.vix_params.theta

        if self.use_portfolio_specific_beta_delta:
            # Use portfolio-specific β/δ sensitivity
            # β_adj = β_sensitivity[portfolio] × vix_normalized
            # δ_adj = δ_sensitivity[portfolio] × vix_normalized
            beta = self.portfolio_betas[portfolio_action] * vix_normalized
            delta = self.portfolio_deltas[portfolio_action] * vix_normalized
        else:
            # Fallback to VIX model's single sensitivity values
            beta, delta = self.vix_model.get_adjustments(vix_avg)

        # Track for analysis
        self.beta_history.append(beta)
        self.delta_history.append(delta)

        # ═══════════════════════════════════════════════════════
        # STEP 3: Portfolio evolution with VIX-adjusted returns
        # ═══════════════════════════════════════════════════════
        if self.current_month < self.total_months - 1:
            # Use ANNUAL parameters and apply dt scaling
            # This avoids double-counting the time step
            base_mu_annual = self.portfolio_means_annual[portfolio_action]
            base_sigma_annual = self.portfolio_stds_annual[portfolio_action]

            if self.use_sentiment and self.use_vix_market_adjustments:
                # Sentiment RL with VIX market adjustments: Use VIX-adjusted parameters
                # μ_adj = μ + β
                mu_adj = base_mu_annual + beta

                # σ adjustment depends on volatility_method:
                # - rolling_vol: δ adjusts σ directly → σ_adj = σ - δ
                # - return_squared: δ adjusts variance → σ_adj = √(σ² - δ)
                if self.volatility_method == 'rolling_vol':
                    sigma_adj = base_sigma_annual - delta
                else:  # return_squared
                    # δ adjusts variance, so: σ²_adj = σ² - δ
                    # Ensure non-negative variance before sqrt
                    variance_adj = max(base_sigma_annual**2 - delta, 0.0004)  # min σ = 0.02
                    sigma_adj = np.sqrt(variance_adj)

                # Bounds to prevent extreme values (annual)
                mu_adj = np.clip(mu_adj, -0.15, 0.30)
                sigma_adj = np.clip(sigma_adj, 0.02, 0.50)
            else:
                # No VIX market adjustments: Use base parameters
                # (This covers both Pure RL and Sentiment RL without market adjustments)
                mu_adj = base_mu_annual
                sigma_adj = base_sigma_annual

            # Generate return using GBM formula with dt = 1/12
            # R = (μ - ½σ²)dt + σ√dt × Z
            # EPISODE-SCOPED DETERMINISTIC SHOCK: Use episode RNG for consistent shock sequences
            # This ensures each episode gets the same shock sequence across training iterations
            z_shock = self.episode_rng.normal(0, 1)
            drift = (mu_adj - 0.5 * sigma_adj**2) * self.dt
            diffusion = sigma_adj * np.sqrt(self.dt) * z_shock

            log_return = drift + diffusion
            new_wealth = wealth_after_goal * np.exp(log_return)
            self.current_wealth = max(0.0, new_wealth)
        else:
            self.current_wealth = wealth_after_goal
            mu_adj = sigma_adj = z_shock = log_return = 0

        # ═══════════════════════════════════════════════════════
        # STEP 4: Advance VIX for next month using SAME shock
        # ═══════════════════════════════════════════════════════
        # CRITICAL FIX: Pass the same z_shock to VIX model for proper correlation
        new_vix = self.vix_model.step_vix(dt=self.dt, market_shock=z_shock)

        # ═══════════════════════════════════════════════════════
        # STEP 5: Advance time
        # ═══════════════════════════════════════════════════════
        self.current_month += 1
        terminated = self.current_month >= self.total_months
        truncated = False

        observation = self._get_observation()

        # Comprehensive info dict with β and δ
        info = {
            'month': self.current_month,
            'year': self.current_month / self.months_per_year,
            'wealth': self.current_wealth,
            'vix': self.vix_model.current_vix,
            'vix_avg': vix_avg,
            'beta': beta,
            'delta': delta,
            'mu_adj': mu_adj,
            'sigma_adj': sigma_adj,
            'z_shock': z_shock if self.current_month < self.total_months else 0,
            'log_return': log_return if self.current_month < self.total_months else 0,
            'goal_taken': goal_action == 1 and current_year in self.goals_taken,
            'goals_taken_so_far': len(self.goals_taken),
            'total_utility': self.total_utility,
            'portfolio_choice': portfolio_action
        }

        return observation, reward, terminated, truncated, info

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics including β and δ history"""
        return {
            'total_utility': self.total_utility,
            'final_wealth': self.current_wealth,
            'goals_achieved': len(self.goals_taken),
            'goals_taken': self.goals_taken,
            'mean_beta': np.mean(self.beta_history) if self.beta_history else 0,
            'mean_delta': np.mean(self.delta_history) if self.delta_history else 0,
            'std_beta': np.std(self.beta_history) if self.beta_history else 0,
            'std_delta': np.std(self.delta_history) if self.delta_history else 0
        }

    def close(self):
        """Clean up"""
        pass


def make_gbwm_env_monthly(
    num_goals: int = 4,
    use_sentiment: bool = True,
    use_vix_market_adjustments: bool = False,
    vix_params: Optional[VIXModelParams] = None,
    portfolio_betas: Optional[np.ndarray] = None,
    portfolio_deltas: Optional[np.ndarray] = None,
    volatility_method: str = 'rolling_vol',
    vix_model_type: str = 'mrjd',
    use_real_ef: bool = False,
    portfolio_means: np.ndarray = None,
    portfolio_stds: np.ndarray = None
) -> GBWMEnvMonthly:
    """
    Factory function to create monthly GBWM environment

    Args:
        num_goals: Number of financial goals
        use_sentiment: If True, agent sees VIX features in state space
        use_vix_market_adjustments: If True, apply VIX β/δ adjustments to market returns/volatility
        vix_params: Optional VIX model parameters
        portfolio_betas: Optional (num_portfolios,) array of β sensitivities
        portfolio_deltas: Optional (num_portfolios,) array of δ sensitivities
        volatility_method: How δ is applied to σ:
                          'rolling_vol': σ_adj = σ - δ (δ adjusts volatility)
                          'return_squared': σ_adj = √(σ² - δ) (δ adjusts variance)
        vix_model_type: VIX model type - 'mrjd' (Mean-Reverting Jump-Diffusion) 
                       or 'regime_switching' (Markov Regime-Switching) (default: 'mrjd')
        use_real_ef: If True, use real historical data for efficient frontier.
                     If False (default), use paper's hardcoded values.
                     Only used if portfolio_means/portfolio_stds are not provided.
        portfolio_means: Custom portfolio mean returns (15,). Overrides use_real_ef.
        portfolio_stds: Custom portfolio volatilities (15,). Overrides use_real_ef.

    Returns:
        GBWMEnvMonthly instance
    """
    from copy import deepcopy

    # Create a deep copy of config to avoid modifying shared state
    config = deepcopy(DEFAULT_ENV_CONFIG)

    # Set portfolio parameters from efficient frontier (single source of truth)
    # Priority: explicit parameters > use_real_ef > default config
    if portfolio_means is not None and portfolio_stds is not None:
        config.portfolio_config.mean_returns = np.array(portfolio_means)
        config.portfolio_config.return_stds = np.array(portfolio_stds)
    else:
        # Load from efficient frontier
        from src.data.efficient_frontier import get_portfolio_arrays
        ef_means, ef_stds = get_portfolio_arrays(use_real_ef=use_real_ef, num_portfolios=15)
        config.portfolio_config.mean_returns = ef_means
        config.portfolio_config.return_stds = ef_stds

    return GBWMEnvMonthly(
        num_goals=num_goals,
        use_sentiment=use_sentiment,
        use_vix_market_adjustments=use_vix_market_adjustments,
        vix_params=vix_params,
        portfolio_betas=portfolio_betas,
        portfolio_deltas=portfolio_deltas,
        volatility_method=volatility_method,
        vix_model_type=vix_model_type,
        config=config
    )


# Test the environment
if __name__ == "__main__":
    print("=" * 70)
    print("Testing GBWMEnvMonthly (192 monthly steps)")
    print("=" * 70)

    # Test Sentiment RL environment
    print("\n--- Sentiment RL Environment ---")
    env = make_gbwm_env_monthly(num_goals=4, use_sentiment=True)
    obs, info = env.reset()

    print(f"Total months: {env.total_months}")
    print(f"Goal months: {env.goal_months}")
    print(f"Initial observation: {obs}")
    print(f"Initial β: {info['beta']:.4f}, δ: {info['delta']:.4f}")

    total_reward = 0
    for step in range(env.total_months):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        # Print every 12 months (yearly)
        if (step + 1) % 12 == 0:
            print(f"Year {(step+1)//12}: VIX={info['vix']:.1f}, "
                  f"β={info['beta']:+.4f}, δ={info['delta']:+.4f}, "
                  f"wealth={info['wealth']:.0f}")

        if done:
            break

    stats = env.get_episode_stats()
    print(f"\nEpisode stats:")
    print(f"  Total utility: {stats['total_utility']:.1f}")
    print(f"  Goals achieved: {stats['goals_achieved']}/{env.num_goals}")
    print(f"  Mean β: {stats['mean_beta']:.4f}, Mean δ: {stats['mean_delta']:.4f}")

    env.close()

    # Test Pure RL environment
    print("\n--- Pure RL Environment ---")
    env_pure = make_gbwm_env_monthly(num_goals=4, use_sentiment=False)
    obs, info = env_pure.reset()
    print(f"Observation space: {env_pure.observation_space}")
    print(f"Initial observation: {obs}")
    env_pure.close()

    print("\n" + "=" * 70)
    print("Monthly environment test complete!")
