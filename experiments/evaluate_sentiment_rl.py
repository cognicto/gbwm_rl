#!/usr/bin/env python
"""
Sentiment RL Evaluation Script with Monte Carlo Simulations

This script:
1. Pre-trains β/δ parameters and loads efficient frontier
2. Trains Sentiment RL agents for all goal counts (1, 2, 4, 8, 16)
3. Solves Multi-Goal DP (optimal) for all goal counts
4. Runs Monte Carlo simulations (default 1,000,000) with shared seeds
5. Computes efficiency = Sentiment RL reward / DP reward
6. Generates Figure 1: Efficiency vs Number of Goals

Key differences from Pure RL:
- 192 monthly time steps (16 years × 12 months)
- 5D state: [time, wealth, vix_level, vix_avg, vix_momentum]
- VIX-adjusted μ/σ using pre-trained β and δ
- Uses SentimentAwarePPOAgent with feature encoders

Usage:
    # Quick test (1000 simulations, 1 iteration)
    python experiments/evaluate_sentiment_rl.py --num_simulations 1000 --num_iterations 1

    # Full evaluation (1M simulations, 10 iterations)
    python experiments/evaluate_sentiment_rl.py --num_simulations 1000000 --num_iterations 10
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment.gbwm_env_monthly import GBWMEnvMonthly, make_gbwm_env_monthly
from src.models.sentiment_ppo_agent import SentimentAwarePPOAgent
from src.models.vix_market_model import VIXMarketModel, VIXModelParams, create_vix_model, MRJDParams
from src.algorithms.dynamic_programming import MultiGoalGBWMDP, MultiGoalDPConfig
from src.data.efficient_frontier import compute_efficient_frontier, get_portfolio_arrays
from src.data.beta_delta_learner import learn_beta_delta
from config.training_config import TrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_initial_wealth(num_goals: int) -> float:
    """Calculate initial wealth using paper formula: W0 = 12 * NG^0.85 * 10000"""
    return 12 * (num_goals ** 0.85) * 10000


def get_goal_years(num_goals: int) -> List[int]:
    """Get goal schedule based on number of goals"""
    if num_goals == 1:
        return [16]
    elif num_goals == 2:
        return [8, 16]
    elif num_goals == 4:
        return [4, 8, 12, 16]
    elif num_goals == 8:
        return [2, 4, 6, 8, 10, 12, 14, 16]
    elif num_goals == 16:
        return list(range(1, 17))
    else:
        raise ValueError(f"Unsupported number of goals: {num_goals}")


def get_goal_months(num_goals: int, months_per_year: int = 12) -> List[int]:
    """Convert goal years to goal months"""
    return [y * months_per_year for y in get_goal_years(num_goals)]


def get_goal_cost(year: int) -> float:
    """Calculate goal cost at given year: C(t) = 100000 * 1.08^t"""
    return 100000 * (1.08 ** year)


def get_goal_utility(year: int) -> float:
    """Calculate goal utility at given year: U(t) = 10 + t"""
    return 10 + year


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SentimentEvaluationConfig:
    """Configuration for Sentiment RL evaluation"""
    goal_counts: List[int] = None
    num_simulations: int = 1000000
    num_iterations: int = 10
    batch_size: int = 4800
    learning_rate: float = 0.01
    hidden_dim: int = 128  # Larger for sentiment features

    # Time configuration (monthly)
    years_horizon: int = 16
    months_per_year: int = 12

    @property
    def time_horizon(self) -> int:
        return self.years_horizon * self.months_per_year  # 192 months

    # Baseline comparison mode
    # - 'annual_stable': Pure RL (annual) vs DP in stable market, Sentiment RL vs DP in VIX market
    # - 'monthly_vix': All methods at monthly granularity, DP uses stable market
    baseline_mode: str = "annual_stable"

    # Network architecture (simplified for annual_stable mode)
    policy_type: str = "standard"    # Simplified for annual mode
    value_type: str = "standard"     # Simplified for annual mode  
    encoder_type: str = "feature"    # Simplified for annual mode

    # Pure RL network architecture (only used for non-annual_stable modes)
    pure_rl_policy_type: str = None  # If None, uses policy_type
    pure_rl_value_type: str = None   # If None, uses value_type
    pure_rl_encoder_type: str = None # If None, uses encoder_type

    # VIX parameters (only used for monthly_vix mode)
    vix_kappa: float = 3.0
    vix_theta: float = 20.0
    vix_sigma_v: float = 1.5
    vix_lambda_jump: float = 0.025
    vix_mu_jump: float = 12.0
    vix_sigma_jump: float = 8.0

    # Pre-training (simplified)
    use_cache: bool = True
    force_recompute: bool = False

    # Efficient frontier (simplified - use paper values for consistency)
    use_real_ef: bool = False  

    # VIX adjustment settings (only used for non-annual_stable modes)
    use_delta_adjustment: bool = False  # Disabled for annual_stable mode
    volatility_method: str = 'rolling_vol'  
    dp_vix_market: bool = False  # DP uses stable market
    vix_model_type: str = "mrjd"

    device: str = "auto"  # Auto-detect GPU
    random_seed: int = 42
    output_dir: str = None
    skip_training: bool = False  # Resume from simulation only

    def __post_init__(self):
        if self.goal_counts is None:
            self.goal_counts = [1, 2, 4, 8, 16]
        if self.output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = f"data/comparisons/sentiment_rl_{timestamp}"
        # Default Pure RL architecture to same as Sentiment RL for fair comparison
        if self.pure_rl_policy_type is None:
            self.pure_rl_policy_type = self.policy_type
        if self.pure_rl_value_type is None:
            self.pure_rl_value_type = self.value_type
        if self.pure_rl_encoder_type is None:
            self.pure_rl_encoder_type = self.encoder_type


@dataclass
class SimulationResult:
    """Container for simulation results"""
    method_name: str
    num_goals: int
    mean_reward: float
    std_reward: float
    efficiency: float
    num_simulations: int
    goal_success_rate: float
    mean_final_wealth: float
    std_final_wealth: float = 0.0
    mean_vix_sentiment: float = 0.0


# =============================================================================
# SENTIMENT RL EVALUATOR
# =============================================================================

class SentimentRLEvaluator:
    """
    Evaluator for Sentiment RL vs DP comparison.

    Workflow:
    1. Load pre-trained β/δ and efficient frontier
    2. Train Sentiment RL agents for each goal count
    3. Solve Multi-Goal DP for each goal count
    4. Generate shared random seeds and market shocks
    5. Simulate both methods with identical market conditions + VIX
    6. Compute efficiency and generate Figure 1

    Key Features:
    - 192 monthly time steps
    - 5D state: [time, wealth, vix_level, vix_avg, vix_momentum]
    - VIX-adjusted μ/σ via pre-trained β and δ
    - Shared random seeds for fair comparison
    """

    def __init__(self, config: SentimentEvaluationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-trained parameters
        self.beta: float = None
        self.delta: float = None
        self.efficient_frontier: Dict = None
        self.vix_params: VIXModelParams = None

        # Portfolio parameters
        self.portfolio_means_annual: np.ndarray = None
        self.portfolio_stds_annual: np.ndarray = None

        # Storage for agents and policies - organized by baseline mode
        self.dp_policies: Dict[int, MultiGoalGBWMDP] = {}
        self.dp_policies_annual: Dict[int, MultiGoalGBWMDP] = {}
        self.dp_policies_monthly: Dict[int, MultiGoalGBWMDP] = {}
        
        # Agent containers (only initialized when needed by specific modes)
        self.sentiment_agents: Dict[int, SentimentAwarePPOAgent] = {}  # For monthly_vix mode
        self.pure_rl_agents: Dict[int, SentimentAwarePPOAgent] = {}  # For monthly_vix mode
        self.pure_rl_agents_annual: Dict[int, SentimentAwarePPOAgent] = {}  # For annual_stable mode

        # Results storage - mode-aware initialization
        if config.baseline_mode == 'monthly_vix':
            self.results: Dict[str, Dict[int, SimulationResult]] = {
                "DP (Annual)": {},           # DP Annual
                "Pure RL (Annual)": {},      # Pure RL Annual  
                "DP (Monthly)": {},          # DP Monthly
                "Pure RL (Monthly)": {},     # Pure RL Monthly
                "Sentiment RL (Monthly)": {} # Sentiment RL Monthly
            }
        elif config.baseline_mode == 'annual_stable':
            self.results: Dict[str, Dict[int, SimulationResult]] = {
                "DP": {},
                "DP (Annual Stable)": {},
                "Pure RL (Annual)": {}
            }
        else:
            raise ValueError(f"Unsupported baseline_mode: {config.baseline_mode}. Only 'monthly_vix' and 'annual_stable' are supported.")

        # Shared random state for fair comparison
        self.shared_seeds: np.ndarray = None
        self.shared_rng: np.random.Generator = None
        self.market_shocks: np.ndarray = None  # Shape: (num_simulations, time_horizon)

        # Checkpoint system for long evaluations
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.enable_checkpoints = config.num_simulations >= 100000  # Auto-enable for large sims
        
        logger.info(f"SentimentRLEvaluator initialized")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Goal counts: {config.goal_counts}")
        logger.info(f"  Simulations: {config.num_simulations:,}")
        logger.info(f"  Time horizon: {config.time_horizon} monthly steps")
        if self.enable_checkpoints:
            logger.info(f"  Checkpoints enabled: {self.checkpoint_dir}")

    def load_pretrained_parameters(self):
        """Load pre-trained β/δ and efficient frontier from cache"""
        logger.info("=" * 60)
        logger.info("LOADING PRE-TRAINED PARAMETERS")
        logger.info("=" * 60)

        # Load β and δ
        logger.info("\n[Stage A] Loading β/δ parameters...")
        logger.info(f"  Volatility method: {self.config.volatility_method}")
        logger.info(f"  Use δ adjustment: {self.config.use_delta_adjustment}")
        try:
            beta_delta_result = learn_beta_delta(
                use_cache=self.config.use_cache,
                force_recompute=self.config.force_recompute,
                volatility_method=self.config.volatility_method
            )
            # FIXED: learn_beta_delta returns portfolio_betas (15-element array), not 'beta'
            # Use portfolio-specific values for more accurate simulation
            self.portfolio_betas = beta_delta_result['portfolio_betas']
            self.portfolio_deltas = beta_delta_result['portfolio_deltas']

            # Use mean as single sensitivity for VIXModelParams
            # (weighted toward mid-risk portfolios which are most common)
            self.beta = float(np.mean(self.portfolio_betas))
            self.delta = float(np.mean(self.portfolio_deltas))

            logger.info(f"  Portfolio β range: [{self.portfolio_betas.min():.4f}, {self.portfolio_betas.max():.4f}]")
            logger.info(f"  Portfolio δ range: [{self.portfolio_deltas.min():.4f}, {self.portfolio_deltas.max():.4f}]")
            logger.info(f"  Mean β (drift adjustment): {self.beta:.6f}")
            logger.info(f"  Mean δ (volatility adjustment): {self.delta:.6f}")
        except Exception as e:
            logger.warning(f"Failed to load β/δ: {e}. Using defaults.")
            self.beta = 0.03
            self.delta = 0.05
            # Create default portfolio-specific arrays
            self.portfolio_betas = np.full(15, self.beta)
            self.portfolio_deltas = np.full(15, self.delta)

        # Handle --no_delta_adjustment flag or annual_stable mode: zero out all δ values
        if not self.config.use_delta_adjustment or self.config.baseline_mode == "annual_stable":
            if self.config.baseline_mode == "annual_stable":
                logger.info("  δ adjustment DISABLED (annual_stable mode)")
            else:
                logger.info("  δ adjustment DISABLED (--no_delta_adjustment)")
            logger.info("  Setting all portfolio_deltas = 0 (only μ adjustment will be used)")
            self.portfolio_deltas = np.zeros(15)
            self.delta = 0.0

        # Load efficient frontier using unified get_portfolio_arrays function
        # This ensures consistency with Pure RL and DP
        logger.info(f"\n[Stage B] Loading efficient frontier (use_real_ef={self.config.use_real_ef})...")
        try:
            # Use unified function - same source as Pure RL and DP
            self.portfolio_means_annual, self.portfolio_stds_annual = get_portfolio_arrays(
                use_real_ef=self.config.use_real_ef,
                num_portfolios=15
            )
            logger.info(f"  Portfolios: {len(self.portfolio_means_annual)}")
            logger.info(f"  Return range: {self.portfolio_means_annual[0]:.2%} to {self.portfolio_means_annual[-1]:.2%}")
            logger.info(f"  Vol range: {self.portfolio_stds_annual[0]:.2%} to {self.portfolio_stds_annual[-1]:.2%}")
        except Exception as e:
            logger.warning(f"Failed to load EF: {e}. Using fallback defaults.")
            self.portfolio_means_annual = np.linspace(0.0526, 0.0886, 15)
            self.portfolio_stds_annual = np.linspace(0.0374, 0.1954, 15)

        # Configure VIX model
        self.vix_params = VIXModelParams(
            kappa=self.config.vix_kappa,
            theta=self.config.vix_theta,
            sigma_v=self.config.vix_sigma_v,
            lambda_jump=self.config.vix_lambda_jump,
            mu_jump=self.config.vix_mu_jump,
            sigma_jump=self.config.vix_sigma_jump,
            beta_sensitivity=self.beta,
            delta_sensitivity=self.delta
        )
        logger.info(f"\n[VIX Model] β_sens={self.beta:.4f}, δ_sens={self.delta:.4f}")

    def _generate_shared_random_state(self):
        """Generate shared random seeds and market shocks for fair comparison"""
        logger.info(f"Generating shared random state for {self.config.num_simulations:,} simulations...")

        np.random.seed(self.config.random_seed)

        # Random seeds for reproducibility
        self.shared_seeds = np.random.randint(0, 2**31 - 1, size=self.config.num_simulations)
        self.shared_rng = np.random.default_rng(self.config.random_seed)

        # Pre-generate market shocks (standard normal) for all simulations
        # Shape: (num_simulations, time_horizon) where time_horizon = 192 monthly steps
        self.market_shocks = np.random.normal(0, 1, size=(self.config.num_simulations, self.config.time_horizon))

        logger.info(f"  Generated {self.config.num_simulations:,} seeds and {self.config.time_horizon}-step shock sequences")

    def train_sentiment_agents(self, architecture_type=None):
        """Train Sentiment-Aware RL agents for all goal counts
        
        Args:
            architecture_type: 'simple', 'advanced', or None (for legacy compatibility)
        """
        # Skip training if not needed for current baseline mode
        if self.config.baseline_mode == "annual_stable":
            logger.info("Skipping Sentiment RL training (annual_stable mode uses only DP + Pure RL annual)")
            return
            
        # Skip training if --skip_training flag is set and models exist
        if self.config.skip_training:
            if self._load_existing_sentiment_models(architecture_type):
                logger.info("Skipping Sentiment RL training (--skip_training flag and models found)")
                return
            else:
                logger.warning("--skip_training set but no existing models found. Proceeding with training.")
                logger.warning("Make sure models exist in the correct output directory.")
            
        logger.info("=" * 60)
        logger.info("TRAINING SENTIMENT-AWARE RL AGENTS")
        if architecture_type:
            logger.info(f"Architecture Type: {architecture_type.upper()}")
        logger.info("=" * 60)

        for num_goals in self.config.goal_counts:
            initial_wealth = get_initial_wealth(num_goals)
            logger.info(f"\nTraining Sentiment RL for {num_goals} goals (W0=${initial_wealth:,.0f})...")

            train_portfolio_betas = None  # Disable VIX market adjustments
            train_portfolio_deltas = None  # Disable VIX market adjustments
            logger.info("  Training with sentiment features but NO VIX market adjustments")

            # Create monthly environment with VIX simulation for sentiment features only
            # No β/δ adjustments to returns/volatility - same market as Pure RL
            env = make_gbwm_env_monthly(
                num_goals=num_goals,
                use_sentiment=True,
                use_vix_market_adjustments=False,  # NEW: Separate sentiment from market adjustments
                vix_params=self.vix_params,
                portfolio_betas=train_portfolio_betas,
                portfolio_deltas=train_portfolio_deltas,
                volatility_method=self.config.volatility_method,  # Pass volatility method
                vix_model_type=self.config.vix_model_type,  # FIXED: Pass VIX model type for consistency
                use_real_ef=self.config.use_real_ef,  # Same EF as DP/Pure RL
                portfolio_means=self.portfolio_means_annual,  # Explicit params for consistency
                portfolio_stds=self.portfolio_stds_annual
            )

            # Training config with adaptive parameters for stability
            # Scale learning rate and entropy based on problem complexity
            lr_scale = max(0.2, 1.0 / np.sqrt(num_goals))  # Reduce LR for more goals
            entropy_scale = min(4.0, 1.0 + 0.5 * num_goals)  # Increase exploration for more goals
            
            adaptive_lr = self.config.learning_rate * lr_scale
            adaptive_entropy = 0.01 * entropy_scale
            adaptive_clip = min(0.3, 0.5 / np.sqrt(num_goals))  # Conservative clipping for complex problems
            
            training_config = TrainingConfig(
                batch_size=self.config.batch_size,
                learning_rate=adaptive_lr,
                n_neurons=self.config.hidden_dim,
                ppo_epochs=4,
                mini_batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=adaptive_clip,
                entropy_coeff=adaptive_entropy,
                max_grad_norm=0.5,
                time_horizon=self.config.time_horizon,
                device=self.config.device
            )
            
            logger.info(f"  Adaptive params: LR={adaptive_lr:.4f}, entropy={adaptive_entropy:.3f}, clip={adaptive_clip:.2f}")

            # Create Sentiment-Aware PPO agent
            agent = SentimentAwarePPOAgent(
                env=env,
                config=training_config,
                policy_type=self.config.policy_type,
                value_type=self.config.value_type,
                encoder_type=self.config.encoder_type,
                device=self.config.device,
                sentiment_enabled=True
            )

            # Train
            total_timesteps = self.config.num_iterations * self.config.batch_size * self.config.time_horizon

            start_time = time.time()
            history = []
            for iteration in range(self.config.num_iterations):
                metrics = agent.train_iteration()
                history.append(metrics)
            train_time = time.time() - start_time

            # Store agent in appropriate container based on architecture type
            if architecture_type == 'simple':
                self.sentiment_agents_simple[num_goals] = agent
            elif architecture_type == 'advanced':
                self.sentiment_agents_advanced[num_goals] = agent
            else:
                # Legacy behavior - store in main container
                self.sentiment_agents[num_goals] = agent

            # Save model with different paths based on architecture type
            if architecture_type == 'simple':
                model_dir = self.output_dir / "models" / "sentiment_rl_simple" / f"goals_{num_goals}"
            elif architecture_type == 'advanced':
                model_dir = self.output_dir / "models" / "sentiment_rl_advanced" / f"goals_{num_goals}"
            else:
                # Legacy behavior - use original path
                model_dir = self.output_dir / "models" / "sentiment_rl" / f"goals_{num_goals}"
                
            model_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(model_dir / "model.pth"))

            mean_reward = np.mean([h.get('mean_episode_reward', 0) for h in history[-3:]]) if history else 0
            logger.info(f"  Trained in {train_time:.1f}s, mean reward: {mean_reward:.2f}")

            env.close()

    def train_pure_rl_agents(self):
        """
        Train Pure RL agents at MONTHLY granularity for fair comparison.

        Key differences from Sentiment RL:
        - State: 2D [time, wealth] (no VIX features)
        - Environment: No VIX adjustments to μ/σ
        - Same monthly time steps (192) for fair comparison

        This isolates the VIX effect: both Pure RL and Sentiment RL
        operate at monthly granularity, but only Sentiment RL uses VIX.
        """
        # Skip training if not needed for current baseline mode
        if self.config.baseline_mode == "annual_stable":
            logger.info("Skipping Pure RL monthly training (annual_stable mode uses annual Pure RL)")
            return
            
        # Skip training if --skip_training flag is set and models exist
        if self.config.skip_training:
            if self._load_existing_pure_rl_models():
                logger.info("Skipping Pure RL training (--skip_training flag and models found)")
                return
            else:
                logger.warning("--skip_training set but no existing Pure RL models found. Proceeding with training.")
            
        logger.info("=" * 60)
        logger.info("TRAINING PURE RL AGENTS (MONTHLY, NO VIX)")
        logger.info("=" * 60)
        logger.info("  This provides fair comparison with Sentiment RL")
        logger.info("  Same time granularity, different: VIX in state & μ/σ adjustments")

        for num_goals in self.config.goal_counts:
            initial_wealth = get_initial_wealth(num_goals)
            logger.info(f"\nTraining Pure RL for {num_goals} goals (W0=${initial_wealth:,.0f})...")

            # Create monthly environment WITHOUT VIX (use_sentiment=False)
            # Use same portfolio parameters as DP and Sentiment RL for fair comparison
            env = make_gbwm_env_monthly(
                num_goals=num_goals,
                use_sentiment=False,  # NO VIX - key difference!
                vix_model_type=self.config.vix_model_type,  # For VIX market modes
                vix_params=None,
                portfolio_betas=None,
                portfolio_deltas=None,
                use_real_ef=self.config.use_real_ef,  # Same EF as DP/Sentiment RL
                portfolio_means=self.portfolio_means_annual,  # Explicit params for consistency
                portfolio_stds=self.portfolio_stds_annual
            )

            # Training config (same as Sentiment RL)
            training_config = TrainingConfig(
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                n_neurons=self.config.hidden_dim,
                ppo_epochs=4,
                mini_batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=0.5,
                entropy_coeff=0.01,
                max_grad_norm=0.5,
                time_horizon=self.config.time_horizon,
                device=self.config.device
            )

            # Create Pure RL agent (no sentiment features)
            # Pure RL has 2D state, so must use 'simple' encoder (feature/attention require VIX)
            # Policy and value types can still match Sentiment RL for fair comparison
            pure_rl_encoder = self.config.pure_rl_encoder_type
            if pure_rl_encoder in ['feature', 'attention', 'adaptive']:
                logger.info(f"  Note: '{pure_rl_encoder}' encoder requires VIX features. Using 'simple' for Pure RL.")
                pure_rl_encoder = 'simple'

            agent = SentimentAwarePPOAgent(
                env=env,
                config=training_config,
                policy_type=self.config.pure_rl_policy_type,
                value_type=self.config.pure_rl_value_type,
                encoder_type=pure_rl_encoder,
                device=self.config.device,
                sentiment_enabled=False   # NO VIX features in state
            )

            # Train
            start_time = time.time()
            history = []
            for iteration in range(self.config.num_iterations):
                metrics = agent.train_iteration()
                history.append(metrics)
            train_time = time.time() - start_time

            self.pure_rl_agents[num_goals] = agent

            # Save model
            model_dir = self.output_dir / "models" / "pure_rl_monthly" / f"goals_{num_goals}"
            model_dir.mkdir(parents=True, exist_ok=True)

            agent.save(str(model_dir / "model.pth"))

            mean_reward = np.mean([h.get('mean_episode_reward', 0) for h in history[-3:]]) if history else 0
            logger.info(f"  Trained in {train_time:.1f}s, mean reward: {mean_reward:.2f}")

            env.close()

    def train_pure_rl_vix_market_agents(self):
        """
        Train Pure RL agents in VIX-ADJUSTED market - REMOVED for monthly_vix cleanup.
        Not needed for simplified monthly_vix mode requirements.
        """
        logger.info("Skipping Pure RL VIX Market training (removed in cleanup - not needed for monthly_vix mode)")
        return

    def train_sentiment_rl_stable_agents(self):
        """
        Train Sentiment RL agents in STABLE market - REMOVED for monthly_vix cleanup.
        Not needed for simplified monthly_vix mode requirements.
        """
        logger.info("Skipping Sentiment RL Stable training (removed in cleanup - not needed for monthly_vix mode)")
        return
        logger.info("=" * 60)
        logger.info("TRAINING SENTIMENT RL AGENTS (MONTHLY, STABLE MARKET)")
        logger.info("=" * 60)
        logger.info("  State: VIX features included")
        logger.info("  Market: Stable (no VIX adjustments to μ/σ)")
        
        # Initialize storage for stable market Sentiment RL agents
        self.sentiment_rl_stable_agents = {}
        
        for num_goals in self.config.goal_counts:
            initial_wealth = get_initial_wealth(num_goals)
            
            logger.info(f"\nTraining Sentiment RL (Stable) for {num_goals} goals (W0=${initial_wealth:,.0f})...")
            
            # Create environment without VIX market effects (stable market)
            env = make_gbwm_env_monthly(
                num_goals=num_goals,
                use_sentiment=True,  # VIX in state - key difference!
                vix_params=self.vix_params,
                portfolio_betas=None,  # NO VIX market adjustments
                portfolio_deltas=None,  # NO VIX market adjustments
                volatility_method=self.config.volatility_method,
                vix_model_type=self.config.vix_model_type,
                use_real_ef=self.config.use_real_ef,
                portfolio_means=self.portfolio_means_annual,
                portfolio_stds=self.portfolio_stds_annual
            )
            
            # Training config with adaptive params
            lr_scale = max(0.2, 1.0 / np.sqrt(num_goals))
            entropy_scale = min(4.0, 1.0 + 0.5 * num_goals)
            
            adaptive_lr = self.config.learning_rate * lr_scale
            adaptive_entropy = 0.01 * entropy_scale
            adaptive_clip = min(0.3, 0.5 / np.sqrt(num_goals))
            
            training_config = TrainingConfig(
                batch_size=self.config.batch_size,
                learning_rate=adaptive_lr,
                n_neurons=self.config.hidden_dim,
                ppo_epochs=4,
                mini_batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=adaptive_clip,
                entropy_coeff=adaptive_entropy,
                max_grad_norm=0.5,
                time_horizon=self.config.time_horizon,
                device=self.config.device
            )
            
            logger.info(f"  Adaptive params: LR={adaptive_lr:.4f}, "
                       f"entropy={adaptive_entropy:.3f}, "
                       f"clip={adaptive_clip:.2f}")
            
            # Create agent
            agent = SentimentAwarePPOAgent(
                env=env,
                config=training_config,
                policy_type=self.config.policy_type,
                value_type=self.config.value_type,
                encoder_type=self.config.encoder_type,
                device=self.config.device,
                sentiment_enabled=True
            )
            
            # Train
            start_time = time.time()
            history = []
            for iteration in range(self.config.num_iterations):
                metrics = agent.train_iteration()
                history.append(metrics)
            train_time = time.time() - start_time
            
            # Store agent
            self.sentiment_rl_stable_agents[num_goals] = agent
            
            # Save model
            model_dir = self.output_dir / "models" / "sentiment_rl_stable" / f"goals_{num_goals}"
            model_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(model_dir / "model.pth"))
            
            mean_reward = np.mean([h.get('mean_episode_reward', 0) for h in history[-3:]]) if history else 0
            logger.info(f"  Trained in {train_time:.1f}s, mean reward: {mean_reward:.2f}")
            
            env.close()

    def solve_dp(self):
        """
        Solve Multi-Goal DP for all goal counts.

        IMPORTANT: DP uses the same portfolio parameters as Sentiment RL
        via custom_mu_array and custom_sigma_array to ensure fair comparison.
        """
        logger.info("=" * 60)
        logger.info("SOLVING MULTI-GOAL DYNAMIC PROGRAMMING")
        logger.info("=" * 60)

        for num_goals in self.config.goal_counts:
            initial_wealth = get_initial_wealth(num_goals)
            goal_years = get_goal_years(num_goals)

            logger.info(f"\nSolving DP for {num_goals} goals (W0=${initial_wealth:,.0f})...")
            logger.info(f"  Goal years: {goal_years}")

            # Create DP config with SHARED portfolio parameters
            # This ensures DP and Sentiment RL use identical portfolios
            dp_config = MultiGoalDPConfig(
                initial_wealth=initial_wealth,
                time_horizon=self.config.years_horizon,  # 16 years
                goal_years=goal_years,
                num_portfolios=len(self.portfolio_means_annual),
                grid_density=3.0,  # Higher resolution for better accuracy
                random_seed=self.config.random_seed,
                # CRITICAL: Pass same portfolio arrays as Sentiment RL
                custom_mu_array=self.portfolio_means_annual,
                custom_sigma_array=self.portfolio_stds_annual
            )

            # Solve DP
            start_time = time.time()
            dp = MultiGoalGBWMDP(dp_config)
            dp.solve()
            solve_time = time.time() - start_time

            self.dp_policies[num_goals] = dp

            expected_utility = dp.get_expected_utility()
            logger.info(f"  Solved in {solve_time:.2f}s, expected utility: {expected_utility:.2f}")

    def solve_dp_annual(self):
        """
        Solve Multi-Goal DP with ANNUAL time steps (16 steps).
        
        Optimizes for decisions made once per year with annual market dynamics.
        Used for annual DP simulations and comparisons with annual Pure RL.
        """
        logger.info("=" * 60)
        logger.info("SOLVING MULTI-GOAL DP (ANNUAL GRANULARITY)")
        logger.info("=" * 60)

        for num_goals in self.config.goal_counts:
            initial_wealth = get_initial_wealth(num_goals)
            goal_years = get_goal_years(num_goals)

            logger.info(f"\nSolving Annual DP for {num_goals} goals (W0=${initial_wealth:,.0f})...")
            logger.info(f"  Goal years: {goal_years}")
            logger.info(f"  Time horizon: {self.config.years_horizon} annual steps")

            # Create DP config for ANNUAL decisions
            dp_config = MultiGoalDPConfig(
                initial_wealth=initial_wealth,
                time_horizon=self.config.years_horizon,  # 16 annual steps
                goal_years=goal_years,
                num_portfolios=len(self.portfolio_means_annual),
                grid_density=3.0,  # Higher resolution for better accuracy
                random_seed=self.config.random_seed,
                # Use annual portfolio parameters
                custom_mu_array=self.portfolio_means_annual,
                custom_sigma_array=self.portfolio_stds_annual
            )

            # Solve DP
            start_time = time.time()
            dp = MultiGoalGBWMDP(dp_config)
            dp.solve()
            solve_time = time.time() - start_time

            self.dp_policies_annual[num_goals] = dp

            expected_utility = dp.get_expected_utility()
            logger.info(f"  Annual DP solved in {solve_time:.2f}s, expected utility: {expected_utility:.2f}")

    def solve_dp_monthly(self):
        """
        Solve Multi-Goal DP with MONTHLY time steps (192 steps).
        
        Optimizes for decisions made every month with monthly market dynamics.
        Used for monthly DP simulations and fair comparisons with monthly RL methods.
        """
        logger.info("=" * 60)
        logger.info("SOLVING MULTI-GOAL DP (MONTHLY GRANULARITY)")
        logger.info("=" * 60)

        for num_goals in self.config.goal_counts:
            initial_wealth = get_initial_wealth(num_goals)
            goal_years = get_goal_years(num_goals)

            logger.info(f"\nSolving Monthly DP for {num_goals} goals (W0=${initial_wealth:,.0f})...")
            logger.info(f"  Goal years: {goal_years}")
            logger.info(f"  Time horizon: {self.config.time_horizon} monthly steps")

            # Create DP config for MONTHLY decisions
            # Convert annual parameters to monthly
            monthly_mu = self.portfolio_means_annual / 12  # Monthly expected returns
            monthly_sigma = self.portfolio_stds_annual / np.sqrt(12)  # Monthly volatilities
            
            # CRITICAL FIX: Convert goal years to goal months for monthly DP
            # Goal years [4, 8, 12, 16] become goal months [47, 95, 143, 191] (0-indexed)
            # This matches the RL environment's goal timing
            goal_months = [(year * 12) - 1 for year in goal_years]  # Convert to 0-indexed months
            logger.info(f"  Converted goal years {goal_years} to goal months {goal_months}")
            
            # Create mapping from monthly timesteps to actual years for cost/utility calculation
            timestep_to_year_mapping = {month: year for month, year in zip(goal_months, goal_years)}
            logger.info(f"  Timestep to year mapping: {timestep_to_year_mapping}")

            dp_config = MultiGoalDPConfig(
                initial_wealth=initial_wealth,
                time_horizon=self.config.time_horizon,  # 192 monthly steps
                goal_years=goal_months,  # Use monthly timesteps, not years!
                goal_timestep_to_year_mapping=timestep_to_year_mapping,  # For cost/utility calculation
                num_portfolios=len(monthly_mu),
                grid_density=2.0,  # Slightly lower resolution due to more time steps
                random_seed=self.config.random_seed,
                # Use monthly portfolio parameters
                custom_mu_array=monthly_mu,
                custom_sigma_array=monthly_sigma
            )

            # Solve DP
            start_time = time.time()
            dp = MultiGoalGBWMDP(dp_config)
            dp.solve()
            solve_time = time.time() - start_time

            self.dp_policies_monthly[num_goals] = dp

            expected_utility = dp.get_expected_utility()
            logger.info(f"  Monthly DP solved in {solve_time:.2f}s, expected utility: {expected_utility:.2f}")

    def simulate_dp_baseline(self, use_vix_market: bool = False):
        """
        Run Monte Carlo simulations for DP policies.

        Args:
            use_vix_market: If True, apply VIX adjustments to market returns.
                           If False (default), use stable market (base μ/σ).

        Goal Timing (CRITICAL):
        - DP operates on yearly time t = 0, 1, ..., 15
        - Goals in DP are at goal_years like [4, 8, 12, 16]
        - Monthly simulation: goal at year Y is checked at month Y*12
        - DP policy for year Y goal should be consulted at yearly time t = Y
        - Example: year 4 goal at month 48 → DP consulted at t = 4
        """
        # Skip simulation if not needed for current baseline mode
        if self.config.baseline_mode == "annual_stable":
            logger.info("Skipping DP monthly simulation (annual_stable mode uses annual DP)")
            return
            
        market_type = "VIX-ADJUSTED" if use_vix_market else "STABLE (NO VIX)"
        logger.info("=" * 60)
        logger.info(f"SIMULATING DP BASELINE ({market_type})")
        logger.info("=" * 60)

        dt = 1.0 / self.config.months_per_year  # Monthly time step

        for num_goals in self.config.goal_counts:
            # Determine result key for caching
            if self.config.baseline_mode == 'monthly_vix':
                dp_key = "DP (Monthly)" if not use_vix_market else "DP (VIX)"
            else:
                dp_key = "DP"
            
            # Check cache first - skip simulation if cached results exist
            if self._should_skip_simulation(dp_key, num_goals):
                logger.info(f"Skipping {dp_key} simulation for {num_goals} goals (cached results found)")
                continue
                
            # Use monthly DP policies for monthly simulation
            if num_goals not in self.dp_policies_monthly:
                continue

            dp = self.dp_policies_monthly[num_goals]
            initial_wealth = get_initial_wealth(num_goals)
            goal_years = get_goal_years(num_goals)
            goal_months = get_goal_months(num_goals)

            logger.info(f"\nSimulating DP for {num_goals} goals...")
            logger.info(f"  Goal years: {goal_years}, Goal months: {goal_months}")

            rewards = []
            goal_successes = []
            final_wealths = []

            for sim_idx in tqdm(range(self.config.num_simulations), desc=f"DP {num_goals}g", leave=False):
                if sim_idx % 100000 == 0 and sim_idx > 0:
                    logger.info(f"  DP {num_goals} goals: {sim_idx:,}/{self.config.num_simulations:,}")

                sim_market_shocks = self.market_shocks[sim_idx]

                # Create VIX model for TRUE market reality with SHARED market shocks
                # KEY: VIX affects actual returns, but DP doesn't KNOW about it
                # DP makes decisions based on base μ/σ, but faces VIX-adjusted market
                # CRITICAL: VIX must use same shocks as wealth evolution for correlation
                vix_model = create_vix_model(
                    model_type=self.config.vix_model_type,
                    market_shocks=sim_market_shocks.reshape(1, -1),  # Shape (1, time_horizon)
                    seed=self.config.random_seed + sim_idx,
                    kappa=self.vix_params.kappa,
                    theta=self.vix_params.theta,
                    sigma_v=self.vix_params.sigma_v,
                    lambda_jump=self.vix_params.lambda_jump,
                    mu_jump=self.vix_params.mu_jump,
                    sigma_jump=self.vix_params.sigma_jump,
                    beta_sensitivity=self.vix_params.beta_sensitivity,
                    delta_sensitivity=self.vix_params.delta_sensitivity
                )
                vix_model.reset(episode_idx=sim_idx, sim_idx=0)

                wealth = initial_wealth
                total_reward = 0
                goals_taken = []

                for step in range(self.config.time_horizon):
                    # Check if this step is a goal month
                    # Goals are at months 48, 96, 144, 192 (= years 4, 8, 12, 16 × 12)
                    current_month = step + 1  # 1-indexed month
                    is_goal_month = current_month in goal_months

                    if is_goal_month:
                        # Compute which goal year this corresponds to
                        goal_year = current_month // self.config.months_per_year

                        if goal_year not in goals_taken:
                            # CRITICAL: Get DP strategy at the correct yearly time
                            # DP's goal_years = [4, 8, 12, 16] means goals at t=4, 8, 12, 16
                            # But t=16 is terminal, so year 16 goal is at t=15 with special handling
                            if goal_year == self.config.years_horizon:
                                dp_time = self.config.years_horizon - 1  # t=15 for year 16
                            else:
                                dp_time = goal_year  # t=4 for year 4, etc.

                            # For monthly DP, use current monthly step as time
                            goal_action, portfolio_idx, base_mu, base_sigma = dp.get_optimal_strategy(wealth, step)

                            goal_cost = get_goal_cost(goal_year)
                            goal_utility = get_goal_utility(goal_year)

                            if goal_action == 1 and wealth >= goal_cost:
                                total_reward += goal_utility
                                wealth -= goal_cost
                                goals_taken.append(goal_year)

                    # Get portfolio choice for wealth evolution
                    # For monthly DP, use current monthly step as time
                    _, portfolio_idx, _, _ = dp.get_optimal_strategy(wealth, step)

                    # Wealth evolution
                    if step < self.config.time_horizon - 1:
                        shock = sim_market_shocks[step]

                        # Get base portfolio parameters
                        base_mu = self.portfolio_means_annual[portfolio_idx]
                        base_sigma = self.portfolio_stds_annual[portfolio_idx]

                        if use_vix_market:
                            # VIX-ADJUSTED MARKET: Apply VIX adjustments to actual returns
                            # This is the real-world scenario where VIX predicts returns
                            vix_avg = vix_model.get_vix_average()
                            vix_normalized = (self.vix_params.theta - vix_avg) / self.vix_params.theta
                            portfolio_beta = self.portfolio_betas[portfolio_idx] * vix_normalized
                            portfolio_delta = self.portfolio_deltas[portfolio_idx] * vix_normalized

                            mu_actual = base_mu + portfolio_beta

                            # σ adjustment depends on volatility_method:
                            # - rolling_vol: δ adjusts σ directly → σ_adj = σ - δ
                            # - return_squared: δ adjusts variance → σ_adj = √(σ² - δ)
                            if self.config.volatility_method == 'rolling_vol':
                                sigma_actual = base_sigma - portfolio_delta
                            else:  # return_squared
                                variance_adj = max(base_sigma**2 - portfolio_delta, 0.0004)
                                sigma_actual = np.sqrt(variance_adj)

                            mu_actual = np.clip(mu_actual, -0.15, 0.30)
                            sigma_actual = np.clip(sigma_actual, 0.02, 0.50)
                        else:
                            # STABLE MARKET: Use base μ/σ without VIX adjustments
                            mu_actual = base_mu
                            sigma_actual = base_sigma

                        # GBM wealth evolution
                        drift = (mu_actual - 0.5 * sigma_actual**2) * dt
                        diffusion = sigma_actual * np.sqrt(dt) * shock
                        wealth = wealth * np.exp(drift + diffusion)
                        wealth = max(0, wealth)

                    # Advance VIX for next step
                    vix_model.step_vix(dt=dt)

                rewards.append(total_reward)
                goal_successes.append(len(goals_taken) / len(goal_years))
                final_wealths.append(wealth)

            # dp_key already determined in caching section above
            result = SimulationResult(
                method_name=dp_key,
                num_goals=num_goals,
                mean_reward=np.mean(rewards),
                std_reward=np.std(rewards),
                efficiency=1.0,  # DP is baseline
                num_simulations=self.config.num_simulations,
                goal_success_rate=np.mean(goal_successes),
                mean_final_wealth=np.mean(final_wealths),
                std_final_wealth=np.std(final_wealths)
            )
            
            self.results[dp_key][num_goals] = result
            
            # Cache the simulation result
            self._save_simulation_result(dp_key, num_goals, result)

            logger.info(f"  DP {num_goals} goals: reward={np.mean(rewards):.2f}, "
                        f"success={np.mean(goal_successes):.3f}")

    def simulate_pure_rl(self):
        """
        Run Monte Carlo simulations for Pure RL agents (monthly, no VIX).

        This provides FAIR comparison with Sentiment RL:
        - Same monthly time steps (192)
        - Same DP baseline (no VIX adjustments)
        - Same market shocks
        - ONLY DIFFERENCE: No VIX in state, no VIX adjustments to μ/σ

        State: 2D [normalized_time, normalized_wealth]
        """
        # Skip simulation if not needed for current baseline mode
        if self.config.baseline_mode == "annual_stable":
            logger.info("Skipping Pure RL monthly simulation (annual_stable mode uses annual Pure RL)")
            return
            
        logger.info("=" * 60)
        logger.info("SIMULATING PURE RL AGENTS (MONTHLY, NO VIX)")
        logger.info("=" * 60)

        dt = 1.0 / self.config.months_per_year

        for num_goals in self.config.goal_counts:
            # Check cache first - skip simulation if cached results exist
            if self._should_skip_simulation("Pure RL (Monthly)", num_goals):
                logger.info(f"Skipping Pure RL (Monthly) simulation for {num_goals} goals (cached results found)")
                continue
                
            if num_goals not in self.pure_rl_agents:
                continue

            agent = self.pure_rl_agents[num_goals]
            agent.policy_net.eval()

            initial_wealth = get_initial_wealth(num_goals)
            goal_years = get_goal_years(num_goals)
            goal_months = get_goal_months(num_goals)

            logger.info(f"\nSimulating Pure RL for {num_goals} goals...")

            rewards = []
            goal_successes = []
            final_wealths = []

            for sim_idx in tqdm(range(self.config.num_simulations), desc=f"Pure RL {num_goals}g", leave=False):
                if sim_idx % 100000 == 0 and sim_idx > 0:
                    logger.info(f"  Pure RL {num_goals} goals: {sim_idx:,}/{self.config.num_simulations:,}")

                market_shocks = self.market_shocks[sim_idx]

                wealth = initial_wealth
                total_reward = 0
                goals_taken = []

                for step in range(self.config.time_horizon):
                    # Build 2D state (NO VIX features)
                    normalized_time = step / self.config.time_horizon
                    normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)
                    state = np.array([normalized_time, normalized_wealth], dtype=np.float32)

                    # Get action from Pure RL agent
                    with torch.no_grad():
                        action = agent.predict(state, deterministic=True)

                    goal_action = int(action[0])
                    portfolio_action = int(action[1])

                    # Check for goal (same logic as Sentiment RL)
                    current_month = step + 1
                    is_goal_month = current_month in goal_months

                    if is_goal_month:
                        goal_year = current_month // self.config.months_per_year

                        if goal_year not in goals_taken:
                            goal_cost = get_goal_cost(goal_year)
                            goal_utility = get_goal_utility(goal_year)

                            if goal_action == 1 and wealth >= goal_cost:
                                total_reward += goal_utility
                                wealth -= goal_cost
                                goals_taken.append(goal_year)

                    # Wealth evolution WITHOUT VIX adjustments (same as DP)
                    if step < self.config.time_horizon - 1:
                        shock = market_shocks[step]

                        # Use base μ/σ (NO VIX adjustment)
                        mu = self.portfolio_means_annual[portfolio_action]
                        sigma = self.portfolio_stds_annual[portfolio_action]

                        # GBM with monthly dt
                        drift = (mu - 0.5 * sigma**2) * dt
                        diffusion = sigma * np.sqrt(dt) * shock
                        wealth = wealth * np.exp(drift + diffusion)
                        wealth = max(0, wealth)

                rewards.append(total_reward)
                goal_successes.append(len(goals_taken) / len(goal_years))
                final_wealths.append(wealth)

            # Calculate efficiency vs DP
            dp_key = "DP (Monthly)" if self.config.baseline_mode == 'monthly_vix' else "DP"
            if dp_key in self.results and num_goals in self.results[dp_key]:
                dp_mean_reward = self.results[dp_key][num_goals].mean_reward
                efficiency = np.mean(rewards) / dp_mean_reward if dp_mean_reward > 0 else 0.0
            else:
                logger.warning(f"DP results not found for {num_goals} goals in Pure RL simulation using key '{dp_key}'. Setting efficiency to 0.0")
                efficiency = 0.0

            result = SimulationResult(
                method_name="Pure RL (Monthly)",
                num_goals=num_goals,
                mean_reward=np.mean(rewards),
                std_reward=np.std(rewards),
                efficiency=efficiency,
                num_simulations=self.config.num_simulations,
                goal_success_rate=np.mean(goal_successes),
                mean_final_wealth=np.mean(final_wealths),
                std_final_wealth=np.std(final_wealths),
                mean_vix_sentiment=0.0  # No VIX
            )
            
            self.results["Pure RL (Monthly)"][num_goals] = result
            
            # Cache the simulation result
            self._save_simulation_result("Pure RL (Monthly)", num_goals, result)

            logger.info(f"  Pure RL {num_goals} goals: reward={np.mean(rewards):.2f}, "
                        f"efficiency={efficiency:.3f}")

    def simulate_pure_rl_vix_market(self):
        """
        Run Monte Carlo simulations for Pure RL agents in VIX-ADJUSTED market - REMOVED for cleanup.
        Not needed for simplified monthly_vix mode requirements.
        """
        logger.info("Skipping Pure RL VIX Market simulation (removed in cleanup - not needed for monthly_vix mode)")
        return


        dt = 1.0 / self.config.months_per_year

        for num_goals in self.config.goal_counts:
            if num_goals not in self.pure_rl_vix_agents:
                continue

            agent = self.pure_rl_vix_agents[num_goals]
            agent.policy_net.eval()

            initial_wealth = get_initial_wealth(num_goals)
            goal_years = get_goal_years(num_goals)
            goal_months = get_goal_months(num_goals)

            logger.info(f"\nSimulating Pure RL (VIX Market) for {num_goals} goals...")

            rewards = []
            goal_successes = []
            final_wealths = []

            for sim_idx in tqdm(range(self.config.num_simulations), desc=f"Pure RL VIX {num_goals}g", leave=False):
                if sim_idx % 100000 == 0 and sim_idx > 0:
                    logger.info(f"  Pure RL VIX {num_goals} goals: {sim_idx:,}/{self.config.num_simulations:,}")

                sim_market_shocks = self.market_shocks[sim_idx]

                # Create VIX model for TRUE market reality with shared shocks
                # KEY: VIX affects actual returns, but Pure RL doesn't SEE it in state
                # Uses same market_shocks for VIX-return correlation (Heston model)
                vix_model = create_vix_model(
                    model_type=self.config.vix_model_type,
                    market_shocks=sim_market_shocks.reshape(1, -1),  # Shape (1, time_horizon)
                    seed=self.config.random_seed + sim_idx,
                    kappa=self.vix_params.kappa,
                    theta=self.vix_params.theta,
                    sigma_v=self.vix_params.sigma_v,
                    lambda_jump=self.vix_params.lambda_jump,
                    mu_jump=self.vix_params.mu_jump,
                    sigma_jump=self.vix_params.sigma_jump,
                    beta_sensitivity=self.vix_params.beta_sensitivity,
                    delta_sensitivity=self.vix_params.delta_sensitivity
                )
                vix_model.reset(episode_idx=sim_idx, sim_idx=0)

                wealth = initial_wealth
                total_reward = 0
                goals_taken = []

                for step in range(self.config.time_horizon):
                    # Build 2D state (NO VIX features - Pure RL doesn't know about VIX)
                    normalized_time = step / self.config.time_horizon
                    normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)
                    state = np.array([normalized_time, normalized_wealth], dtype=np.float32)

                    # Get action from Pure RL agent (trained without VIX)
                    with torch.no_grad():
                        action = agent.predict(state, deterministic=True)

                    goal_action = int(action[0])
                    portfolio_action = int(action[1])

                    # Check for goal (same logic as other methods)
                    current_month = step + 1
                    is_goal_month = current_month in goal_months

                    if is_goal_month:
                        goal_year = current_month // self.config.months_per_year

                        if goal_year not in goals_taken:
                            goal_cost = get_goal_cost(goal_year)
                            goal_utility = get_goal_utility(goal_year)

                            if goal_action == 1 and wealth >= goal_cost:
                                total_reward += goal_utility
                                wealth -= goal_cost
                                goals_taken.append(goal_year)

                    # Wealth evolution WITH VIX adjustments (same as Sentiment RL)
                    if step < self.config.time_horizon - 1:
                        shock = sim_market_shocks[step]

                        # Get VIX adjustment (Pure RL doesn't see this, but market uses it)
                        vix_avg = vix_model.get_vix_average()
                        vix_normalized = (self.vix_params.theta - vix_avg) / self.vix_params.theta

                        base_mu = self.portfolio_means_annual[portfolio_action]
                        base_sigma = self.portfolio_stds_annual[portfolio_action]

                        # Apply VIX adjustments to market (same as Sentiment RL)
                        portfolio_beta = self.portfolio_betas[portfolio_action] * vix_normalized
                        portfolio_delta = self.portfolio_deltas[portfolio_action] * vix_normalized

                        mu_adj = base_mu + portfolio_beta

                        # σ adjustment depends on volatility_method
                        if self.config.volatility_method == 'rolling_vol':
                            sigma_adj = base_sigma - portfolio_delta
                        else:  # return_squared
                            variance_adj = max(base_sigma**2 - portfolio_delta, 0.0004)
                            sigma_adj = np.sqrt(variance_adj)

                        mu_adj = np.clip(mu_adj, -0.15, 0.30)
                        sigma_adj = np.clip(sigma_adj, 0.02, 0.50)

                        # GBM with monthly dt
                        drift = (mu_adj - 0.5 * sigma_adj**2) * dt
                        diffusion = sigma_adj * np.sqrt(dt) * shock
                        wealth = wealth * np.exp(drift + diffusion)
                        wealth = max(0, wealth)

                    # Advance VIX for next step
                    vix_model.step_vix(dt=dt)

                rewards.append(total_reward)
                goal_successes.append(len(goals_taken) / len(goal_years))
                final_wealths.append(wealth)

            # Calculate efficiency vs DP (stable market baseline)
            if "DP" in self.results and num_goals in self.results["DP"]:
                dp_mean_reward = self.results["DP"][num_goals].mean_reward
                efficiency = np.mean(rewards) / dp_mean_reward if dp_mean_reward > 0 else 0.0
            else:
                logger.warning(f"DP results not found for {num_goals} goals in Pure RL (VIX Market) simulation. Setting efficiency to 0.0")
                efficiency = 0.0

            # Store in appropriate key - VIX Market mode not in baseline initialization
            # Use Pure RL (Monthly) instead as this is VIX market simulation
            if "Pure RL (Monthly)" not in self.results:
                logger.warning(f"Pure RL (Monthly) key not found, skipping VIX market results storage")
                continue
            
            # Note: This method appears to be for VIX market but we store in Pure RL Monthly
            # since Pure RL (VIX Market) is not in the initialized keys
            logger.warning(f"Storing VIX market results in Pure RL (Monthly) - consider adding Pure RL (VIX Market) to initialization")
            self.results["Pure RL (Monthly)"][num_goals] = SimulationResult(
                method_name="Pure RL (Monthly)",  # Changed from VIX Market to match initialized key
                num_goals=num_goals,
                mean_reward=np.mean(rewards),
                std_reward=np.std(rewards),
                efficiency=efficiency,
                num_simulations=self.config.num_simulations,
                goal_success_rate=np.mean(goal_successes),
                mean_final_wealth=np.mean(final_wealths),
                std_final_wealth=np.std(final_wealths),
                mean_vix_sentiment=0.0  # Pure RL doesn't observe VIX
            )

            logger.info(f"  Pure RL (VIX Market) {num_goals} goals: reward={np.mean(rewards):.2f}, "
                        f"efficiency={efficiency:.3f}")

    def simulate_sentiment_rl(self, use_vix_market: bool = False, result_key: str = "Sentiment RL", agent_container: str = None):
        """
        Run Monte Carlo simulations for Sentiment RL agents.

        Args:
            use_vix_market: If True, apply VIX adjustments to market returns
            result_key: Key to store results in self.results dictionary
            agent_container: 'simple', 'advanced', or None (for legacy compatibility)

        Goal Timing (matches GBWMEnvMonthly):
        - Environment checks goals at: self._is_goal_month(self.current_month + 1)
        - goal_months = [48, 96, 144, 192] for 4 goals
        - At step where current_month = 47, checks month 48 → goal year 4

        State Normalization (matches GBWMEnvMonthly._get_observation):
        - normalized_time = current_month / total_months
        - normalized_wealth = wealth / (initial_wealth * 10)
        """
        # Skip simulation if not needed for current baseline mode
        if self.config.baseline_mode == "annual_stable":
            logger.info("Skipping Sentiment RL simulation (annual_stable mode focuses on DP vs Pure RL)")
            return
            
        logger.info("=" * 60)
        logger.info("SIMULATING SENTIMENT RL AGENTS")
        if agent_container:
            logger.info(f"Agent Container: {agent_container.upper()}")
        logger.info("=" * 60)

        dt = 1.0 / self.config.months_per_year

        # Select appropriate agent container
        if agent_container == 'simple':
            agent_dict = self.sentiment_agents_simple
        elif agent_container == 'advanced':
            agent_dict = self.sentiment_agents_advanced
        elif agent_container == 'stable':
            agent_dict = self.sentiment_rl_stable_agents
        else:
            # Legacy behavior - use main container
            agent_dict = self.sentiment_agents

        for num_goals in self.config.goal_counts:
            # Check cache first - skip simulation if cached results exist
            if self._should_skip_simulation(result_key, num_goals):
                logger.info(f"Skipping {result_key} simulation for {num_goals} goals (cached results found)")
                continue
                
            if num_goals not in agent_dict:
                continue

            agent = agent_dict[num_goals]
            agent.policy_net.eval()

            initial_wealth = get_initial_wealth(num_goals)
            goal_years = get_goal_years(num_goals)
            goal_months = get_goal_months(num_goals)

            logger.info(f"\nSimulating Sentiment RL for {num_goals} goals...")
            logger.info(f"  Goal years: {goal_years}, Goal months: {goal_months}")

            rewards = []
            goal_successes = []
            final_wealths = []
            vix_sentiments = []

            for sim_idx in tqdm(range(self.config.num_simulations), desc=f"Sent RL {num_goals}g", leave=False):
                if sim_idx % 100000 == 0 and sim_idx > 0:
                    logger.info(f"  Sentiment RL {num_goals} goals: {sim_idx:,}/{self.config.num_simulations:,}")

                sim_market_shocks = self.market_shocks[sim_idx]

                # Create VIX model with SHARED market shocks for VIX-return correlation
                # CRITICAL: VIX must use same shocks as wealth evolution (Heston model)
                vix_model = create_vix_model(
                    model_type=self.config.vix_model_type,
                    market_shocks=sim_market_shocks.reshape(1, -1),  # Shape (1, time_horizon)
                    seed=self.config.random_seed + sim_idx,
                    kappa=self.vix_params.kappa,
                    theta=self.vix_params.theta,
                    sigma_v=self.vix_params.sigma_v,
                    lambda_jump=self.vix_params.lambda_jump,
                    mu_jump=self.vix_params.mu_jump,
                    sigma_jump=self.vix_params.sigma_jump,
                    beta_sensitivity=self.vix_params.beta_sensitivity,
                    delta_sensitivity=self.vix_params.delta_sensitivity
                )
                vix_model.reset(episode_idx=sim_idx, sim_idx=0)

                wealth = initial_wealth
                total_reward = 0
                goals_taken = []
                episode_vix_sentiments = []

                for step in range(self.config.time_horizon):
                    # Get VIX features for state
                    vix_features = vix_model.get_sentiment_features()
                    vix_avg = vix_model.get_vix_average()
                    beta, delta = vix_model.get_adjustments(vix_avg)

                    episode_vix_sentiments.append(vix_features[0])  # vix_level_norm

                    # Build 5D state - MUST match environment's _get_observation() method
                    # Environment: norm_time = current_month / total_months
                    # Environment: norm_wealth = min(current_wealth / (initial_wealth * 10), 1.0)
                    normalized_time = step / self.config.time_horizon
                    normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)
                    state = np.array([normalized_time, normalized_wealth] + list(vix_features), dtype=np.float32)

                    # Get action from Sentiment RL agent
                    with torch.no_grad():
                        action = agent.predict(state, deterministic=True)

                    goal_action = int(action[0])
                    portfolio_action = int(action[1])

                    # Check for goal - matches environment's step() logic
                    # Environment: self._is_goal_month(self.current_month + 1)
                    # At step where current_month = step, checks month (step + 1)
                    current_month = step + 1  # 1-indexed month being processed
                    is_goal_month = current_month in goal_months

                    if is_goal_month:
                        goal_year = current_month // self.config.months_per_year

                        if goal_year not in goals_taken:
                            goal_cost = get_goal_cost(goal_year)
                            goal_utility = get_goal_utility(goal_year)

                            if goal_action == 1 and wealth >= goal_cost:
                                total_reward += goal_utility
                                wealth -= goal_cost
                                goals_taken.append(goal_year)

                    # Wealth evolution with VIX-adjusted parameters
                    if step < self.config.time_horizon - 1:
                        shock = sim_market_shocks[step]

                        base_mu = self.portfolio_means_annual[portfolio_action]
                        base_sigma = self.portfolio_stds_annual[portfolio_action]

                        # Use portfolio-specific β/δ for accurate VIX adjustment
                        # β_adj = β_sensitivity[portfolio] × (θ - VIX_avg) / θ
                        vix_normalized = (self.vix_params.theta - vix_avg) / self.vix_params.theta
                        portfolio_beta = self.portfolio_betas[portfolio_action] * vix_normalized
                        portfolio_delta = self.portfolio_deltas[portfolio_action] * vix_normalized

                        # Sentiment RL uses VIX-adjusted parameters
                        if use_vix_market:
                            mu_adj = base_mu + portfolio_beta

                            # σ adjustment depends on volatility_method:
                            # - rolling_vol: δ adjusts σ directly → σ_adj = σ - δ
                            # - return_squared: δ adjusts variance → σ_adj = √(σ² - δ)
                            if self.config.volatility_method == 'rolling_vol':
                                sigma_adj = base_sigma - portfolio_delta
                            else:  # return_squared
                                variance_adj = max(base_sigma**2 - portfolio_delta, 0.0004)
                                sigma_adj = np.sqrt(variance_adj)

                            mu_adj = np.clip(mu_adj, -0.15, 0.30)
                            sigma_adj = np.clip(sigma_adj, 0.02, 0.50)
                        else:
                            # Use base parameters (stable market)
                            mu_adj = base_mu
                            sigma_adj = base_sigma

                        # GBM with monthly dt
                        drift = (mu_adj - 0.5 * sigma_adj**2) * dt
                        diffusion = sigma_adj * np.sqrt(dt) * shock
                        wealth = wealth * np.exp(drift + diffusion)
                        wealth = max(0, wealth)

                    # Advance VIX
                    vix_model.step_vix(dt=dt)

                rewards.append(total_reward)
                goal_successes.append(len(goals_taken) / len(goal_years))
                final_wealths.append(wealth)
                vix_sentiments.append(np.mean(episode_vix_sentiments))

            # Calculate efficiency vs DP
            if self.config.baseline_mode == 'monthly_vix':
                dp_key = "DP (Monthly)"
            elif self.config.baseline_mode == 'annual_stable':
                dp_key = "DP" if "DP" in self.results else "DP (Annual Stable)"
            else:
                raise ValueError(f"Unsupported baseline_mode: {self.config.baseline_mode}")
            
            # Check if the specific goal count exists in DP results
            if num_goals not in self.results[dp_key]:
                logger.error(f"DP results not found for {num_goals} goals. Available keys: {list(self.results[dp_key].keys())}")
                efficiency = 0.0
                dp_mean_reward = 1.0  # Prevent division by zero
            else:
                dp_mean_reward = self.results[dp_key][num_goals].mean_reward
            efficiency = np.mean(rewards) / dp_mean_reward if dp_mean_reward > 0 else 0.0

            result = SimulationResult(
                method_name=result_key,
                num_goals=num_goals,
                mean_reward=np.mean(rewards),
                std_reward=np.std(rewards),
                efficiency=efficiency,
                num_simulations=self.config.num_simulations,
                goal_success_rate=np.mean(goal_successes),
                mean_final_wealth=np.mean(final_wealths),
                std_final_wealth=np.std(final_wealths),
                mean_vix_sentiment=np.mean(vix_sentiments)
            )

            self.results[result_key][num_goals] = result
            
            # Cache the simulation result
            self._save_simulation_result(result_key, num_goals, result)

            logger.info(f"  Sentiment RL {num_goals} goals: reward={np.mean(rewards):.2f}, "
                        f"efficiency={efficiency:.3f}")

    def generate_figure1(self):
        """Generate Figure 1: Efficiency vs Number of Goals (for non-annual_stable modes)"""
        # Skip if annual_stable mode (uses its own visualization)
        if self.config.baseline_mode == "annual_stable":
            logger.info("Skipping Figure 1 generation (annual_stable mode uses generate_annual_stable_figure)")
            return
            
        logger.info("=" * 60)
        logger.info("GENERATING FIGURE 1: EFFICIENCY VS NUMBER OF GOALS")
        logger.info("=" * 60)

        fig, ax = plt.subplots(figsize=(14, 8))

        # Define styles for all methods
        styles = {
            "DP": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "linewidth": 2.5},
            "Pure RL": {"color": "#ff7f0e", "marker": "s", "linestyle": "--", "linewidth": 2.5},
            "Pure RL (VIX Market)": {"color": "#d62728", "marker": "D", "linestyle": "-.", "linewidth": 2.5},
            "Sentiment RL": {"color": "#2ca02c", "marker": "^", "linestyle": "-", "linewidth": 2.5}
        }

        # Plot each method
        for method_name, method_results in self.results.items():
            if not method_results:
                continue

            goals = sorted(method_results.keys())
            efficiencies = [method_results[g].efficiency * 100 for g in goals]

            style = styles.get(method_name)
            if style is None:
                continue
            ax.plot(goals, efficiencies,
                   color=style["color"],
                   marker=style["marker"],
                   linestyle=style["linestyle"],
                   linewidth=style["linewidth"],
                   markersize=10,
                   label=method_name)

        # Formatting
        ax.set_xlabel("Number of Goals", fontsize=14, fontweight='bold')
        ax.set_ylabel("Efficiency (% of DP Optimal)", fontsize=14, fontweight='bold')
        ax.set_title(f"Monthly Comparison: Isolating VIX State vs VIX Market Effect\n"
                     f"({self.config.num_simulations:,} Monte Carlo Simulations, "
                     f"{self.config.time_horizon} Monthly Steps)",
                    fontsize=16, fontweight='bold')

        ax.set_xticks(self.config.goal_counts)
        ax.set_ylim(0, 110)

        ax.axhline(y=100, color='black', linestyle='--', alpha=0.3, label='DP Optimal (100%)')

        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        fig_path = self.output_dir / "figure1_sentiment_efficiency_vs_goals.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved: {fig_path}")

    def generate_annual_stable_figure(self):
        """Generate figure for annual_stable comparison mode - DP vs Pure RL (Annual only)"""
        logger.info("=" * 60)
        logger.info("GENERATING ANNUAL STABLE COMPARISON FIGURE")
        logger.info("=" * 60)

        # Single plot: Pure RL (Annual) vs DP (Annual Stable)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        dp_annual = self.results.get("DP (Annual Stable)", {})
        pure_rl_annual = self.results.get("Pure RL (Annual)", {})

        if dp_annual and pure_rl_annual:
            goals = sorted(dp_annual.keys())
            dp_rewards = [dp_annual[g].mean_reward for g in goals]
            pure_rewards = [pure_rl_annual[g].mean_reward for g in goals]
            pure_eff = [pure_rl_annual[g].efficiency * 100 for g in goals]

            # Bar chart
            x_pos = np.arange(len(goals))
            ax.bar(x_pos - 0.2, dp_rewards, 0.4, label='DP (Optimal)', color='#1f77b4', alpha=0.8)
            ax.bar(x_pos + 0.2, pure_rewards, 0.4, label='Pure RL (Annual)', color='#ff7f0e', alpha=0.8)

            # Add efficiency labels on Pure RL bars
            for i, (g, eff) in enumerate(zip(goals, pure_eff)):
                ax.annotate(f'{eff:.1f}%', xy=(x_pos[i] + 0.2, pure_rewards[i]),
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Efficiency line plot overlay
            ax2 = ax.twinx()
            ax2.plot(x_pos, pure_eff, 'ro-', linewidth=2, markersize=6, label='Efficiency %')
            ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Optimal (100%)')
            ax2.set_ylabel('Efficiency (% of DP)', fontsize=12, fontweight='bold', color='red')
            ax2.set_ylim(0, max(pure_eff) * 1.1)
            ax2.tick_params(axis='y', labelcolor='red')

            ax.set_xlabel('Number of Goals', fontsize=14, fontweight='bold')
            ax.set_ylabel('Mean Reward', fontsize=14, fontweight='bold')
            ax.set_title(f'Pure RL (Annual) vs DP Optimal - Efficiency Analysis\n' +
                        f'{self.config.num_simulations:,} simulations, 16-year horizon',
                        fontsize=16, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(goals)
            ax.legend(loc='upper left', fontsize=11)
            ax2.legend(loc='upper right', fontsize=11)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No results to display\nCheck DP and Pure RL results', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)

        plt.tight_layout()

        fig_path = self.output_dir / "figure_annual_stable_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved: {fig_path}")

    def print_annual_stable_summary(self):
        """Print summary table for annual_stable comparison mode - DP vs Pure RL only"""
        logger.info("\n" + "=" * 80)
        logger.info("ANNUAL STABLE COMPARISON SUMMARY")
        logger.info("=" * 80)

        print("\n" + "=" * 80)
        print("COMPARISON: Pure RL (Annual) vs DP (Optimal)")
        print("  - DP: Theoretical optimum using MultiGoal DP")
        print("  - Pure RL: Annual trajectories (16 steps), Stable market (no VIX)")
        print("  - Both use same base market randomness (derived from monthly)")
        print("=" * 80)

        # Header
        print(f"\n{'Goals':<8} | {'DP (Optimal)':<15} | {'Pure RL (Annual)':<17} | {'Efficiency':<12}")
        print(f"{'':8} | {'Reward':<15} | {'Reward':<17} | {'(% of DP)':<12}")
        print("-" * 80)

        dp_annual = self.results.get("DP (Annual Stable)", {})
        pure_rl = self.results.get("Pure RL (Annual)", {})

        if not dp_annual or not pure_rl:
            print("ERROR: Missing results for comparison")
            print(f"  DP results: {bool(dp_annual)}")
            print(f"  Pure RL results: {bool(pure_rl)}")
            return

        total_efficiency = 0
        valid_goals = 0

        for num_goals in self.config.goal_counts:
            dp_result = dp_annual.get(num_goals)
            rl_result = pure_rl.get(num_goals)

            if dp_result and rl_result:
                efficiency = rl_result.efficiency * 100
                total_efficiency += efficiency
                valid_goals += 1
                
                print(f"{num_goals:<8} | "
                      f"{dp_result.mean_reward:<15.2f} | "
                      f"{rl_result.mean_reward:<17.2f} | "
                      f"{efficiency:<12.1f}%")

        print("=" * 80)

        # Calculate summary statistics
        if valid_goals > 0:
            avg_efficiency = total_efficiency / valid_goals
            print(f"\nSUMMARY:")
            print(f"  Average Pure RL Efficiency: {avg_efficiency:.1f}% of DP optimal")
            
            if avg_efficiency > 90:
                print("  Pure RL performs very well (>90% efficiency)")
            elif avg_efficiency > 70:
                print("  Pure RL shows decent performance (70-90% efficiency)")
            else:
                print("  Pure RL significantly underperforms (<70% efficiency)")
                
            # Goal success analysis
            dp_success_rates = [dp_annual[g].goal_success_rate for g in self.config.goal_counts if g in dp_annual]
            rl_success_rates = [pure_rl[g].goal_success_rate for g in self.config.goal_counts if g in pure_rl]
            
            if dp_success_rates and rl_success_rates:
                print(f"  Average DP Goal Success Rate: {np.mean(dp_success_rates):.1%}")
                print(f"  Average RL Goal Success Rate: {np.mean(rl_success_rates):.1%}")

        print("=" * 80)

    def save_results(self):
        """Save all results to JSON"""

        def convert_to_native(obj):
            """Convert numpy types to Python native types for JSON serialization"""
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            return obj

        results_dict = {
            'config': {
                'num_simulations': self.config.num_simulations,
                'num_iterations': self.config.num_iterations,
                'time_horizon': self.config.time_horizon,
                'beta': float(self.beta) if self.beta else None,
                'delta': float(self.delta) if self.delta else None,
                'policy_type': self.config.policy_type,
                'encoder_type': self.config.encoder_type
            },
            'results': {}
        }

        for method_name, method_results in self.results.items():
            results_dict['results'][method_name] = {}
            for num_goals, result in method_results.items():
                # Convert dataclass to dict and then convert numpy types
                result_dict = asdict(result)
                results_dict['results'][method_name][str(num_goals)] = convert_to_native(result_dict)

        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results saved to: {results_path}")

    def _save_checkpoint(self, stage: str, data: dict):
        """Save checkpoint for long evaluations"""
        if not self.enable_checkpoints:
            return
        
        checkpoint_data = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'data': data
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{stage}.json"
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            logger.info(f"  Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, stage: str) -> Optional[dict]:
        """Load checkpoint if available"""
        if not self.enable_checkpoints:
            return None
            
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{stage}.json"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {stage}: {e}")
        return None

    def print_summary(self):
        """Print simple summary table for monthly_vix mode"""
        logger.info("\n" + "=" * 100)
        logger.info("EVALUATION SUMMARY - MONTHLY VIX COMPARISON")
        logger.info("=" * 100)

        # Print Monthly Results
        print("\n" + "=" * 100)
        print("MONTHLY COMPARISON RESULTS")
        print("=" * 100)

        print(f"\n{'Goals':<6} | {'DP (Monthly)':<12} | {'Pure RL (Monthly)':<17} | {'Sentiment RL':<14} | {'Pure Eff':<9} | {'Sent Eff':<9}")
        print("-" * 100)

        for num_goals in self.config.goal_counts:
            dp_result = self.results.get("DP (Monthly)", {}).get(num_goals)
            pure_result = self.results.get("Pure RL (Monthly)", {}).get(num_goals)
            sent_result = self.results.get("Sentiment RL (Monthly)", {}).get(num_goals)

            if dp_result and pure_result and sent_result:
                # Calculate efficiency percentages
                pure_eff = pure_result.efficiency * 100 if pure_result.efficiency <= 1.0 else pure_result.efficiency
                sent_eff = sent_result.efficiency * 100 if sent_result.efficiency <= 1.0 else sent_result.efficiency

                print(f"{num_goals:<6} | "
                      f"{dp_result.mean_reward:<12.2f} | "
                      f"{pure_result.mean_reward:<17.2f} | "
                      f"{sent_result.mean_reward:<14.2f} | "
                      f"{pure_eff:<9.1f}% | "
                      f"{sent_eff:<9.1f}%")

        print("=" * 100)

        # Print Annual Results  
        print("\n" + "=" * 100)
        print("ANNUAL COMPARISON RESULTS (derived from monthly seeds)")
        print("=" * 100)

        print(f"\n{'Goals':<6} | {'DP (Annual)':<12} | {'Pure RL (Annual)':<17} | {'DP Ann Eff':<11} | {'Pure Ann Eff':<12}")
        print("-" * 100)

        for num_goals in self.config.goal_counts:
            dp_annual_result = self.results.get("DP (Annual)", {}).get(num_goals)
            pure_annual_result = self.results.get("Pure RL (Annual)", {}).get(num_goals)
            dp_monthly_result = self.results.get("DP (Monthly)", {}).get(num_goals)

            if dp_annual_result and pure_annual_result and dp_monthly_result:
                # Calculate efficiency relative to monthly DP (since both use same seeds)
                dp_ann_eff = dp_annual_result.efficiency * 100 if dp_annual_result.efficiency <= 1.0 else dp_annual_result.efficiency
                pure_ann_eff = (pure_annual_result.mean_reward / dp_monthly_result.mean_reward) * 100

                print(f"{num_goals:<6} | "
                      f"{dp_annual_result.mean_reward:<12.2f} | "
                      f"{pure_annual_result.mean_reward:<17.2f} | "
                      f"{dp_ann_eff:<11.1f}% | "
                      f"{pure_ann_eff:<12.1f}%")

        print("=" * 100)

        # Summary statistics
        efficiencies_pure = []
        efficiencies_sent = []
        efficiencies_pure_annual = []
        
        for num_goals in self.config.goal_counts:
            pure_result = self.results.get("Pure RL (Monthly)", {}).get(num_goals)
            sent_result = self.results.get("Sentiment RL (Monthly)", {}).get(num_goals)
            pure_annual_result = self.results.get("Pure RL (Annual)", {}).get(num_goals)
            dp_monthly_result = self.results.get("DP (Monthly)", {}).get(num_goals)
            
            if pure_result:
                efficiencies_pure.append(pure_result.efficiency * 100)
            if sent_result:
                efficiencies_sent.append(sent_result.efficiency * 100)
            if pure_annual_result and dp_monthly_result:
                efficiencies_pure_annual.append((pure_annual_result.mean_reward / dp_monthly_result.mean_reward) * 100)

        if efficiencies_pure and efficiencies_sent:
            print(f"\nAVERAGE MONTHLY EFFICIENCY:")
            print(f"  Pure RL (Monthly):    {np.mean(efficiencies_pure):.1f}% of DP optimal")
            print(f"  Sentiment RL:         {np.mean(efficiencies_sent):.1f}% of DP optimal")
            print(f"  Sentiment advantage:  {np.mean(efficiencies_sent) - np.mean(efficiencies_pure):+.1f} percentage points")

        if efficiencies_pure_annual:
            print(f"\nAVERAGE ANNUAL EFFICIENCY:")
            print(f"  Pure RL (Annual):     {np.mean(efficiencies_pure_annual):.1f}% of Monthly DP baseline")

        print("=" * 100)

    def print_wealth_summary(self):
        """Print terminal wealth comparison table for available methods"""
        logger.info("\n" + "=" * 90)
        logger.info("TERMINAL WEALTH COMPARISON")
        logger.info("=" * 90)
        
        print("\n" + "=" * 90)
        print("TERMINAL WEALTH COMPARISON (Mean ± Std)")
        print("=" * 90)
        
        # Header with available methods only
        print(f"\n{'Goals':<6} | {'DP (Monthly)':<18} | {'Pure RL (Monthly)':<18} | {'Sentiment RL':<18} | {'DP Annual':<18} | {'PureRL Annual':<18}")
        print(f"{'':6} | {'Mean±Std (k$)':<18} | {'Mean±Std (k$)':<18} | {'Mean±Std (k$)':<18} | {'Mean±Std (k$)':<18} | {'Mean±Std (k$)':<18}")
        print("-" * 90)
        
        for num_goals in self.config.goal_counts:
            dp_result = self.results.get("DP (Monthly)", {}).get(num_goals)
            pure_result = self.results.get("Pure RL (Monthly)", {}).get(num_goals)
            sent_result = self.results.get("Sentiment RL (Monthly)", {}).get(num_goals)
            dp_annual_result = self.results.get("DP (Annual)", {}).get(num_goals)
            pure_annual_result = self.results.get("Pure RL (Annual)", {}).get(num_goals)
            
            # Helper function to format wealth
            def format_wealth(result):
                if result:
                    mean_k = result.mean_final_wealth / 1000
                    std_k = result.std_final_wealth / 1000
                    return f"${mean_k:.0f}±{std_k:.0f}k"
                return "N/A"
            
            # Print row with all wealth data
            print(f"{num_goals:<6} | "
                  f"{format_wealth(dp_result):<18} | "
                  f"{format_wealth(pure_result):<18} | "
                  f"{format_wealth(sent_result):<18} | "
                  f"{format_wealth(dp_annual_result):<18} | "
                  f"{format_wealth(pure_annual_result):<18}")
        
        print("=" * 130)
        
        # Summary insights
        print("\n" + "=" * 80)
        print("TERMINAL WEALTH INSIGHTS")
        print("=" * 80)
        
        # Compare different effects - use baseline-appropriate keys
        for num_goals in self.config.goal_counts:
            # Use baseline-mode appropriate keys
            if self.config.baseline_mode == 'annual_stable':
                dp_result = self.results.get("DP", {}).get(num_goals) or self.results.get("DP (Annual Stable)", {}).get(num_goals)
                pure_result = self.results.get("Pure RL (Annual)", {}).get(num_goals) 
                sent_result = None  # No sentiment RL in annual_stable mode
            else:  # monthly_vix mode
                dp_result = self.results.get("DP (Monthly)", {}).get(num_goals)
                pure_result = self.results.get("Pure RL (Monthly)", {}).get(num_goals)
                sent_result = self.results.get("Sentiment RL (Monthly)", {}).get(num_goals)
            
            dp_annual_result = self.results.get("DP (Annual)", {}).get(num_goals)
            pure_annual_result = self.results.get("Pure RL (Annual)", {}).get(num_goals)
            
            if dp_result and pure_result:
                print(f"\n{num_goals} Goals:")
                print(f"  DP (Monthly):     ${dp_result.mean_final_wealth/1000:.0f}k ± {dp_result.std_final_wealth/1000:.0f}k")
                print(f"  Pure RL (Month):  ${pure_result.mean_final_wealth/1000:.0f}k ± {pure_result.std_final_wealth/1000:.0f}k")
                if sent_result:
                    print(f"  Sentiment RL:     ${sent_result.mean_final_wealth/1000:.0f}k ± {sent_result.std_final_wealth/1000:.0f}k")
                
                if dp_annual_result:
                    print(f"  DP (Annual):      ${dp_annual_result.mean_final_wealth/1000:.0f}k ± {dp_annual_result.std_final_wealth/1000:.0f}k")
                    
                if pure_annual_result:
                    print(f"  Pure RL (Annual): ${pure_annual_result.mean_final_wealth/1000:.0f}k ± {pure_annual_result.std_final_wealth/1000:.0f}k")
                
                # Calculate wealth preservation vs utility maximization trade-offs
                if sent_result and pure_result.mean_final_wealth > 0:
                    sent_wealth_gain = ((sent_result.mean_final_wealth - pure_result.mean_final_wealth) 
                                      / pure_result.mean_final_wealth * 100)
                    print(f"  → Sentiment RL wealth advantage: {sent_wealth_gain:+.1f}% vs Pure RL")
                    
                if sent_result and dp_result.mean_final_wealth > 0 and sent_result.mean_final_wealth > 0:
                    dp_sent_ratio = sent_result.mean_final_wealth / dp_result.mean_final_wealth
                    print(f"  → Sentiment RL achieves {dp_sent_ratio:.1%} of DP wealth")
        
        print("=" * 80)

    def run(self):
        """
        Run complete evaluation pipeline based on baseline_mode.

        Modes:
        - 'annual_stable': Pure RL (annual) vs DP stable, Sentiment RL (monthly) vs DP VIX
        - 'monthly_vix': All methods at monthly granularity, DP uses stable market
        """
        # Step 1: Load parameters and run cross-mode consistency validation
        self.load_pretrained_parameters()
        validation_passed = self.validate_cross_mode_consistency()
        
        if not validation_passed:
            logger.warning("Cross-mode validation failed, but continuing evaluation...")
            logger.warning("Results may contain inconsistencies across baseline modes.")
        
        # Step 2: Run the specific baseline mode evaluation
        if self.config.baseline_mode == "annual_stable":
            self._run_annual_stable_comparison()
        elif self.config.baseline_mode == "monthly_vix":
            self._run_monthly_vix_comparison()
        else:
            raise ValueError(f"Unsupported baseline_mode: {self.config.baseline_mode}. Only 'monthly_vix' and 'annual_stable' are supported.")

    def _run_annual_stable_comparison(self):
        """
        Compare with DIFFERENT baselines (each method vs its optimal environment):
        - Pure RL (annual, 16 steps) vs DP in STABLE market (no VIX)

        """
        start_time = time.time()

        logger.info("\n" + "=" * 70)
        logger.info("ANNUAL STABLE COMPARISON MODE")
        logger.info("  Pure RL (annual) vs DP in stable market")

        logger.info("=" * 70)

        # Step 1: Generate MONTHLY shocks FIRST (base truth for all)
        # Parameters already loaded in run() method
        # Shape: (num_simulations, 192)
        self._generate_shared_random_state()
        logger.info(f"Generated monthly market shocks: {self.market_shocks.shape}")

        # Step 2: DERIVE annual shocks from monthly shocks for consistency
        # Z_annual = sum(Z_monthly over 12 months) / sqrt(12)
        logger.info("Deriving annual shocks from monthly (for consistent market)...")
        annual_shocks = np.zeros((self.config.num_simulations, self.config.years_horizon))
        for year in range(self.config.years_horizon):
            start_month = year * self.config.months_per_year
            end_month = (year + 1) * self.config.months_per_year
            # Aggregate monthly shocks to annual (preserves N(0,1) distribution)
            annual_shocks[:, year] = (
                self.market_shocks[:, start_month:end_month].sum(axis=1)
                / np.sqrt(self.config.months_per_year)
            )
        logger.info(f"Derived annual shocks: {annual_shocks.shape}")

        # Step 3: INLINE ANNUAL DP EVALUATION (Stable Market)
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING ANNUAL DP EVALUATION (Stable Market)")
        logger.info("=" * 60)
        
        # Solve DP at annual granularity
        self.solve_dp_annual()  # Use dedicated annual solver
        
        # Simulate DP with annual shocks - create inline implementation
        logger.info("Simulating DP at annual granularity...")
        self._simulate_dp_with_annual_shocks(annual_shocks, result_key="DP (Annual Stable)")

        # Step 4: INLINE ANNUAL PURE RL EVALUATION (Stable Market)
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING ANNUAL PURE RL EVALUATION (Stable Market)")
        logger.info("=" * 60)
        
        # Train Pure RL at annual granularity - create inline implementation  
        logger.info("Training Pure RL at annual granularity...")
        self._train_pure_rl_with_annual_env()
        
        # Simulate Pure RL with annual shocks - create inline implementation
        logger.info("Simulating Pure RL with annual shocks...")
        self._simulate_pure_rl_with_annual_shocks(annual_shocks, result_key="Pure RL (Annual)")

        # Step 5: Solve DP for Sentiment RL comparison (yearly decisions)
        #logger.info("\n" + "=" * 60)
        #logger.info("RUNNING SENTIMENT RL EVALUATION (VIX Market)")
        #logger.info("=" * 60)
        #self.solve_dp()

        # Step 6: Train Sentiment RL agents (monthly, with VIX)
        #self.train_sentiment_agents()

        # Step 7: Simulate DP baseline for Sentiment RL comparison
        # NOTE: DP uses STABLE market (no VIX) as the theoretical optimal baseline
        # Sentiment RL faces VIX-adjusted market but compares against stable DP
        # This measures whether VIX information helps beat stable-world optimal
        #self.simulate_dp_baseline(use_vix_market=False)
        #self.results["DP (Monthly Stable)"] = self.results.pop("DP")

        # Step 8: Simulate Sentiment RL with VIX
        #self.simulate_sentiment_rl()

        # Step 9: Generate separate figure for annual_stable mode
        self.generate_annual_stable_figure()

        # Step 10: Save results
        self.save_results()

        # Step 11: Print summary
        self.print_annual_stable_summary()

        total_time = time.time() - start_time
        logger.info(f"\nTotal evaluation time: {total_time:.1f}s ({total_time/60:.1f} min)")

    def _run_monthly_vix_comparison(self):
        """
        Compare all methods at MONTHLY granularity with SAME baseline:
        - DP (monthly simulation, STABLE market - no VIX adjustments)
        - Pure RL (monthly, no VIX in state, stable market)
        - Pure RL (VIX Market): (monthly, no VIX in state, VIX-adjusted market)
        - Sentiment RL (Stable): (monthly, VIX in state, stable market)
        - Sentiment RL (VIX Market): (monthly, VIX in state, VIX-adjusted market)

        All use same 192 monthly time steps and same market shocks.
        DP baseline uses stable market (no VIX) for fair comparison.

        This design isolates two effects:
        1. VIX MARKET effect: Pure RL vs Pure RL (VIX Market)
        2. VIX STATE effect: Pure RL vs Sentiment RL (Stable)
        3. Combined effect: Pure RL vs Sentiment RL (VIX Market)
        """
        start_time = time.time()

        dp_market_type = "VIX-ADJUSTED" if self.config.dp_vix_market else "STABLE (no VIX)"

        logger.info("\n" + "=" * 70)
        logger.info("MONTHLY VIX COMPARISON MODE")
        logger.info("  All methods at 192 monthly steps")
        logger.info(f"  DP uses {dp_market_type} market")
        logger.info("  Pure RL (VIX Market) isolates VIX state vs market effect")
        logger.info("=" * 70)

        # Step 1: Load pre-trained parameters
        self.load_pretrained_parameters()

        # Step 2: Generate shared random state (same for all methods)
        self._generate_shared_random_state()

        # Step 3: Solve DP (optimal baseline) - use monthly solver
        self.solve_dp_monthly()
        
        # Step 3b: Solve DP Annual (for annual comparison)
        self.solve_dp_annual()

        # Step 4: Train Pure RL agents (monthly, stable market) 
        self.train_pure_rl_agents()

        # Step 5: Train Sentiment RL agents (monthly, VIX market)
        self.train_sentiment_agents()

        # Step 6: Add Annual Methods for Comparison
        # Generate annual shocks from monthly for consistency
        logger.info("Deriving annual shocks from monthly (for consistent market)...")
        annual_shocks = np.zeros((self.config.num_simulations, self.config.years_horizon))
        for year in range(self.config.years_horizon):
            start_month = year * self.config.months_per_year
            end_month = (year + 1) * self.config.months_per_year
            annual_shocks[:, year] = (
                self.market_shocks[:, start_month:end_month].sum(axis=1)
                / np.sqrt(self.config.months_per_year)
            )

        # Step 6: Simulate DP Annual (for comparison) 
        logger.info("\nRunning DP Annual simulation...")
        self._simulate_dp_with_annual_shocks(annual_shocks, result_key="DP (Annual)")

        # Step 7: Train and simulate Pure RL Annual (for comparison)
        logger.info("\nTraining Pure RL Annual...")
        self._train_pure_rl_with_annual_env()
        self._simulate_pure_rl_with_annual_shocks(annual_shocks, result_key="Pure RL (Annual)")

        # Step 8: Simulate DP Monthly (stable market baseline)
        self.simulate_dp_baseline(use_vix_market=False)

        # Step 9: Simulate Pure RL Monthly (stable market)
        self.simulate_pure_rl()

        # Step 10: Simulate Sentiment RL Monthly (VIX market) 
        self.simulate_sentiment_rl(use_vix_market=False, result_key="Sentiment RL (Monthly)")

        # Step 11: Generate comparison figure
        self.generate_figure1()

        # Step 12: Save results
        self.save_results()

        # Step 13: Print summary
        self.print_summary()

        total_time = time.time() - start_time
        logger.info(f"\nTotal evaluation time: {total_time:.1f}s ({total_time/60:.1f} min)")






    def validate_cross_mode_consistency(self):
        """
        Validate that same methods produce consistent results across baseline modes.
        
        Checks for consistency in:
        1. VIX model implementation between MRJD and regime-switching
        2. DP algorithm results with same parameters
        3. Sentiment RL with same architecture across different market conditions
        
        Returns True if validation passes, False if inconsistencies found.
        """
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-MODE CONSISTENCY VALIDATION")
        logger.info("=" * 70)
        
        validation_passed = True
        
        # Check 1: VIX model consistency
        logger.info("Checking VIX model implementation consistency...")
        
        # Create small test scenario with same parameters
        test_shocks = np.random.normal(0, 1, size=(10, 12))  # 10 sims, 12 months
        test_seed = 12345
        
        # Test MRJD model
        try:
            from src.models.vix_market_model import create_vix_model
            
            mrjd_model = create_vix_model(
                model_type='mrjd',
                market_shocks=test_shocks,
                seed=test_seed,
                kappa=self.config.vix_kappa,
                theta=self.config.vix_theta,
                sigma_v=self.config.vix_sigma_v
            )
            
            regime_model = create_vix_model(
                model_type='regime_switching',
                market_shocks=test_shocks,
                seed=test_seed,
                kappa=self.config.vix_kappa,
                theta=self.config.vix_theta,
                sigma_v=self.config.vix_sigma_v
            )
            
            # Both should initialize without errors
            mrjd_model.reset(episode_idx=0, sim_idx=0)
            regime_model.reset(episode_idx=0, sim_idx=0)
            
            # Step through and check for reasonable VIX values
            mrjd_vix_values = []
            regime_vix_values = []
            
            for step in range(5):
                mrjd_model.step_vix(dt=1/12)
                regime_model.step_vix(dt=1/12)
                
                mrjd_vix = mrjd_model.get_vix_average()
                regime_vix = regime_model.get_vix_average()
                
                mrjd_vix_values.append(mrjd_vix)
                regime_vix_values.append(regime_vix)
                
                # Check for reasonable VIX ranges (5-80 typical)
                if not (5 <= mrjd_vix <= 80):
                    logger.warning(f"MRJD VIX outside normal range at step {step}: {mrjd_vix:.2f}")
                    validation_passed = False
                    
                if not (5 <= regime_vix <= 80):
                    logger.warning(f"Regime VIX outside normal range at step {step}: {regime_vix:.2f}")
                    validation_passed = False
            
            logger.info(f"  MRJD VIX range: {min(mrjd_vix_values):.2f} - {max(mrjd_vix_values):.2f}")
            logger.info(f"  Regime VIX range: {min(regime_vix_values):.2f} - {max(regime_vix_values):.2f}")
            logger.info("  ✅ VIX models initialize and step correctly")
            
        except Exception as e:
            logger.error(f"  ❌ VIX model error: {e}")
            validation_passed = False
        
        # Check 2: Portfolio parameter consistency
        logger.info("\nChecking portfolio parameter consistency...")
        
        if hasattr(self, 'portfolio_means_annual') and hasattr(self, 'portfolio_stds_annual'):
            # Check for reasonable ranges
            mean_range = (self.portfolio_means_annual.min(), self.portfolio_means_annual.max())
            std_range = (self.portfolio_stds_annual.min(), self.portfolio_stds_annual.max())
            
            # Reasonable annual return: 2% to 12%
            if not (0.02 <= mean_range[0] <= mean_range[1] <= 0.15):
                logger.warning(f"  Portfolio returns outside reasonable range: {mean_range}")
                validation_passed = False
            
            # Reasonable annual volatility: 1% to 25%
            if not (0.01 <= std_range[0] <= std_range[1] <= 0.30):
                logger.warning(f"  Portfolio volatilities outside reasonable range: {std_range}")
                validation_passed = False
                
            logger.info(f"  Portfolio returns: {mean_range[0]:.2%} - {mean_range[1]:.2%}")
            logger.info(f"  Portfolio volatilities: {std_range[0]:.2%} - {std_range[1]:.2%}")
            logger.info("  ✅ Portfolio parameters within reasonable ranges")
        else:
            logger.warning("  ⚠️ Portfolio parameters not loaded")
            
        # Check 3: Beta/Delta parameter consistency
        logger.info("\nChecking beta/delta parameter consistency...")
        
        if hasattr(self, 'portfolio_betas') and hasattr(self, 'portfolio_deltas'):
            beta_range = (self.portfolio_betas.min(), self.portfolio_betas.max())
            delta_range = (self.portfolio_deltas.min(), self.portfolio_deltas.max())
            
            # Beta should be small, typically -0.1 to +0.1
            if not (-0.2 <= beta_range[0] <= beta_range[1] <= 0.2):
                logger.warning(f"  Beta values outside typical range: {beta_range}")
                validation_passed = False
                
            # Delta should be small, typically -0.05 to +0.05
            if not (-0.1 <= delta_range[0] <= delta_range[1] <= 0.1):
                logger.warning(f"  Delta values outside typical range: {delta_range}")
                validation_passed = False
                
            logger.info(f"  Portfolio betas: {beta_range[0]:.4f} - {beta_range[1]:.4f}")
            logger.info(f"  Portfolio deltas: {delta_range[0]:.4f} - {delta_range[1]:.4f}")
            logger.info("  ✅ Beta/delta parameters within reasonable ranges")
        else:
            logger.warning("  ⚠️ Beta/delta parameters not loaded")
        
        # Check 4: Environment creation consistency
        logger.info("\nChecking environment creation consistency...")
        
        try:
            # Test environment creation with both VIX model types
            test_env_mrjd = make_gbwm_env_monthly(
                num_goals=4,
                use_sentiment=True,
                vix_model_type='mrjd',
                vix_params=self.vix_params if hasattr(self, 'vix_params') else None
            )
            
            test_env_regime = make_gbwm_env_monthly(
                num_goals=4,
                use_sentiment=True,
                vix_model_type='regime_switching',
                vix_params=self.vix_params if hasattr(self, 'vix_params') else None
            )
            
            # Test reset and step
            obs1 = test_env_mrjd.reset(seed=12345)
            obs2 = test_env_regime.reset(seed=12345)
            
            # Both should return 5D observation spaces for sentiment
            if len(obs1[0]) != 5:
                logger.warning(f"  MRJD env observation dimension incorrect: {len(obs1[0])}")
                validation_passed = False
                
            if len(obs2[0]) != 5:
                logger.warning(f"  Regime env observation dimension incorrect: {len(obs2[0])}")
                validation_passed = False
            
            # Test step
            action = [0, 7]  # skip goal, mid-risk portfolio
            _, _, done1, _, info1 = test_env_mrjd.step(action)
            _, _, done2, _, info2 = test_env_regime.step(action)
            
            # Both should have VIX info
            if 'vix' not in info1:
                logger.warning("  MRJD env missing VIX info")
                validation_passed = False
                
            if 'vix' not in info2:
                logger.warning("  Regime env missing VIX info")
                validation_passed = False
            
            test_env_mrjd.close()
            test_env_regime.close()
            
            logger.info("  ✅ Environment creation works for both VIX model types")
            
        except Exception as e:
            logger.error(f"  ❌ Environment creation error: {e}")
            validation_passed = False
        
        # Check 5: Random seed consistency
        logger.info("\nChecking random seed consistency...")
        
        if hasattr(self, 'shared_seeds') and self.shared_seeds is not None:
            # Verify seeds are unique
            unique_seeds = np.unique(self.shared_seeds)
            if len(unique_seeds) < len(self.shared_seeds) * 0.95:
                logger.warning(f"  Too many duplicate seeds: {len(unique_seeds)}/{len(self.shared_seeds)}")
                validation_passed = False
            else:
                logger.info(f"  Seed uniqueness: {len(unique_seeds)}/{len(self.shared_seeds)} unique")
                logger.info("  ✅ Random seeds are appropriately diverse")
        else:
            logger.warning("  ⚠️ Shared seeds not generated yet")
        
        # Final validation summary
        logger.info("\n" + "=" * 70)
        if validation_passed:
            logger.info("✅ CROSS-MODE CONSISTENCY VALIDATION PASSED")
            logger.info("All baseline modes should produce consistent results")
        else:
            logger.warning("❌ CROSS-MODE CONSISTENCY VALIDATION FAILED")
            logger.warning("Some inconsistencies detected - results may vary across modes")
        logger.info("=" * 70)
        
        return validation_passed

    # =============================================================================
    # MODEL LOADING METHODS FOR SKIP TRAINING FUNCTIONALITY
    # =============================================================================

    def _load_existing_sentiment_models(self, architecture_type=None):
        """
        Load existing trained Sentiment RL models to resume from simulation only.
        
        Args:
            architecture_type: 'simple', 'advanced', or None (for legacy compatibility)
            
        Returns:
            bool: True if all models loaded successfully, False otherwise
        """
        try:
            logger.info("Checking for existing Sentiment RL models...")
            
            # Determine model directory based on architecture type
            if architecture_type == 'simple':
                model_base_dir = self.output_dir / "models" / "sentiment_rl_simple"
                agent_container = "sentiment_agents_simple"
            elif architecture_type == 'advanced':
                model_base_dir = self.output_dir / "models" / "sentiment_rl_advanced"
                agent_container = "sentiment_agents_advanced"
            else:
                model_base_dir = self.output_dir / "models" / "sentiment_rl"
                agent_container = "sentiment_agents"
            
            # Initialize agent container if needed
            if not hasattr(self, agent_container):
                setattr(self, agent_container, {})
            agent_dict = getattr(self, agent_container)
            
            loaded_count = 0
            for num_goals in self.config.goal_counts:
                model_path = model_base_dir / f"goals_{num_goals}" / "model.pth"
                
                if model_path.exists():
                    logger.info(f"  Loading Sentiment RL model for {num_goals} goals: {model_path}")
                    
                    # Create environment to instantiate agent (same as training)
                    env = make_gbwm_env_monthly(
                        num_goals=num_goals,
                        use_sentiment=True,
                        vix_params=self.vix_params,
                        portfolio_betas=self.portfolio_betas,
                        portfolio_deltas=self.portfolio_deltas,
                        volatility_method=self.config.volatility_method,
                        vix_model_type=self.config.vix_model_type,
                        use_real_ef=self.config.use_real_ef,
                        portfolio_means=self.portfolio_means_annual,
                        portfolio_stds=self.portfolio_stds_annual
                    )
                    
                    # Create training config (needed for agent instantiation)
                    training_config = TrainingConfig(
                        batch_size=self.config.batch_size,
                        learning_rate=self.config.learning_rate,
                        n_neurons=self.config.hidden_dim,
                        time_horizon=self.config.time_horizon,
                        device=self.config.device
                    )
                    
                    # Create agent with same architecture as training
                    agent = SentimentAwarePPOAgent(
                        env=env,
                        config=training_config,
                        policy_type=self.config.policy_type,
                        value_type=self.config.value_type,
                        encoder_type=self.config.encoder_type,
                        device=self.config.device,
                        sentiment_enabled=True
                    )
                    
                    # Load model weights
                    agent.load(str(model_path))
                    agent_dict[num_goals] = agent
                    
                    env.close()
                    loaded_count += 1
                    logger.info(f"    ✓ Loaded successfully")
                else:
                    logger.warning(f"  Model not found for {num_goals} goals: {model_path}")
                    return False
            
            logger.info(f"Successfully loaded {loaded_count}/{len(self.config.goal_counts)} Sentiment RL models")
            return loaded_count == len(self.config.goal_counts)
            
        except Exception as e:
            logger.error(f"Failed to load existing Sentiment RL models: {e}")
            return False

    def _load_existing_pure_rl_models(self):
        """
        Load existing trained Pure RL models to resume from simulation only.
        
        Returns:
            bool: True if all models loaded successfully, False otherwise
        """
        try:
            logger.info("Checking for existing Pure RL models...")
            
            model_base_dir = self.output_dir / "models" / "pure_rl_monthly"
            loaded_count = 0
            
            for num_goals in self.config.goal_counts:
                model_path = model_base_dir / f"goals_{num_goals}" / "model.pth"
                
                if model_path.exists():
                    logger.info(f"  Loading Pure RL model for {num_goals} goals: {model_path}")
                    
                    # Create environment to instantiate agent (same as training)
                    env = make_gbwm_env_monthly(
                        num_goals=num_goals,
                        use_sentiment=False,  # NO VIX - key difference!
                        vix_model_type=self.config.vix_model_type,
                        vix_params=None,
                        portfolio_betas=None,
                        portfolio_deltas=None,
                        use_real_ef=self.config.use_real_ef,
                        portfolio_means=self.portfolio_means_annual,
                        portfolio_stds=self.portfolio_stds_annual
                    )
                    
                    # Create training config (needed for agent instantiation)
                    training_config = TrainingConfig(
                        batch_size=self.config.batch_size,
                        learning_rate=self.config.learning_rate,
                        n_neurons=self.config.hidden_dim,
                        time_horizon=self.config.time_horizon,
                        device=self.config.device
                    )
                    
                    # Create Pure RL agent (no sentiment features)
                    pure_rl_encoder = self.config.pure_rl_encoder_type
                    if pure_rl_encoder in ['feature', 'attention', 'adaptive']:
                        pure_rl_encoder = 'simple'
                    
                    agent = SentimentAwarePPOAgent(
                        env=env,
                        config=training_config,
                        policy_type=self.config.pure_rl_policy_type,
                        value_type=self.config.pure_rl_value_type,
                        encoder_type=pure_rl_encoder,
                        device=self.config.device,
                        sentiment_enabled=False
                    )
                    
                    # Load model weights
                    agent.load(str(model_path))
                    self.pure_rl_agents[num_goals] = agent
                    
                    env.close()
                    loaded_count += 1
                    logger.info(f"    ✓ Loaded successfully")
                else:
                    logger.warning(f"  Model not found for {num_goals} goals: {model_path}")
                    return False
            
            logger.info(f"Successfully loaded {loaded_count}/{len(self.config.goal_counts)} Pure RL models")
            return loaded_count == len(self.config.goal_counts)
            
        except Exception as e:
            logger.error(f"Failed to load existing Pure RL models: {e}")
            return False

    def _load_existing_annual_pure_rl_models(self):
        """
        Load existing trained Pure RL Annual models to resume from simulation only.
        
        Returns:
            bool: True if all models loaded successfully, False otherwise
        """
        try:
            logger.info("Checking for existing Pure RL Annual models...")
            
            model_base_dir = self.output_dir / "pure_rl_annual_models"
            loaded_count = 0
            
            for num_goals in self.config.goal_counts:
                model_path = model_base_dir / f"goals_{num_goals}" / "model.pth"
                
                if model_path.exists():
                    logger.info(f"  Loading Pure RL Annual model for {num_goals} goals: {model_path}")
                    
                    # Import annual environment
                    from src.environment.gbwm_env import GBWMEnvironment
                    from config.environment_config import EnvironmentConfig, GoalConfig
                    
                    initial_wealth = get_initial_wealth(num_goals)
                    goal_years = get_goal_years(num_goals)
                    
                    # Create annual environment (same as training)
                    goal_config = GoalConfig(goal_years=goal_years)
                    env_config = EnvironmentConfig(
                        initial_wealth=initial_wealth,
                        time_horizon=self.config.years_horizon,
                        goal_config=goal_config
                    )
                    env_config.portfolio_config.mean_returns = self.portfolio_means_annual
                    env_config.portfolio_config.return_stds = self.portfolio_stds_annual
                    env = GBWMEnvironment(env_config)
                    
                    # Create training config
                    training_config = TrainingConfig(
                        batch_size=self.config.batch_size,
                        learning_rate=self.config.learning_rate,
                        n_neurons=64,  # Same as training
                        time_horizon=self.config.years_horizon,
                        device=self.config.device
                    )
                    
                    # Create agent
                    agent = SentimentAwarePPOAgent(
                        env=env,
                        config=training_config,
                        policy_type='standard',
                        value_type='standard',
                        encoder_type='simple',
                        device=self.config.device,
                        sentiment_enabled=False
                    )
                    
                    # Load model weights
                    agent.load(str(model_path))
                    
                    # Initialize the container if it doesn't exist
                    if not hasattr(self, 'pure_rl_agents_annual'):
                        self.pure_rl_agents_annual = {}
                    
                    self.pure_rl_agents_annual[num_goals] = agent
                    
                    env.close()
                    loaded_count += 1
                    logger.info(f"    ✓ Loaded successfully")
                else:
                    logger.warning(f"  Model not found for {num_goals} goals: {model_path}")
                    return False
            
            logger.info(f"Successfully loaded {loaded_count}/{len(self.config.goal_counts)} Pure RL Annual models")
            return loaded_count == len(self.config.goal_counts)
            
        except Exception as e:
            logger.error(f"Failed to load existing Pure RL Annual models: {e}")
            return False

    # =============================================================================
    # SIMULATION RESULTS CACHING METHODS
    # =============================================================================
    
    def _get_simulation_cache_key(self, method_name: str, num_goals: int) -> str:
        """Generate a unique cache key for simulation results"""
        import hashlib
        
        # Create config hash to detect parameter changes
        config_dict = {
            'baseline_mode': self.config.baseline_mode,
            'num_simulations': self.config.num_simulations,
            'volatility_method': self.config.volatility_method,
            'vix_model_type': self.config.vix_model_type,
            'use_real_ef': self.config.use_real_ef,
            'time_horizon': self.config.time_horizon,
            'num_goals': num_goals,
            'method_name': method_name
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{method_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_{num_goals}goals_{self.config.num_simulations}sims_{config_hash}"
    
    def _get_simulation_cache_path(self, method_name: str, num_goals: int) -> Path:
        """Get the file path for cached simulation results"""
        cache_key = self._get_simulation_cache_key(method_name, num_goals)
        cache_dir = self.output_dir / "results" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{cache_key}.json"
    
    def _save_simulation_result(self, method_name: str, num_goals: int, result: SimulationResult):
        """Save simulation result to cache"""
        try:
            cache_path = self._get_simulation_cache_path(method_name, num_goals)
            
            # Convert SimulationResult to dict for JSON serialization
            result_dict = asdict(result)
            result_dict['_cache_timestamp'] = datetime.now().isoformat()
            result_dict['_config_hash'] = self._get_simulation_cache_key(method_name, num_goals)
            
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                """Recursively convert numpy types to Python native types"""
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            result_dict = convert_numpy_types(result_dict)
            
            with open(cache_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            logger.info(f"  Cached {method_name} results for {num_goals} goals: {cache_path}")
            
        except Exception as e:
            logger.warning(f"Failed to cache simulation results: {e}")
    
    def _load_simulation_result(self, method_name: str, num_goals: int) -> Optional[SimulationResult]:
        """Load simulation result from cache if it exists and is valid"""
        try:
            cache_path = self._get_simulation_cache_path(method_name, num_goals)
            
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'r') as f:
                result_dict = json.load(f)
            
            # Verify cache key matches current config
            expected_key = self._get_simulation_cache_key(method_name, num_goals)
            if result_dict.get('_config_hash') != expected_key:
                logger.warning(f"Cache key mismatch for {method_name} {num_goals} goals - config changed, invalidating cache")
                return None
            
            # Remove metadata before creating SimulationResult
            result_dict.pop('_cache_timestamp', None)
            result_dict.pop('_config_hash', None)
            
            # Create SimulationResult from cached data
            result = SimulationResult(**result_dict)
            
            logger.info(f"  Loaded cached {method_name} results for {num_goals} goals: {cache_path}")
            return result
            
        except Exception as e:
            logger.warning(f"Failed to load cached simulation results for {method_name} {num_goals} goals: {e}")
            return None
    
    def _should_skip_simulation(self, method_name: str, num_goals: int) -> bool:
        """Check if simulation should be skipped due to cached results"""
        if not self.config.skip_training:  # Reuse skip_training flag for simulations
            return False
            
        cached_result = self._load_simulation_result(method_name, num_goals)
        if cached_result:
            # Store in results dictionary
            if method_name not in self.results:
                self.results[method_name] = {}
            self.results[method_name][num_goals] = cached_result
            return True
        
        return False
    
    def _clear_simulation_cache(self):
        """Clear all cached simulation results"""
        try:
            cache_dir = self.output_dir / "results" / "cache"
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared simulation cache directory: {cache_dir}")
            else:
                logger.info("No simulation cache directory found")
        except Exception as e:
            logger.error(f"Failed to clear simulation cache: {e}")
    
    def _show_cache_info(self):
        """Display information about cached simulation results"""
        try:
            cache_dir = self.output_dir / "results" / "cache"
            if not cache_dir.exists():
                print("No simulation cache directory found")
                return
                
            cache_files = list(cache_dir.glob("*.json"))
            if not cache_files:
                print("No cached simulation results found")
                return
                
            print(f"\nSimulation Cache Information ({len(cache_files)} files):")
            print("=" * 70)
            
            # Group by method and goal count
            cache_data = {}
            total_size = 0
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    
                    method = data.get('method_name', 'Unknown')
                    goals = data.get('num_goals', 0)
                    sims = data.get('num_simulations', 0)
                    timestamp = data.get('_cache_timestamp', 'Unknown')
                    size = cache_file.stat().st_size
                    total_size += size
                    
                    if method not in cache_data:
                        cache_data[method] = []
                    
                    cache_data[method].append({
                        'goals': goals,
                        'simulations': sims,
                        'timestamp': timestamp,
                        'size': size,
                        'file': cache_file.name
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to read cache file {cache_file}: {e}")
            
            # Display organized cache info
            for method, entries in sorted(cache_data.items()):
                print(f"\n{method}:")
                for entry in sorted(entries, key=lambda x: x['goals']):
                    size_kb = entry['size'] / 1024
                    timestamp = entry['timestamp'][:19] if entry['timestamp'] != 'Unknown' else 'Unknown'
                    print(f"  {entry['goals']} goals: {entry['simulations']:,} sims, "
                          f"{size_kb:.1f}KB, {timestamp} ({entry['file']})")
            
            total_size_mb = total_size / (1024 * 1024)
            print(f"\nTotal cache size: {total_size_mb:.2f} MB")
            print("\nTo clear cache, use: --clear_cache")
            
        except Exception as e:
            logger.error(f"Failed to show cache info: {e}")

    # =============================================================================
    # INLINE IMPLEMENTATIONS FOR MISSING METHODS
    # =============================================================================

    def _simulate_dp_with_annual_shocks(self, annual_shocks: np.ndarray, result_key: str = "DP (Annual)"):
        """Simulate DP with annual time steps and shocks"""
        dt = 1.0  # Annual time step
        
        for num_goals in self.config.goal_counts:
            # Check cache first - skip simulation if cached results exist
            if self._should_skip_simulation(result_key, num_goals):
                logger.info(f"Skipping {result_key} simulation for {num_goals} goals (cached results found)")
                continue
            # Use annual DP policies for annual simulation
            if num_goals not in self.dp_policies_annual:
                continue
            
            dp = self.dp_policies_annual[num_goals]
            initial_wealth = get_initial_wealth(num_goals)
            goal_years = get_goal_years(num_goals)
            
            logger.info(f"\nSimulating DP (Annual) for {num_goals} goals...")
            
            rewards = []
            goal_successes = []
            final_wealths = []
            
            # Debug first few simulations to understand goal taking pattern
            debug_sim_count = 3 if num_goals <= 4 else 1
            
            for sim_idx in tqdm(range(self.config.num_simulations), desc=f"DP Annual {num_goals}g", leave=False):
                if sim_idx % 10000 == 0 and sim_idx > 0:
                    logger.info(f"  DP Annual {num_goals} goals: {sim_idx:,}/{self.config.num_simulations:,}")
                
                sim_annual_shocks = annual_shocks[sim_idx]
                wealth = initial_wealth
                total_reward = 0
                goals_taken = []
                
                debug_this_sim = sim_idx < debug_sim_count
                if debug_this_sim:
                    logger.info(f"    DEBUG sim {sim_idx}: Initial wealth=${wealth:,.0f}, Goal years: {goal_years}")
                
                for t in range(self.config.years_horizon):
                    # Get optimal strategy from Multi-Goal DP (includes BOTH goal and portfolio decisions)
                    goal_action, portfolio_idx, mu, sigma = dp.get_optimal_strategy(wealth, t)
                    
                    # Check if goal year and apply optimal goal decision  
                    # DP time t directly corresponds to goal_years (e.g., t=4 for 4th year goal)
                    is_goal_year = t in dp.config.goal_years
                    effective_year = t
                    
                    # Handle terminal goal: goal at year T is processed at t=T-1
                    if not is_goal_year and t == self.config.years_horizon - 1 and self.config.years_horizon in dp.config.goal_years:
                        is_goal_year = True
                        effective_year = self.config.years_horizon
                    
                    if is_goal_year:
                        goal_cost = dp.config.get_goal_cost(effective_year)
                        goal_utility = dp.config.get_goal_utility(effective_year)
                        
                        if debug_this_sim:
                            logger.info(f"    Year {effective_year} (t={t}): wealth=${wealth:,.0f}, cost=${goal_cost:,.0f}, action={goal_action}, utility={goal_utility}")
                        
                        # Use DP's optimal goal decision (not heuristic "if affordable")
                        if goal_action == 1 and wealth >= goal_cost:
                            total_reward += goal_utility
                            wealth -= goal_cost
                            goals_taken.append(effective_year)
                            if debug_this_sim:
                                logger.info(f"    GOAL TAKEN! New wealth=${wealth:,.0f}, total_reward={total_reward}")
                        elif debug_this_sim:
                            reason = "insufficient wealth" if wealth < goal_cost else "DP says skip"
                            logger.info(f"    Goal skipped: {reason}")
                    
                    # Wealth evolution with market shock
                    if t < self.config.years_horizon - 1:
                        shock = sim_annual_shocks[t]
                        old_wealth = wealth
                        # Use consistent annual GBM formula (dt = 1.0 for annual steps)
                        drift = (mu - 0.5 * sigma**2) * dt
                        diffusion = sigma * np.sqrt(dt) * shock  # sqrt(1.0) = 1.0, so this equals sigma * shock
                        wealth = wealth * np.exp(drift + diffusion)
                        wealth = max(0, wealth)
                        
                        if debug_this_sim and t < 3:  # Only log first few years
                            logger.info(f"    Portfolio evolution Y{t+1}→{t+2}: ${old_wealth:,.0f} → ${wealth:,.0f} (μ={mu:.3f}, σ={sigma:.3f})")
                
                if debug_this_sim:
                    logger.info(f"    Final: reward={total_reward}, goals_taken={goals_taken}, final_wealth=${wealth:,.0f}")
                
                rewards.append(total_reward)
                goal_successes.append(len(goals_taken) / len(goal_years))
                final_wealths.append(wealth)
            
            result = SimulationResult(
                method_name=result_key,
                num_goals=num_goals,
                mean_reward=np.mean(rewards),
                std_reward=np.std(rewards),
                efficiency=1.0,  # DP is baseline
                num_simulations=self.config.num_simulations,
                goal_success_rate=np.mean(goal_successes),
                mean_final_wealth=np.mean(final_wealths),
                std_final_wealth=np.std(final_wealths)
            )
            
            self.results[result_key][num_goals] = result
            
            # Cache the simulation result
            self._save_simulation_result(result_key, num_goals, result)
            
            logger.info(f"  DP Annual {num_goals} goals: reward={np.mean(rewards):.2f}, success={np.mean(goal_successes):.3f}")

    def _train_pure_rl_with_annual_env(self):
        """Train Pure RL agents at annual granularity"""
        from src.environment.gbwm_env import GBWMEnvironment  # Annual environment
        from config.environment_config import EnvironmentConfig, GoalConfig
        
        # Skip training if --skip_training flag is set and models exist
        if self.config.skip_training:
            if self._load_existing_annual_pure_rl_models():
                logger.info("Skipping Pure RL Annual training (--skip_training flag and models found)")
                return
            else:
                logger.warning("--skip_training set but no existing Pure RL Annual models found. Proceeding with training.")
        
        logger.info("Training Pure RL at annual granularity...")
        
        self.pure_rl_agents_annual = {}
        
        for num_goals in self.config.goal_counts:
            initial_wealth = get_initial_wealth(num_goals)
            goal_years = get_goal_years(num_goals)
            logger.info(f"\nTraining Pure RL (Annual) for {num_goals} goals (W0=${initial_wealth:,.0f})...")
            
            # Create goal config for the specific number of goals
            goal_config = GoalConfig(goal_years=goal_years)
            
            # Create annual environment (16 time steps, no sentiment)
            # CRITICAL: Use same portfolio parameters as DP simulation for consistency
            env_config = EnvironmentConfig(
                initial_wealth=initial_wealth,
                time_horizon=self.config.years_horizon,  # 16 years
                goal_config=goal_config
            )
            
            # Set portfolio parameters to match DP simulation
            env_config.portfolio_config.mean_returns = self.portfolio_means_annual
            env_config.portfolio_config.return_stds = self.portfolio_stds_annual
            
            env = GBWMEnvironment(env_config)
            
            # Training config
            training_config = TrainingConfig(
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                n_neurons=64,  # Smaller for annual RL
                ppo_epochs=4,
                mini_batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=0.5,
                entropy_coeff=0.01,
                max_grad_norm=0.5,
                time_horizon=self.config.years_horizon,  # 16 annual steps
                device=self.config.device
            )
            
            # Create Pure RL agent for annual environment
            agent = SentimentAwarePPOAgent(
                env=env,
                config=training_config,
                policy_type='standard',  # Simple for annual RL
                value_type='standard',
                encoder_type='simple',
                device=self.config.device,
                sentiment_enabled=False
            )
            
            # Train
            start_time = time.time()
            history = []
            for iteration in range(self.config.num_iterations):
                metrics = agent.train_iteration()
                history.append(metrics)
            train_time = time.time() - start_time
            
            self.pure_rl_agents_annual[num_goals] = agent
            
            # Save model
            model_dir = self.output_dir / "pure_rl_annual_models" / f"goals_{num_goals}"
            model_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(model_dir / "model.pth"))
            
            mean_reward = np.mean([h.get('mean_episode_reward', 0) for h in history[-3:]]) if history else 0
            logger.info(f"  Trained in {train_time:.1f}s, mean reward: {mean_reward:.2f}")
            
            env.close()

    def _simulate_pure_rl_with_annual_shocks(self, annual_shocks: np.ndarray, result_key: str = "Pure RL (Annual)"):
        """Simulate Pure RL agents with annual time steps and shocks"""
        dt = 1.0  # Annual time step
        
        for num_goals in self.config.goal_counts:
            # Check cache first - skip simulation if cached results exist
            if self._should_skip_simulation(result_key, num_goals):
                logger.info(f"Skipping {result_key} simulation for {num_goals} goals (cached results found)")
                continue
            if num_goals not in self.pure_rl_agents_annual:
                continue
                
            agent = self.pure_rl_agents_annual[num_goals]
            agent.policy_net.eval()
            
            initial_wealth = get_initial_wealth(num_goals)
            goal_years = get_goal_years(num_goals)
            
            logger.info(f"\nSimulating Pure RL (Annual) for {num_goals} goals...")
            
            rewards = []
            goal_successes = []
            final_wealths = []
            
            for sim_idx in tqdm(range(self.config.num_simulations), desc=f"Pure RL Annual {num_goals}g", leave=False):
                if sim_idx % 10000 == 0 and sim_idx > 0:
                    logger.info(f"  Pure RL Annual {num_goals} goals: {sim_idx:,}/{self.config.num_simulations:,}")
                
                sim_annual_shocks = annual_shocks[sim_idx]
                wealth = initial_wealth
                total_reward = 0
                goals_taken = []
                
                for year in range(self.config.years_horizon):
                    # Build 2D state for annual environment
                    normalized_time = year / self.config.years_horizon
                    normalized_wealth = min(wealth / (initial_wealth * 10), 1.0)
                    state = np.array([normalized_time, normalized_wealth], dtype=np.float32)
                    
                    # Get action from Pure RL agent
                    with torch.no_grad():
                        action = agent.predict(state, deterministic=True)
                    
                    goal_action = int(action[0])
                    portfolio_action = int(action[1])
                    
                    # Check for goal
                    if (year + 1) in goal_years:
                        goal_cost = get_goal_cost(year + 1)
                        goal_utility = get_goal_utility(year + 1)
                        
                        if goal_action == 1 and wealth >= goal_cost:
                            total_reward += goal_utility
                            wealth -= goal_cost
                            goals_taken.append(year + 1)
                    
                    # Wealth evolution (stable market, no VIX)
                    if year < self.config.years_horizon - 1:
                        shock = sim_annual_shocks[year]
                        
                        mu = self.portfolio_means_annual[portfolio_action]
                        sigma = self.portfolio_stds_annual[portfolio_action]
                        
                        # Annual GBM evolution
                        drift = (mu - 0.5 * sigma**2) * dt
                        diffusion = sigma * np.sqrt(dt) * shock
                        wealth = wealth * np.exp(drift + diffusion)
                        wealth = max(0, wealth)
                
                rewards.append(total_reward)
                goal_successes.append(len(goals_taken) / len(goal_years))
                final_wealths.append(wealth)
            
            # Calculate efficiency vs DP (Annual) - annual methods should compare to annual DP baseline
            dp_annual_key = "DP (Annual)"
            
            # Check if DP results exist for this goal count
            if dp_annual_key in self.results and num_goals in self.results[dp_annual_key]:
                dp_mean_reward = self.results[dp_annual_key][num_goals].mean_reward
                efficiency = np.mean(rewards) / dp_mean_reward if dp_mean_reward > 0 else 0.0
            else:
                logger.warning(f"DP results not found for {num_goals} goals in {dp_annual_key}. Setting efficiency to 0.0")
                efficiency = 0.0
            
            result = SimulationResult(
                method_name=result_key,
                num_goals=num_goals,
                mean_reward=np.mean(rewards),
                std_reward=np.std(rewards),
                efficiency=efficiency,
                num_simulations=self.config.num_simulations,
                goal_success_rate=np.mean(goal_successes),
                mean_final_wealth=np.mean(final_wealths),
                std_final_wealth=np.std(final_wealths),
                mean_vix_sentiment=0.0  # No VIX
            )
            
            self.results[result_key][num_goals] = result
            
            # Cache the simulation result
            self._save_simulation_result(result_key, num_goals, result)
            
            logger.info(f"  Pure RL Annual {num_goals} goals: reward={np.mean(rewards):.2f}, efficiency={efficiency:.3f}")

    def _simulate_dp_annual(self, annual_shocks: np.ndarray, result_key: str):
        """Alias for _simulate_dp_with_annual_shocks for VIX state analysis mode"""
        return self._simulate_dp_with_annual_shocks(annual_shocks, result_key)
    
    def _train_pure_rl_annual(self):
        """Alias for _train_pure_rl_with_annual_env for VIX state analysis mode"""
        return self._train_pure_rl_with_annual_env()
    
    def _simulate_pure_rl_annual(self, annual_shocks: np.ndarray, result_key: str):
        """Alias for _simulate_pure_rl_with_annual_shocks for VIX state analysis mode"""
        return self._simulate_pure_rl_with_annual_shocks(annual_shocks, result_key)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sentiment RL Evaluation with Monte Carlo Simulations')

    # Simulation settings
    parser.add_argument('--num_simulations', type=int, default=1000000,
                        help='Number of Monte Carlo simulations (default: 1000000)')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='Number of RL training iterations per goal count (default: 10)')
    parser.add_argument('--batch_size', type=int, default=4800,
                        help='Trajectories per PPO update (default: 4800)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for neural networks (default: 128)')

    # Sentiment RL architecture
    parser.add_argument('--policy_type', type=str, default='hierarchical',
                        choices=['standard', 'hierarchical'],
                        help='Sentiment RL policy network type (default: hierarchical)')
    parser.add_argument('--value_type', type=str, default='dual_head',
                        choices=['standard', 'dual_head', 'ensemble'],
                        help='Sentiment RL value network type (default: dual_head)')
    parser.add_argument('--encoder_type', type=str, default='attention',
                        choices=['feature', 'simple', 'adaptive', 'attention'],
                        help='Sentiment RL encoder type (default: attention)')

    # Pure RL architecture (defaults to same as Sentiment RL if not specified)
    parser.add_argument('--pure_rl_policy_type', type=str, default=None,
                        choices=['standard', 'hierarchical'],
                        help='Pure RL policy type (default: same as --policy_type)')
    parser.add_argument('--pure_rl_value_type', type=str, default=None,
                        choices=['standard', 'dual_head', 'ensemble'],
                        help='Pure RL value type (default: same as --value_type)')
    parser.add_argument('--pure_rl_encoder_type', type=str, default=None,
                        choices=['feature', 'simple', 'adaptive', 'attention'],
                        help='Pure RL encoder type (default: same as --encoder_type)')

    # Baseline comparison mode
    parser.add_argument('--baseline_mode', type=str, default='annual_stable',
                        choices=['annual_stable', 'monthly_vix'],
                        help='Comparison mode: annual_stable (Pure RL annual vs Sentiment RL monthly), '
                             'monthly_vix (all methods at monthly granularity) (default: annual_stable)')

    # Other settings
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated if not provided)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--goal_counts', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                        help='Goal counts to evaluate (default: 1 2 4 8 16)')
    parser.add_argument('--use_real_ef', action='store_true',
                        help='Use real historical data for efficient frontier (default: use paper values)')

    # Delta adjustment settings
    parser.add_argument('--use_delta_adjustment', action='store_true', default=True,
                        help='Enable δ (volatility) adjustment (default: True)')
    parser.add_argument('--no_delta_adjustment', action='store_true',
                        help='Disable δ adjustment (only use μ adjustment)')
    parser.add_argument('--volatility_method', type=str, default='rolling_vol',
                        choices=['rolling_vol', 'return_squared'],
                        help='Method for δ regression: rolling_vol (3-month forward vol) '
                             'or return_squared (R² as variance proxy) (default: rolling_vol)')
    parser.add_argument('--force_recompute', action='store_true',
                        help='Force fresh β/δ calculation (ignore cache)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable caching of β/δ parameters')

    # DP market configuration
    parser.add_argument('--dp_vix_market', action='store_true',
                        help='DP faces VIX-adjusted market (same as Sentiment RL). '
                             'Default: DP uses stable market.')
    
    # VIX model configuration
    parser.add_argument('--vix_model_type', type=str, default='mrjd',
                        choices=['mrjd', 'regime_switching'],
                        help='VIX model type: mrjd (Mean-Reverting Jump-Diffusion) '
                             'or regime_switching (Markov Regime-Switching) (default: mrjd)')
    
    # Resume functionality
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training phase and resume from simulation only')
    
    # Cache management
    parser.add_argument('--clear_cache', action='store_true',
                        help='Clear all cached simulation results before running')
    parser.add_argument('--cache_info', action='store_true',
                        help='Display information about cached results and exit')

    args = parser.parse_args()

    # Handle --no_delta_adjustment flag
    use_delta_adjustment = not args.no_delta_adjustment

    # Handle cache flags
    use_cache = not args.no_cache
    force_recompute = args.force_recompute
    
    # Auto-detect device
    device = "auto"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
        print("No GPU detected, using CPU")

    config = SentimentEvaluationConfig(
        goal_counts=args.goal_counts,
        num_simulations=args.num_simulations,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        baseline_mode=args.baseline_mode,
        policy_type=args.policy_type,
        value_type=args.value_type,
        encoder_type=args.encoder_type,
        pure_rl_policy_type=args.pure_rl_policy_type,
        pure_rl_value_type=args.pure_rl_value_type,
        pure_rl_encoder_type=args.pure_rl_encoder_type,
        random_seed=args.seed,
        output_dir=args.output_dir,
        use_real_ef=args.use_real_ef,
        use_delta_adjustment=use_delta_adjustment,
        volatility_method=args.volatility_method,
        use_cache=use_cache,
        force_recompute=force_recompute,
        dp_vix_market=args.dp_vix_market,
        vix_model_type=args.vix_model_type,
        device=device,
        skip_training=args.skip_training
    )

    logger.info("=" * 70)
    logger.info("SENTIMENT RL EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Baseline mode: {config.baseline_mode}")
    if config.baseline_mode == "annual_stable":
        logger.info("  - Pure RL: annual (16 steps), stable market")
        logger.info("  - Sentiment RL: monthly (192 steps), stable market")
    else:
        logger.info("  - All methods: monthly (192 steps)")
        logger.info("  - DP uses stable market")
    logger.info(f"Goal counts: {config.goal_counts}")
    logger.info(f"Simulations: {config.num_simulations:,}")
    logger.info(f"Training iterations: {config.num_iterations}")
    logger.info(f"Time horizon: {config.time_horizon} monthly steps")
    logger.info(f"Use real EF: {config.use_real_ef}")
    logger.info(f"Volatility method: {config.volatility_method}")
    logger.info(f"Use δ adjustment: {config.use_delta_adjustment}")
    logger.info("")
    logger.info("Sentiment RL Architecture:")
    logger.info(f"  Policy: {config.policy_type}, Value: {config.value_type}, Encoder: {config.encoder_type}")
    logger.info("Pure RL Architecture:")
    logger.info(f"  Policy: {config.pure_rl_policy_type}, Value: {config.pure_rl_value_type}, Encoder: {config.pure_rl_encoder_type}")
    logger.info("")
    logger.info(f"Output: {config.output_dir}")
    logger.info("=" * 70)

    evaluator = SentimentRLEvaluator(config)
    
    # Handle cache management options
    if args.cache_info:
        evaluator._show_cache_info()
        return
    
    if args.clear_cache:
        evaluator._clear_simulation_cache()
        logger.info("Cache cleared. Continuing with evaluation...")
    
    evaluator.run()


if __name__ == '__main__':
    main()
