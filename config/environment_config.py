"""Environment configuration for GBWM"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
from itertools import product


@dataclass
class GoalConfig:
    """Configuration for financial goals"""

    # Goal timing (which years goals are available)
    goal_years: List[int] = None

    # Cost function parameters: C(t) = base_cost * growth_rate^t
    # Note: base_goal_cost = 100000 gives realistic costs matching paper scale
    # At year 16: 100000 * 1.08^16 ≈ $342,594
    base_goal_cost: float = 100000.0
    goal_cost_growth_rate: float = 1.08  # 8% annual growth

    # Utility function parameters: U(t) = base_utility + t
    base_utility: float = 10.0
    utility_time_bonus: float = 1.0  # Linear increase with time

    def __post_init__(self):
        if self.goal_years is None:
            # Default: 4 goals at years 4, 8, 12, 16
            self.goal_years = [4, 8, 12, 16]

    def get_goal_cost(self, time: int) -> float:
        """Calculate goal cost at given time"""
        return self.base_goal_cost * (self.goal_cost_growth_rate ** time)

    def get_goal_utility(self, time: int) -> float:
        """Calculate goal utility at given time"""
        return self.base_utility + self.utility_time_bonus * time


# =============================================================================
# Extended Goal Configuration (Concurrent & Partial Goals Support)

# =============================================================================

@dataclass
class GoalOption:
    """
    A single fulfillment option for a goal.

    Each goal can have multiple options including skip (cost=0, utility=0),
    partial fulfillment options, and full fulfillment.

    Example (Car goal):
        - skip: cost=0, utility=0
        - basic: cost=28000, utility=80
        - hybrid: cost=32000, utility=125
        - full: cost=50000, utility=300
    """
    cost: float           # Cost to fulfill this option
    utility: float        # Utility gained from this option
    label: str = ""       # Human-readable label (e.g., "full", "partial", "skip")

    def __post_init__(self):
        if self.cost < 0:
            raise ValueError(f"Goal option cost cannot be negative: {self.cost}")
        if self.utility < 0:
            raise ValueError(f"Goal option utility cannot be negative: {self.utility}")


@dataclass
class GoalDefinition:
    """
    Definition of a single goal with its fulfillment options.

    Supports partial goals where the investor can choose from multiple
    levels of fulfillment, each with different costs and utilities.

    The options list MUST include a skip option (cost=0, utility=0).

    Example:
        GoalDefinition(
            name="Car",
            year=8,
            options=[
                GoalOption(cost=0, utility=0, label="skip"),
                GoalOption(cost=28000, utility=80, label="basic"),
                GoalOption(cost=50000, utility=300, label="full"),
            ]
        )
    """
    name: str                       # Goal name (e.g., "Car", "College")
    year: int                       # Time period when goal is available
    options: List[GoalOption] = None  # List of fulfillment options

    def __post_init__(self):
        if self.options is None:
            # Default: all-or-nothing goal with cost=100000, utility=10+year
            self.options = [
                GoalOption(cost=0, utility=0, label="skip"),
                GoalOption(cost=100000 * (1.08 ** self.year),
                          utility=10 + self.year, label="full"),
            ]

        # Validate options
        if not self.options:
            raise ValueError(f"Goal '{self.name}' must have at least one option")

        # Ensure skip option exists
        has_skip = any(opt.cost == 0 and opt.utility == 0 for opt in self.options)
        if not has_skip:
            raise ValueError(f"Goal '{self.name}' must have a skip option (cost=0, utility=0)")

        # Sort options by cost for consistency
        self.options = sorted(self.options, key=lambda x: x.cost)


@dataclass
class ExtendedGoalConfig:
    """
    Extended goal configuration supporting concurrent and partial goals.

    This follows the paper's formulation where:
    - Multiple goals can occur at the same time period (concurrent goals)
    - Each goal can have multiple fulfillment options (partial goals)
    - The algorithm generates cost/utility vectors by combining all options

    Example with concurrent goals at year 8:
        ExtendedGoalConfig(
            goals=[
                GoalDefinition(name="Car", year=8, options=[...]),
                GoalDefinition(name="Tuition", year=8, options=[...]),  # Concurrent!
                GoalDefinition(name="Vacation", year=5, options=[...]),
            ],
            time_horizon=16
        )
    """
    goals: List[GoalDefinition] = None
    time_horizon: int = 16

    def __post_init__(self):
        if self.goals is None:
            self.goals = []

        # Validate all goals are within time horizon
        for goal in self.goals:
            if goal.year < 0 or goal.year >= self.time_horizon:
                raise ValueError(
                    f"Goal '{goal.name}' at year {goal.year} is outside "
                    f"time horizon [0, {self.time_horizon})"
                )

    def get_goals_at_time(self, t: int) -> List[GoalDefinition]:
        """Get all goals available at time t"""
        return [g for g in self.goals if g.year == t]

    def get_all_goal_years(self) -> List[int]:
        """Get sorted list of unique years with goals"""
        return sorted(set(g.year for g in self.goals))

    @classmethod
    def from_legacy(cls, goal_config: GoalConfig, time_horizon: int = 16) -> 'ExtendedGoalConfig':
        """
        Convert legacy GoalConfig to ExtendedGoalConfig for backward compatibility.

        Creates single-option (all-or-nothing) goals matching the legacy behavior.
        """
        goals = []
        for year in goal_config.goal_years:
            cost = goal_config.get_goal_cost(year)
            utility = goal_config.get_goal_utility(year)
            goals.append(GoalDefinition(
                name=f"Goal_Year{year}",
                year=year,
                options=[
                    GoalOption(cost=0, utility=0, label="skip"),
                    GoalOption(cost=cost, utility=utility, label="full"),
                ]
            ))
        return cls(goals=goals, time_horizon=time_horizon)

    @classmethod
    def create_paper_example_section23(cls) -> 'ExtendedGoalConfig':
        """
        Create the example from paper Section 2.3: Three concurrent goals at year 5.

        Goal 1: All-or-nothing (cost 0 or 7, utility 0 or 100)
        Goal 2: Partial (cost 0, 9, or 20; utility 0, 90, or 300)
        Goal 3: Partial (cost 0, 10, 20, 30, or 40; utility 0, 40, 250, 400, or 500)
        """
        goal1 = GoalDefinition(
            name="Goal1",
            year=5,
            options=[
                GoalOption(cost=0, utility=0, label="skip"),
                GoalOption(cost=7, utility=100, label="full"),
            ]
        )

        goal2 = GoalDefinition(
            name="Goal2",
            year=5,  # Concurrent with Goal1
            options=[
                GoalOption(cost=0, utility=0, label="skip"),
                GoalOption(cost=9, utility=90, label="partial"),
                GoalOption(cost=20, utility=300, label="full"),
            ]
        )

        goal3 = GoalDefinition(
            name="Goal3",
            year=5,  # Concurrent with Goal1 and Goal2
            options=[
                GoalOption(cost=0, utility=0, label="skip"),
                GoalOption(cost=10, utility=40, label="partial1"),
                GoalOption(cost=20, utility=250, label="partial2"),
                GoalOption(cost=30, utility=400, label="partial3"),
                GoalOption(cost=40, utility=500, label="full"),
            ]
        )

        return cls(goals=[goal1, goal2, goal3], time_horizon=11)


@dataclass
class PortfolioConfig:
    """
    Configuration for investment portfolios.

    Uses the unified get_portfolio_arrays() function from src/data/efficient_frontier.py
    to ensure consistent portfolio parameters across DP, RL, and Sentiment RL.
    """

    # Portfolio returns (from paper - efficient frontier)
    mean_returns: np.ndarray = None
    return_stds: np.ndarray = None
    correlation_matrix: np.ndarray = None

    # Portfolio weights (for each of 15 portfolios: [bonds, us_stocks, intl_stocks])
    portfolio_weights: np.ndarray = None

    # Asset parameters (US bonds, US stocks, International stocks)
    # These are fallback values from the paper
    asset_means: Tuple[float, float, float] = (0.0493, 0.0770, 0.0886)
    asset_stds: Tuple[float, float, float] = (0.0412, 0.1990, 0.1978)
    asset_correlations: Tuple[Tuple[float, float, float], ...] = (
        (1.0, -0.2077, -0.2685),
        (-0.2077, 1.0, 0.7866),
        (-0.2685, 0.7866, 1.0)
    )

    # Whether to use real historical data for efficient frontier
    use_real_data: bool = False

    # Number of portfolios on efficient frontier
    num_portfolios: int = 15

    def __post_init__(self):
        # Use unified portfolio loading from src/data/efficient_frontier.py
        self._load_portfolio_arrays()

        if self.correlation_matrix is None:
            self.correlation_matrix = np.array(self.asset_correlations)

        if self.portfolio_weights is None:
            # Default weights: linear interpolation from conservative to aggressive
            self.portfolio_weights = self._compute_default_weights()

    def _load_portfolio_arrays(self):
        """
        Load portfolio arrays using the unified get_portfolio_arrays() function.

        This is the single source of truth for portfolio parameters used by:
        - Dynamic Programming (DP)
        - Reinforcement Learning (RL)
        - Sentiment RL
        """
        try:
            from src.data.efficient_frontier import get_portfolio_arrays

            self.mean_returns, self.return_stds = get_portfolio_arrays(
                use_real_ef=self.use_real_data,
                num_portfolios=self.num_portfolios
            )

        except ImportError:
            # Fallback if efficient_frontier module not available
            self._use_fallback_portfolios()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Failed to load portfolio arrays: {e}, using fallback"
            )
            self._use_fallback_portfolios()

    def _use_fallback_portfolios(self):
        """Fallback portfolio arrays if main loading fails"""
        # Use paper's hardcoded values
        self.mean_returns = np.linspace(0.052632, 0.088636, self.num_portfolios)
        self.return_stds = np.linspace(0.037351, 0.195437, self.num_portfolios)

    def _compute_default_weights(self) -> np.ndarray:
        """Compute default portfolio weights using linear interpolation."""
        weights = np.zeros((self.num_portfolios, 3))
        for i in range(self.num_portfolios):
            t = i / (self.num_portfolios - 1)
            weights[i, 0] = 0.85 * (1 - t) + 0.05 * t  # Bonds: 85% → 5%
            weights[i, 1] = 0.10 * (1 - t) + 0.50 * t  # US Stocks: 10% → 50%
            weights[i, 2] = 0.05 * (1 - t) + 0.45 * t  # Intl Stocks: 5% → 45%
        return weights


@dataclass
class EnvironmentConfig:
    """Complete environment configuration"""

    # Time settings
    time_horizon: int = 16
    dt: float = 1.0  # Time step (1 year)

    # Wealth settings
    initial_wealth: float = 120000.0
    max_wealth: float = 10000000.0  # For normalization

    # Goal and portfolio configurations
    goal_config: GoalConfig = None
    portfolio_config: PortfolioConfig = None

    # Environment parameters
    random_seed: int = 42

    # Historical data configuration
    data_mode: str = "simulation"  # "simulation" or "historical"
    historical_data_path: str = "data/raw/market_data/"
    processed_data_path: str = "data/processed/"
    
    # Historical data parameters
    min_sequence_length: int = 200  # Minimum time periods needed for training
    historical_validation_split: float = 0.2  # Reserve 20% for validation
    historical_start_date: str = "1970-01-01"  # Default start date for historical data
    historical_end_date: str = "2023-12-31"    # Default end date for historical data
    
    # Data augmentation options
    use_data_augmentation: bool = True  # Enable data augmentation techniques
    augmentation_noise_std: float = 0.01  # Standard deviation for noise injection
    augmentation_return_scaling: float = 0.05  # Scaling factor for return perturbation
    
    # Historical data validation
    allow_missing_data: bool = True  # Allow some missing values in historical data
    max_missing_ratio: float = 0.05  # Maximum allowed ratio of missing data
    interpolate_missing: bool = True  # Interpolate missing values
    
    # Logging and debugging
    log_historical_stats: bool = False  # Log detailed historical data statistics
    save_historical_sequences: bool = False  # Save used sequences for debugging

    def __post_init__(self):
        if self.goal_config is None:
            self.goal_config = GoalConfig()
        if self.portfolio_config is None:
            self.portfolio_config = PortfolioConfig()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate environment configuration parameters"""
        if self.data_mode not in ["simulation", "historical"]:
            raise ValueError(f"Invalid data_mode: {self.data_mode}. Must be 'simulation' or 'historical'")
        
        if self.time_horizon <= 0:
            raise ValueError(f"time_horizon must be positive, got {self.time_horizon}")
        
        if self.min_sequence_length < self.time_horizon:
            raise ValueError(f"min_sequence_length ({self.min_sequence_length}) must be >= time_horizon ({self.time_horizon})")
        
        if not (0.0 <= self.historical_validation_split <= 1.0):
            raise ValueError(f"historical_validation_split must be in [0,1], got {self.historical_validation_split}")
        
        if not (0.0 <= self.max_missing_ratio <= 1.0):
            raise ValueError(f"max_missing_ratio must be in [0,1], got {self.max_missing_ratio}")
    
    def get_historical_config(self) -> dict:
        """Get historical data configuration as dictionary"""
        return {
            'data_path': self.historical_data_path,
            'processed_path': self.processed_data_path,
            'start_date': self.historical_start_date,
            'end_date': self.historical_end_date,
            'min_sequence_length': self.min_sequence_length,
            'validation_split': self.historical_validation_split,
            'use_augmentation': self.use_data_augmentation,
            'noise_std': self.augmentation_noise_std,
            'return_scaling': self.augmentation_return_scaling,
            'allow_missing': self.allow_missing_data,
            'max_missing_ratio': self.max_missing_ratio,
            'interpolate_missing': self.interpolate_missing
        }


# Helper function to create historical environment config
def create_historical_config(num_goals: int = 4,
                            initial_wealth: float = None,
                            historical_data_path: str = "data/raw/market_data/",
                            start_date: str = "2010-01-01",
                            end_date: str = "2023-12-31",
                            **kwargs) -> EnvironmentConfig:
    """
    Create environment configuration for historical data mode
    
    Args:
        num_goals: Number of financial goals
        initial_wealth: Initial wealth (calculated if None)
        historical_data_path: Path to historical market data
        start_date: Start date for historical data
        end_date: End date for historical data
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured EnvironmentConfig for historical mode
    """
    config = EnvironmentConfig(
        data_mode="historical",
        historical_data_path=historical_data_path,
        historical_start_date=start_date,
        historical_end_date=end_date
    )
    
    # Set goal schedule based on number of goals
    if num_goals == 1:
        config.goal_config.goal_years = [16]
    elif num_goals == 2:
        config.goal_config.goal_years = [8, 16]
    elif num_goals == 4:
        config.goal_config.goal_years = [4, 8, 12, 16]
    elif num_goals == 8:
        config.goal_config.goal_years = [2, 4, 6, 8, 10, 12, 14, 16]
    elif num_goals == 16:
        config.goal_config.goal_years = list(range(1, 17))
    else:
        raise ValueError(f"Unsupported number of goals: {num_goals}")
    
    # Set initial wealth
    if initial_wealth is not None:
        config.initial_wealth = initial_wealth
    else:
        # Use paper formula: W0 = 12 * (NG)^0.85 * 10000
        config.initial_wealth = 12 * (num_goals ** 0.85) * 10000
    
    # Apply additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def create_simulation_config(num_goals: int = 4,
                           initial_wealth: float = None,
                           **kwargs) -> EnvironmentConfig:
    """
    Create environment configuration for simulation mode (default behavior)
    
    Args:
        num_goals: Number of financial goals
        initial_wealth: Initial wealth (calculated if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured EnvironmentConfig for simulation mode
    """
    config = EnvironmentConfig(data_mode="simulation")
    
    # Set goal schedule based on number of goals
    if num_goals == 1:
        config.goal_config.goal_years = [16]
    elif num_goals == 2:
        config.goal_config.goal_years = [8, 16]
    elif num_goals == 4:
        config.goal_config.goal_years = [4, 8, 12, 16]
    elif num_goals == 8:
        config.goal_config.goal_years = [2, 4, 6, 8, 10, 12, 14, 16]
    elif num_goals == 16:
        config.goal_config.goal_years = list(range(1, 17))
    else:
        raise ValueError(f"Unsupported number of goals: {num_goals}")
    
    # Set initial wealth
    if initial_wealth is not None:
        config.initial_wealth = initial_wealth
    else:
        # Use paper formula: W0 = 12 * (NG)^0.85 * 10000
        config.initial_wealth = 12 * (num_goals ** 0.85) * 10000
    
    # Apply additional parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


# Default environment configuration (simulation mode)
DEFAULT_ENV_CONFIG = EnvironmentConfig()